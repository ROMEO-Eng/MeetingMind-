import json

import pytest
from langchain_core.exceptions import OutputParserException

from backend.app.models import DecisionItem, TaskItem
from backend.app.parsers.structured import (
    ModelOutputError,
    build_meeting_parser,
    deduplicate_records,
    normalize_extraction,
    safe_parse,
)


class MissingFenceParser:
    def parse(self, text: str):
        raise OutputParserException("The model omitted the expected format.")


def test_safe_parse_accepts_fenced_json_and_fallback_json() -> None:
    expected = {"Tasks": [{"task": "Send report", "Owner": None}]}
    assert safe_parse(f"```json\n{json.dumps(expected)}\n```") == expected
    assert safe_parse(f"prefix {json.dumps(expected)} suffix", MissingFenceParser()) == expected


def test_safe_parse_rejects_malformed_output() -> None:
    with pytest.raises(ModelOutputError, match="invalid JSON"):
        safe_parse("```json\n{broken\n```")


def test_meeting_parser_declares_collection_fields_as_arrays() -> None:
    instructions = build_meeting_parser().get_format_instructions()

    for field in ("Decisions", "Tasks", "Key points", "Open questions"):
        assert f'"{field}": array' in instructions


def test_qwen_style_fenced_json_normalizes_all_analysis_collections() -> None:
    model_output = {
        "meeting_title": "Upcoming release",
        "meeting_summary": "The team reviewed the upcoming release.",
        "Decisions": [
            {
                "decision": "Release version 2.1 next Monday.",
                "context": "The team agreed on the release date.",
            }
        ],
        "Tasks": [
            {
                "task": "Prepare the launch checklist",
                "owner": "Alex",
                "deadline": "Friday",
                "priority": None,
            },
            {
                "task": "Coordinate the final UI review with the design team",
                "owner": "Sarah",
                "deadline": None,
                "priority": None,
            },
        ],
        "key_points": [
            "The team reviewed the upcoming release.",
            "The team will review the release status again next week.",
        ],
        "open_questions": ["The marketing campaign decision remains unresolved."],
    }
    raw = f"```json\n{json.dumps(model_output)}\n```"

    parsed = safe_parse(raw, build_meeting_parser())
    normalized = normalize_extraction(
        parsed,
        "chunk-001",
        "The team agreed to release version 2.1 next Monday.",
    )

    assert [item.decision for item in normalized["decisions"]] == [
        "Release version 2.1 next Monday."
    ]
    assert [
        (task.task, task.owner, task.deadline)
        for task in normalized["tasks"]
    ] == [
        ("Prepare the launch checklist", "Alex", "Friday"),
        ("Coordinate the final UI review with the design team", "Sarah", None),
    ]
    assert normalized["key_points"] == model_output["key_points"]
    assert normalized["open_questions"] == model_output["open_questions"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("high", "High"), ("Medium", "Medium"), ("LOW", "Low"), (None, None), ("not specified", None)],
)
def test_priority_validation_normalizes_allowed_values(value, expected) -> None:
    task = TaskItem(task="Follow up", priority=value)
    assert task.priority == expected


def test_invalid_priority_fails_validation() -> None:
    with pytest.raises(ValueError, match="Priority must be"):
        TaskItem(task="Follow up", priority="Urgent")


def test_normalize_extraction_keeps_missing_owner_and_deadline_null() -> None:
    normalized = normalize_extraction(
        {
            "Meeting Title": "Legal approval",
            "Meeting Summary": "The team needs legal approval.",
            "Decisions": [],
            "Tasks": [{"task": "Get legal approval", "Owner": "Not specified"}],
            "Key points": [],
            "Open questions": [],
        },
        "chunk-001",
        "No person was assigned to get legal approval.",
    )

    task = normalized["tasks"][0]
    assert task.owner is None
    assert task.deadline is None
    assert task.priority is None
    assert task.source_ids == ["chunk-001"]


def test_normalize_extraction_maps_colab_case_and_separator_variants() -> None:
    model_output = {
        "meeting_title": "Upcoming release",
        "meeting_summary": "The team reviewed the upcoming release.",
        "decisions": [
            {"Decision": "Release version 2.1 next Monday."},
        ],
        "tasks": [
            {
                "Task": "Prepare the launch checklist",
                "Owner": "Alex",
                "Deadline": "Friday",
            },
            {
                "Task": "Coordinate the final UI review with the design team",
                "Owner": "Sarah",
                "Deadline": None,
            },
        ],
        "Key Points": [
            "The team reviewed the upcoming release.",
            "The team will review the release status again next week.",
        ],
        "open_questions": [
            "The marketing campaign decision remains open.",
        ],
    }
    normalized = normalize_extraction(
        safe_parse(json.dumps(model_output), build_meeting_parser()),
        "chunk-001",
        "The product team met to review the upcoming release.",
    )

    assert [item.decision for item in normalized["decisions"]] == [
        "Release version 2.1 next Monday."
    ]
    assert [
        (task.task, task.owner, task.deadline)
        for task in normalized["tasks"]
    ] == [
        ("Prepare the launch checklist", "Alex", "Friday"),
        ("Coordinate the final UI review with the design team", "Sarah", None),
    ]
    assert normalized["key_points"] == [
        "The team reviewed the upcoming release.",
        "The team will review the release status again next week.",
    ]
    assert normalized["open_questions"] == [
        "The marketing campaign decision remains open."
    ]


def test_normalize_extraction_rejects_incomplete_model_output() -> None:
    with pytest.raises(ModelOutputError, match="missing required fields"):
        normalize_extraction(
            {"Meeting Summary": "The team reviewed the release."},
            "chunk-001",
            "The team reviewed the release.",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("Decisions", "Release version 2.1 next Monday.", "Meeting decisions must be a list"),
        ("Tasks", "Alex will prepare the checklist.", "Meeting tasks must be a list"),
        ("Key points", "The release is next Monday.", "Meeting key points must be a list"),
        ("Open questions", "What about marketing?", "Meeting open questions must be a list"),
    ],
)
def test_normalize_extraction_rejects_malformed_collection_fields(
    field: str,
    value: str,
    error: str,
) -> None:
    extracted = {
        "Meeting Title": "Upcoming release",
        "Meeting Summary": "The team reviewed the release.",
        "Decisions": [],
        "Tasks": [],
        "Key points": [],
        "Open questions": [],
    }
    extracted[field] = value

    with pytest.raises(ModelOutputError, match=error):
        normalize_extraction(
            extracted,
            "chunk-001",
            "The team agreed to release version 2.1 next Monday.",
        )


def test_deduplication_merges_sources_without_guessing_fields() -> None:
    first = TaskItem(task="Send launch notes", source_ids=["chunk-001"])
    second = TaskItem(task="Send launch notes", owner="Maya", source_ids=["chunk-002"])
    merged = deduplicate_records([first, second], "task")

    assert len(merged) == 1
    assert merged[0].owner == "Maya"
    assert merged[0].deadline is None
    assert merged[0].source_ids == ["chunk-001", "chunk-002"]


def test_decision_deduplication_retains_context_and_sources() -> None:
    decisions = [
        DecisionItem(decision="Keep current pricing", source_ids=["chunk-001"]),
        DecisionItem(
            decision="Keep current pricing",
            context="Review after launch.",
            source_ids=["chunk-002"],
        ),
    ]
    merged = deduplicate_records(decisions, "decision")

    assert len(merged) == 1
    assert merged[0].context == "Review after launch."
    assert merged[0].source_ids == ["chunk-001", "chunk-002"]
