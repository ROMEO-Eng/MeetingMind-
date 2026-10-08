import json

import pytest
from langchain_core.exceptions import OutputParserException

from backend.app.models import DecisionItem, TaskItem
from backend.app.parsers.structured import (
    ModelOutputError,
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
        {"Tasks": [{"task": "Get legal approval", "Owner": "Not specified"}]},
        "chunk-001",
        "No person was assigned to get legal approval.",
    )

    task = normalized["tasks"][0]
    assert task.owner is None
    assert task.deadline is None
    assert task.priority is None
    assert task.source_ids == ["chunk-001"]


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
