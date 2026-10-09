"""StructuredOutputParser helpers and strict fact normalization."""

import json
import re
from typing import Any

from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.exceptions import OutputParserException

from backend.app.models import DecisionItem, TaskItem


class ModelOutputError(ValueError):
    """Raised when model output cannot be safely parsed or validated."""


_EXTRACTION_FIELD_ALIASES = {
    "meetingtitle": "title",
    "title": "title",
    "meetingsummary": "summary",
    "summary": "summary",
    "executivesummary": "summary",
    "decisions": "decisions",
    "decision": "decisions",
    "tasks": "tasks",
    "task": "tasks",
    "actionitems": "tasks",
    "keypoints": "key_points",
    "keypoint": "key_points",
    "openquestions": "open_questions",
    "openquestion": "open_questions",
    "unresolvedquestions": "open_questions",
}


def build_meeting_parser() -> StructuredOutputParser:
    schemas = [
        ResponseSchema(name="Meeting Title", description="A short title present in or directly supported by the transcript."),
        ResponseSchema(name="Meeting Summary", description="An executive summary using only explicit transcript facts."),
        ResponseSchema(
            name="Decisions",
            description="JSON array of objects with decision and context. Include only explicit decisions.",
            type="array",
        ),
        ResponseSchema(
            name="Tasks",
            description="JSON array of objects with task, Owner, Deadline, and Priority. Use null for any unstated value.",
            type="array",
        ),
        ResponseSchema(
            name="Key points",
            description="JSON array of important explicitly stated meeting points.",
            type="array",
        ),
        ResponseSchema(
            name="Open questions",
            description="JSON array of explicitly raised unresolved questions.",
            type="array",
        ),
    ]
    return StructuredOutputParser.from_response_schemas(schemas)


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise ModelOutputError("The model response did not contain a JSON object.")
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as error:
        raise ModelOutputError("The model returned invalid JSON. Please retry the analysis.") from error
    if not isinstance(value, dict):
        raise ModelOutputError("The model response must be a JSON object.")
    return value


def safe_parse(text: str, parser: StructuredOutputParser | None = None) -> dict[str, Any]:
    if parser is not None:
        try:
            value = parser.parse(text)
        except OutputParserException:
            value = extract_json_object(text)
    else:
        value = extract_json_object(text)

    if not isinstance(value, dict):
        raise ModelOutputError("The model response must be a JSON object.")
    return value


def normalize_extraction(
    extracted: dict[str, Any],
    chunk_id: str,
    source_excerpt: str,
) -> dict[str, Any]:
    fields = _canonicalize_extraction(extracted)
    raw_decisions = fields["decisions"]
    raw_tasks = fields["tasks"]
    decisions: list[DecisionItem] = []
    tasks: list[TaskItem] = []

    for item in raw_decisions:
        if isinstance(item, str):
            item = {"decision": item}
        if not isinstance(item, dict):
            raise ModelOutputError("Each decision must be a string or an object.")
        decision_text = _item_field(item, "decision")
        if not isinstance(decision_text, str) or not decision_text.strip():
            raise ModelOutputError("Each decision must include non-empty decision text.")
        context = _item_field(item, "context")
        if context is not None and not isinstance(context, str):
            raise ModelOutputError("Decision context must be a string or null.")
        decision = DecisionItem(
            decision=decision_text.strip(),
            context=(context or "").strip(),
            source_ids=[chunk_id],
            source_excerpt=source_excerpt,
        )
        decisions.append(decision)

    for item in raw_tasks:
        if isinstance(item, str):
            item = {"task": item}
        if not isinstance(item, dict):
            raise ModelOutputError("Each task must be a string or an object.")
        task_text = _item_field(item, "task")
        if not isinstance(task_text, str) or not task_text.strip():
            raise ModelOutputError("Each task must include non-empty task text.")
        owner = _item_field(item, "owner")
        deadline = _item_field(item, "deadline")
        priority = _item_field(item, "priority")
        for name, value in (
            ("owner", owner),
            ("deadline", deadline),
            ("priority", priority),
        ):
            if value is not None and not isinstance(value, str):
                raise ModelOutputError(f"Task {name} must be a string or null.")
        try:
            task = TaskItem(
                task=task_text.strip(),
                owner=owner,
                deadline=deadline,
                priority=priority,
                source_ids=[chunk_id],
                source_excerpt=source_excerpt,
            )
        except ValueError:
            task = TaskItem(
                task=task_text.strip(),
                owner=owner,
                deadline=deadline,
                priority=None,
                source_ids=[chunk_id],
                source_excerpt=source_excerpt,
            )
        tasks.append(task)

    return {
        "title": fields["title"].strip(),
        "summary": fields["summary"].strip(),
        "decisions": decisions,
        "tasks": tasks,
        "key_points": _string_list(fields["key_points"], "key points"),
        "open_questions": _string_list(fields["open_questions"], "open questions"),
    }


def _canonicalize_extraction(extracted: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, value in extracted.items():
        canonical_name = _EXTRACTION_FIELD_ALIASES.get(_normalize_field_name(name))
        if canonical_name is None:
            continue
        if canonical_name in fields:
            raise ModelOutputError(
                f"The model returned more than one value for {canonical_name.replace('_', ' ')}."
            )
        fields[canonical_name] = value

    required = ("title", "summary", "decisions", "tasks", "key_points", "open_questions")
    missing = [name.replace("_", " ") for name in required if name not in fields]
    if missing:
        raise ModelOutputError(
            f"The model output is missing required fields: {', '.join(missing)}."
        )
    for name in ("title", "summary"):
        if not isinstance(fields[name], str):
            raise ModelOutputError(f"Meeting {name} must be a string.")
    if not fields["summary"].strip():
        raise ModelOutputError("Meeting summary must not be empty.")
    for name in ("decisions", "tasks", "key_points", "open_questions"):
        if not isinstance(fields[name], list):
            raise ModelOutputError(f"Meeting {name.replace('_', ' ')} must be a list.")
    return fields


def _normalize_field_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _item_field(item: dict[str, Any], name: str) -> Any:
    normalized_name = _normalize_field_name(name)
    matches = [
        value
        for key, value in item.items()
        if _normalize_field_name(key) == normalized_name
    ]
    if len(matches) > 1:
        raise ModelOutputError(f"The model returned duplicate {name} fields.")
    return matches[0] if matches else None


def _string_list(value: list[Any], name: str) -> list[str]:
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ModelOutputError(f"Each {name} entry must be a non-empty string.")
    return list(dict.fromkeys(item.strip() for item in value))


def deduplicate_records(records: list[Any], identity_field: str) -> list[Any]:
    unique: list[Any] = []
    by_identity: dict[str, int] = {}
    for record in records:
        identity = str(getattr(record, identity_field, "")).strip().casefold()
        if not identity:
            continue
        if identity not in by_identity:
            by_identity[identity] = len(unique)
            unique.append(record)
            continue

        existing = unique[by_identity[identity]]
        sources = list(dict.fromkeys([*existing.source_ids, *record.source_ids]))
        existing.source_ids = sources
        if not existing.source_excerpt and record.source_excerpt:
            existing.source_excerpt = record.source_excerpt
        if isinstance(existing, TaskItem):
            if existing.owner is None and record.owner is not None:
                existing.owner = record.owner
            if existing.deadline is None and record.deadline is not None:
                existing.deadline = record.deadline
            if existing.priority is None and record.priority is not None:
                existing.priority = record.priority
        elif isinstance(existing, DecisionItem) and not existing.context and record.context:
            existing.context = record.context
    return unique
