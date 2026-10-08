"""StructuredOutputParser helpers and strict fact normalization."""

import json
import re
from typing import Any

from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.exceptions import OutputParserException

from backend.app.models import DecisionItem, MeetingAnalysisResponse, TaskItem


class ModelOutputError(ValueError):
    """Raised when model output cannot be safely parsed or validated."""


def build_meeting_parser() -> StructuredOutputParser:
    schemas = [
        ResponseSchema(name="Meeting Title", description="A short title present in or directly supported by the transcript."),
        ResponseSchema(name="Meeting Summary", description="An executive summary using only explicit transcript facts."),
        ResponseSchema(name="Decisions", description="List of objects with decision and context. Include only explicit decisions."),
        ResponseSchema(name="Tasks", description="List of objects with task, Owner, Deadline, Priority. Use null for any unstated owner, deadline, or priority."),
        ResponseSchema(name="Key points", description="List of important explicitly stated meeting points."),
        ResponseSchema(name="Open questions", description="List of explicitly raised unresolved questions."),
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
    raw_decisions = extracted.get("Decisions")
    raw_tasks = extracted.get("Tasks")
    decisions: list[DecisionItem] = []
    tasks: list[TaskItem] = []

    for item in raw_decisions if isinstance(raw_decisions, list) else []:
        if isinstance(item, str):
            item = {"decision": item}
        if not isinstance(item, dict) or not str(item.get("decision") or "").strip():
            continue
        decision = DecisionItem(
            decision=str(item["decision"]).strip(),
            context=str(item.get("context") or "").strip(),
            source_ids=[chunk_id],
            source_excerpt=source_excerpt,
        )
        decisions.append(decision)

    for item in raw_tasks if isinstance(raw_tasks, list) else []:
        if isinstance(item, str):
            item = {"task": item}
        if not isinstance(item, dict) or not str(item.get("task") or "").strip():
            continue
        try:
            task = TaskItem(
                task=str(item["task"]).strip(),
                owner=item.get("Owner", item.get("owner")),
                deadline=item.get("Deadline", item.get("deadline")),
                priority=item.get("Priority", item.get("priority")),
                source_ids=[chunk_id],
                source_excerpt=source_excerpt,
            )
        except ValueError:
            task = TaskItem(
                task=str(item["task"]).strip(),
                owner=item.get("Owner", item.get("owner")),
                deadline=item.get("Deadline", item.get("deadline")),
                priority=None,
                source_ids=[chunk_id],
                source_excerpt=source_excerpt,
            )
        tasks.append(task)

    return {
        "title": str(extracted.get("Meeting Title") or "").strip(),
        "summary": str(extracted.get("Meeting Summary") or "").strip(),
        "decisions": decisions,
        "tasks": tasks,
        "key_points": _string_list(extracted.get("Key points")),
        "open_questions": _string_list(extracted.get("Open questions")),
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


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
