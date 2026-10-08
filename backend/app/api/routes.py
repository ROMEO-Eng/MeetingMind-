"""HTTP endpoints for meeting analysis, grounded QA, samples, and downloads."""

import csv
from io import StringIO
import logging
from pathlib import Path
import subprocess

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from backend.app.chains.errors import ModelUnavailableError
from backend.app.config import settings
from backend.app.loaders.transcripts import (
    TranscriptInputError,
    fetch_youtube_transcript,
    load_transcript_file,
    normalize_transcript,
)
from backend.app.models import (
    ErrorResponse,
    HealthResponse,
    MeetingAnalysisResponse,
    QuestionRequest,
    QuestionResponse,
)
from backend.app.parsers.structured import ModelOutputError
from backend.app.pipeline.sample import SAMPLE_ANALYSIS, SAMPLE_CHUNKS
from backend.app.pipeline.service import MeetingService, service

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def _service() -> MeetingService:
    return service


def _nvidia_gpu_detected() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.strip() for line in result.stdout.splitlines())


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return _nvidia_gpu_detected()
    return torch.cuda.is_available() or _nvidia_gpu_detected()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    gpu_available = _gpu_available()
    model_status, model_status_detail = _service().model_state
    return HealthResponse(
        status="ok" if gpu_available else "degraded",
        api_status="connected",
        model_status=model_status,
        model_status_detail=model_status_detail,
        model_ready=model_status == "ready",
        gpu_available=gpu_available,
        model_name=settings.model_name,
    )


@router.post(
    "/analyse",
    response_model=MeetingAnalysisResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def analyse(
    source_type: str = Form(...),
    youtube_url: str | None = Form(None),
    transcript: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> MeetingAnalysisResponse:
    try:
        if source_type == "youtube":
            if not youtube_url or not youtube_url.strip():
                raise TranscriptInputError("Enter a YouTube video link.")
            source_text = await run_in_threadpool(fetch_youtube_transcript, youtube_url)
            source_name = "YouTube transcript"
        elif source_type == "text":
            source_text = normalize_transcript(transcript or "")
            source_name = "Pasted transcript"
        elif source_type == "upload":
            if file is None:
                raise TranscriptInputError("Choose a transcript file to upload.")
            content = await file.read(settings.max_upload_bytes + 1)
            source_name = Path(file.filename or "meeting transcript").name
            source_text = load_transcript_file(
                source_name,
                content,
                settings.max_upload_bytes,
            )
        else:
            raise TranscriptInputError("Select YouTube, pasted transcript, or file upload as the source.")

        if not source_text:
            raise TranscriptInputError("The transcript is empty.")
        return await run_in_threadpool(_service().analyze, source_text, source_name)
    except TranscriptInputError as error:
        status_code = 413 if "too large" in str(error).casefold() else 422
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ModelOutputError as error:
        LOGGER.warning("Model output validation failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="The AI could not structure the meeting notes. Please retry the analysis.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Meeting analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Meeting analysis failed. Please check the backend and try again.",
        ) from error
    finally:
        if file is not None:
            await file.close()


@router.post(
    "/qa",
    response_model=QuestionResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def qa(request: QuestionRequest) -> QuestionResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Enter a question about the meeting.")
    try:
        return await run_in_threadpool(
            _service().answer,
            request.meeting_id,
            request.question,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Meeting Q&A failed")
        raise HTTPException(
            status_code=500,
            detail="The meeting question could not be answered. Please try again.",
        ) from error


@router.post("/sample", response_model=MeetingAnalysisResponse)
def sample() -> MeetingAnalysisResponse:
    _service().register_sample(SAMPLE_ANALYSIS, SAMPLE_CHUNKS)
    return SAMPLE_ANALYSIS


@router.get("/meetings/{meeting_id}/downloads")
def download_meeting(meeting_id: str, format: str = "csv") -> StreamingResponse:
    record = _service().get_record(meeting_id)
    if record is None:
        raise HTTPException(status_code=404, detail="This meeting session has expired.")
    if format not in {"csv", "markdown"}:
        raise HTTPException(status_code=422, detail="Choose csv or markdown as the download format.")
    if format == "csv":
        return _download_csv(record.analysis)
    return _download_markdown(record.analysis)


def _download_csv(analysis: MeetingAnalysisResponse) -> StreamingResponse:
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["Task", "Owner", "Deadline", "Priority", "Source"])
    for task in analysis.tasks:
        writer.writerow(
            [
                _spreadsheet_safe(task.task),
                _spreadsheet_safe(task.owner or ""),
                _spreadsheet_safe(task.deadline or ""),
                task.priority or "",
                ", ".join(task.source_ids),
            ]
        )
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="meetingmind-action-items.csv"'},
    )


def _download_markdown(analysis: MeetingAnalysisResponse) -> StreamingResponse:
    lines = [
        f"# {analysis.title}",
        "",
        "## Executive summary",
        "",
        analysis.executive_summary,
        "",
        "## Decisions",
        "",
    ]
    for decision in analysis.decisions:
        lines.extend(
            [
                f"- **{decision.decision}**",
                f"  - Context: {decision.context or 'Not stated'}",
                f"  - Source: {', '.join(decision.source_ids) or 'Not available'}",
            ]
        )
    lines.extend(["", "## Action items", "", "| Task | Owner | Deadline | Priority | Source |", "|---|---|---|---|---|"])
    for task in analysis.tasks:
        values = [
            task.task,
            task.owner or "Not stated",
            task.deadline or "Not stated",
            task.priority or "Not stated",
            ", ".join(task.source_ids) or "Not available",
        ]
        lines.append("| " + " | ".join(_markdown_safe(value) for value in values) + " |")
    lines.extend(["", "## Key points", ""])
    lines.extend(f"- {point}" for point in analysis.key_points)
    lines.extend(["", "## Open questions", ""])
    lines.extend(f"- {question}" for question in analysis.open_questions)
    content = "\n".join(lines) + "\n"
    return StreamingResponse(
        iter([content]),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="meetingmind-notes.md"'},
    )


def _spreadsheet_safe(value: str) -> str:
    return f"'{value}" if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value


def _markdown_safe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
