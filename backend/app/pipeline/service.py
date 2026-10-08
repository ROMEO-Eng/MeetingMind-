"""Model lifecycle, analysis orchestration, and in-memory meeting sessions."""

from collections import OrderedDict
from threading import Lock, RLock
import json
import logging
import re
from uuid import uuid4

from backend.app.chains.prompts import NO_ANSWER, build_chains
from backend.app.chains.errors import ModelUnavailableError
from backend.app.config import Settings, settings
from backend.app.embeddings.vector_store import FaissVectorStore
from backend.app.models import (
    DecisionItem,
    MeetingAnalysisResponse,
    MeetingRecord,
    ModelStatus,
    QuestionResponse,
    SourceChunk,
    TaskItem,
)
from backend.app.parsers.structured import (
    ModelOutputError,
    deduplicate_records,
    normalize_extraction,
    safe_parse,
)
from backend.app.preprocessing.chunking import split_into_chunks
from backend.app.loaders.transcripts import normalize_transcript

LOGGER = logging.getLogger(__name__)


class MeetingService:
    def __init__(self, config: Settings = settings):
        self.config = config
        self._runtime_lock = Lock()
        self._analysis_lock = Lock()
        self._registry_lock = RLock()
        self._llm = None
        self._embeddings = None
        self._chains = None
        self._model_state: tuple[ModelStatus, str] = (
            "unavailable",
            "The model has not been loaded yet. It will load when you analyze a meeting.",
        )
        self._meetings: OrderedDict[str, tuple[MeetingRecord, FaissVectorStore | None]] = OrderedDict()

    @property
    def model_ready(self) -> bool:
        return self._model_state[0] == "ready"

    @property
    def model_state(self) -> tuple[ModelStatus, str]:
        return self._model_state

    def _ensure_runtime(self) -> None:
        if self.model_ready:
            return
        with self._runtime_lock:
            if self.model_ready:
                return
            from backend.app.chains.local_llm import load_local_model

            self._model_state = (
                "loading",
                "Loading Mistral Nemo and the meeting analysis components.",
            )
            try:
                llm, embeddings = load_local_model(self.config)
                chains = build_chains(llm)
            except Exception as error:
                LOGGER.exception("Local AI model initialization failed")
                self._model_state = ("unavailable", _model_failure_detail(error))
                raise
            self._llm, self._embeddings, self._chains = llm, embeddings, chains
            self._model_state = ("ready", "Mistral Nemo and its required components are loaded.")

    def analyze(self, text: str, source_name: str) -> MeetingAnalysisResponse:
        transcript = normalize_transcript(text)
        if len(transcript.split()) < 8:
            raise ValueError("Add more transcript text. At least 8 words are required.")
        self._ensure_runtime()
        chunks, truncated = split_into_chunks(
            transcript,
            chunk_words=self.config.chunk_words,
            overlap=self.config.chunk_overlap,
            max_chunks=self.config.max_chunks,
        )

        with self._analysis_lock:
            extracted = []
            for chunk in chunks:
                raw = self._chains["extract"].invoke({"chunk": chunk.text})["text"]
                values = normalize_extraction(
                    safe_parse(raw, self._chains["parser"]),
                    chunk.chunk_id,
                    _source_excerpt(chunk.text),
                )
                extracted.append(values)

            decisions = deduplicate_records(
                [decision for item in extracted for decision in item["decisions"]],
                "decision",
            )
            tasks = deduplicate_records(
                [task for item in extracted for task in item["tasks"]],
                "task",
            )
            key_points = _deduplicate_strings(
                [point for item in extracted for point in item["key_points"]]
            )
            open_questions = _deduplicate_strings(
                [question for item in extracted for question in item["open_questions"]]
            )
            title = next((item["title"] for item in extracted if item["title"]), "")
            summary_source = {
                "title": title,
                "summaries": [item["summary"] for item in extracted if item["summary"]],
            }
            raw_final = self._chains["final"].invoke(
                {"extractions": json.dumps(summary_source, ensure_ascii=False)}
            )["text"]
            final = safe_parse(raw_final, self._chains["summary_parser"])
            title = _clean_title(final.get("Meeting Title")) or title or "Meeting notes"
            summary = str(final.get("Meeting Summary") or "").strip()
            if not summary:
                summary = " ".join(summary_source["summaries"]).strip()

        meeting_id = uuid4().hex
        source_chunks = [
            SourceChunk(chunk_id=chunk.chunk_id, text=chunk.text)
            for chunk in chunks
        ]
        response = MeetingAnalysisResponse(
            meeting_id=meeting_id,
            source_name=source_name,
            title=title,
            executive_summary=summary,
            decisions=decisions,
            tasks=tasks,
            key_points=key_points,
            open_questions=open_questions,
            word_count=len(transcript.split()),
            estimated_minutes=max(1, round(len(transcript.split()) / 130)),
            chunk_count=len(chunks),
            chunks_truncated=truncated,
        )
        record = MeetingRecord(
            meeting_id=meeting_id,
            source_name=source_name,
            chunks=source_chunks,
            analysis=response,
        )
        self._register(record, None)
        return response

    def register_sample(
        self,
        analysis: MeetingAnalysisResponse,
        chunks: list[SourceChunk],
    ) -> None:
        record = MeetingRecord(
            meeting_id=analysis.meeting_id,
            source_name=analysis.source_name,
            chunks=chunks,
            analysis=analysis,
        )
        self._register(record, None)

    def answer(self, meeting_id: str, question: str) -> QuestionResponse:
        with self._registry_lock:
            if meeting_id not in self._meetings:
                raise KeyError("This meeting session has expired. Analyse it again to continue.")
        self._ensure_runtime()
        with self._registry_lock:
            item = self._meetings.get(meeting_id)
            if item is None:
                raise KeyError("This meeting session has expired. Analyse it again to continue.")
            record, vector_store = item
            self._meetings.move_to_end(meeting_id)
            if vector_store is None:
                from backend.app.preprocessing.chunking import TranscriptChunk

                vector_store = FaissVectorStore(
                    self._embeddings,
                    [
                        TranscriptChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            start_word=0,
                            end_word=len(chunk.text.split()),
                        )
                        for chunk in record.chunks
                    ],
                )
                self._meetings[meeting_id] = (record, vector_store)

        from backend.app.rag.grounded_qa import answer_question

        try:
            return answer_question(
                question,
                vector_store,
                self._embeddings,
                self._chains["qa"],
                self.config.rag_top_k,
                self.config.min_retrieval_score,
            )
        except Exception as error:
            LOGGER.exception("Meeting Q&A failed")
            raise RuntimeError("The meeting question could not be answered. Please try again.") from error

    def get_record(self, meeting_id: str) -> MeetingRecord | None:
        with self._registry_lock:
            entry = self._meetings.get(meeting_id)
            return entry[0] if entry else None

    def _register(self, record: MeetingRecord, vector_store: FaissVectorStore | None) -> None:
        with self._registry_lock:
            self._meetings[record.meeting_id] = (record, vector_store)
            self._meetings.move_to_end(record.meeting_id)
            while len(self._meetings) > self.config.meeting_cache_size:
                self._meetings.popitem(last=False)


def _source_excerpt(text: str, limit: int = 320) -> str:
    excerpt = re.sub(r"\s+", " ", text).strip()
    return excerpt if len(excerpt) <= limit else excerpt[:limit].rstrip() + "…"


def _model_failure_detail(error: Exception) -> str:
    cause: BaseException | None = error
    while cause is not None:
        message = str(cause).casefold()
        if any(
            phrase in message
            for phrase in ("cuda out of memory", "out of memory", "not enough memory", "insufficient memory")
        ):
            return "GPU memory requirement not met. Use an NVIDIA T4 with 16 GB VRAM or equivalent."
        if "cuda gpu" in message or "cuda-enabled gpu" in message:
            return "A CUDA-enabled GPU is required for local Mistral Nemo inference."
        cause = cause.__cause__
    return "The local AI model could not be loaded. Check the API logs for details."


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _clean_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    title = value.strip()
    return "" if title.casefold() in {"not specified", "unknown", "none"} else title


service = MeetingService()
