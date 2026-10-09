"""Model lifecycle, analysis orchestration, and in-memory meeting sessions."""

from collections import OrderedDict
from threading import Lock, RLock
import json
import logging
import os
import re
from uuid import uuid4

from backend.app.chains.prompts import NO_ANSWER, build_chains
from backend.app.chains.errors import (
    ModelUnavailableError,
    RemoteInferenceError,
    RemoteResponseError,
)
from backend.app.chains.remote_llm import (
    RemoteInferenceClient,
    RemoteQAChain,
    RemoteTransformersLLM,
)
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
        self._remote_client = RemoteInferenceClient(config)
        self._embeddings = None
        self._chains = None
        self._gpu_available = False
        self._model_name = config.remote_model_name
        self._model_state: tuple[ModelStatus, str] = (
            "unavailable",
            "Start the Colab inference notebook and configure LLM_BASE_URL.",
        )
        self._meetings: OrderedDict[str, tuple[MeetingRecord, FaissVectorStore | None]] = OrderedDict()

    @property
    def model_ready(self) -> bool:
        return self._model_state[0] == "ready"

    @property
    def model_state(self) -> tuple[ModelStatus, str]:
        return self._model_state

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    @property
    def model_name(self) -> str:
        return self._model_name

    def refresh_remote_status(self) -> tuple[ModelStatus, str]:
        if self.config.llm_backend != "remote":
            self._model_state = (
                "unavailable",
                "Unsupported LLM_BACKEND. Configure LLM_BACKEND=remote.",
            )
            self._gpu_available = False
            return self._model_state
        try:
            remote = self._remote_client.health()
        except RemoteInferenceError as error:
            self._model_state = ("unavailable", str(error))
            self._gpu_available = False
            return self._model_state
        except RemoteResponseError:
            LOGGER.warning("Colab health endpoint returned an invalid response")
            self._model_state = (
                "unavailable",
                "The Colab AI service returned an invalid health response.",
            )
            self._gpu_available = False
            return self._model_state

        self._model_state = (
            remote["model_status"],
            remote["detail"] or _default_remote_detail(remote["model_status"]),
        )
        self._gpu_available = remote["gpu_available"]
        self._model_name = remote["model_name"]
        return self._model_state

    def _ensure_runtime(self) -> None:
        status, detail = self.refresh_remote_status()
        if status != "ready":
            raise ModelUnavailableError(
                detail or "The Colab AI service is not ready. Check its status and try again."
            )
        with self._runtime_lock:
            if self._chains is not None and self._embeddings is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                embeddings = SentenceTransformer(
                    self.config.embedding_model,
                    device="cpu",
                )
                llm = RemoteTransformersLLM(
                    client=self._remote_client,
                    model_name=self._model_name,
                )
                chains = build_chains(
                    llm,
                    qa_chain=RemoteQAChain(self._remote_client),
                )
            except Exception as error:
                LOGGER.exception("MeetingMind local retrieval runtime initialization failed")
                raise
            self._embeddings, self._chains = embeddings, chains

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
                parsed = None
                try:
                    parsed = safe_parse(raw, self._chains["parser"])
                    values = normalize_extraction(
                        parsed,
                        chunk.chunk_id,
                        _source_excerpt(chunk.text),
                    )
                except ModelOutputError:
                    if os.getenv("MEETINGMIND_DIAGNOSTIC_MODEL_OUTPUT") == "1":
                        safe_raw = raw[:1200]
                        safe_raw = re.sub(
                            r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
                            r"\1[REDACTED]",
                            safe_raw,
                        )
                        safe_raw = re.sub(
                            r"(?i)(LLM_API_KEY\s*[:=]\s*)\S+",
                            r"\1[REDACTED]",
                            safe_raw,
                        )
                        structure = (
                            {
                                key: {
                                    "type": type(value).__name__,
                                    "length": len(value) if isinstance(value, list) else None,
                                    "first_item_type": (
                                        type(value[0]).__name__
                                        if isinstance(value, list) and value
                                        else None
                                    ),
                                }
                                for key, value in parsed.items()
                            }
                            if isinstance(parsed, dict)
                            else None
                        )
                        # This opt-in diagnostic may contain transcript-derived text; keep it disabled in production.
                        LOGGER.warning(
                            "Temporary model-output diagnostic: raw response excerpt=%r parsed_structure=%s",
                            safe_raw,
                            structure,
                        )
                    raise
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
        from backend.app.preprocessing.chunking import TranscriptChunk

        vector_store = FaissVectorStore(
            self._embeddings,
            [
                TranscriptChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    start_word=chunk.start_word,
                    end_word=chunk.end_word,
                )
                for chunk in chunks
            ],
        )
        self._register(record, vector_store)
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
        except (ModelUnavailableError, RemoteResponseError):
            raise
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
    excerpt = " ".join(text.split()).strip()
    return excerpt if len(excerpt) <= limit else excerpt[:limit].rstrip() + "…"


def _default_remote_detail(status: ModelStatus) -> str:
    return {
        "ready": "The Colab inference model is ready.",
        "loading": "The Colab inference model is loading.",
        "unavailable": "The Colab inference model is unavailable.",
    }[status]


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _clean_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    title = value.strip()
    return "" if title.casefold() in {"not specified", "unknown", "none"} else title


service = MeetingService()
