"""Typed API contracts and validated meeting records."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Priority = Literal["High", "Medium", "Low"]
ModelStatus = Literal["ready", "loading", "unavailable"]


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None


class DecisionItem(BaseModel):
    decision: str
    context: str = ""
    source_ids: list[str] = Field(default_factory=list)
    source_excerpt: str | None = None


class TaskItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None
    priority: Priority | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_excerpt: str | None = None

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def normalize_missing_values(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().casefold() in {"", "not specified", "unknown", "n/a"}:
            return None
        return value

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().casefold() in {"", "not specified", "unknown", "n/a"}:
            return None
        if isinstance(value, str):
            normalized = value.strip().title()
            if normalized in {"High", "Medium", "Low"}:
                return normalized
        raise ValueError("Priority must be High, Medium, Low, or null.")


class MeetingAnalysisResponse(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    meeting_id: str
    source_name: str
    title: str
    executive_summary: str
    decisions: list[DecisionItem] = Field(default_factory=list)
    tasks: list[TaskItem] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    word_count: int = Field(ge=0)
    estimated_minutes: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    chunks_truncated: bool = False


class QuestionRequest(BaseModel):
    meeting_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)


class QuestionSource(BaseModel):
    chunk_id: str
    excerpt: str
    score: float


class QuestionResponse(BaseModel):
    answer: str
    found: bool
    sources: list[QuestionSource] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_status: Literal["connected"]
    model_status: ModelStatus
    model_status_detail: str
    model_ready: bool
    gpu_available: bool
    model_name: str


class ErrorResponse(BaseModel):
    detail: str


class MeetingRecord(BaseModel):
    meeting_id: str
    source_name: str
    chunks: list[SourceChunk]
    analysis: MeetingAnalysisResponse
