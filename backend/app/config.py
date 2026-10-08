"""Runtime settings loaded from environment variables."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_max_new_tokens: int = 900
    max_input_tokens: int = 8192
    max_chunks: int = 20
    chunk_words: int = 600
    chunk_overlap: int = 90
    rag_top_k: int = 4
    min_retrieval_score: float = 0.20
    max_upload_bytes: int = 25 * 1024 * 1024
    meeting_cache_size: int = 24
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            model_max_new_tokens=int(os.getenv("MODEL_MAX_NEW_TOKENS", cls.model_max_new_tokens)),
            max_input_tokens=int(os.getenv("MAX_INPUT_TOKENS", cls.max_input_tokens)),
            max_chunks=int(os.getenv("MAX_CHUNKS", cls.max_chunks)),
            chunk_words=int(os.getenv("CHUNK_WORDS", cls.chunk_words)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            rag_top_k=int(os.getenv("RAG_TOP_K", cls.rag_top_k)),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", cls.min_retrieval_score)),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )


settings = Settings.from_environment()
