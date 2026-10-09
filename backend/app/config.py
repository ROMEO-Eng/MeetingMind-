"""Runtime settings loaded from environment variables."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    llm_backend: str = "remote"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_request_timeout: int = 300
    llm_health_timeout: int = 5
    remote_model_name: str = "Hugging Face model on Colab"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
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
            llm_backend=os.getenv("LLM_BACKEND", cls.llm_backend),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url).strip().rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", cls.llm_api_key),
            llm_request_timeout=int(os.getenv("LLM_REQUEST_TIMEOUT", cls.llm_request_timeout)),
            llm_health_timeout=int(os.getenv("LLM_HEALTH_TIMEOUT", cls.llm_health_timeout)),
            remote_model_name=os.getenv("REMOTE_MODEL_NAME", cls.remote_model_name),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            max_chunks=int(os.getenv("MAX_CHUNKS", cls.max_chunks)),
            chunk_words=int(os.getenv("CHUNK_WORDS", cls.chunk_words)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            rag_top_k=int(os.getenv("RAG_TOP_K", cls.rag_top_k)),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", cls.min_retrieval_score)),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )


settings = Settings.from_environment()
