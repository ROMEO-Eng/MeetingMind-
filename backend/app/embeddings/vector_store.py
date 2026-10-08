"""FAISS inner-product retrieval over normalized MiniLM embeddings."""

from dataclasses import dataclass
from typing import Any

from backend.app.preprocessing.chunking import TranscriptChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TranscriptChunk
    score: float


class FaissVectorStore:
    def __init__(self, embedding_model: Any, chunks: list[TranscriptChunk]):
        if not chunks:
            raise ValueError("Cannot create a search index without transcript chunks.")
        import faiss

        self._chunks = chunks
        vectors = embedding_model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def search(self, question: str, embedding_model: Any, top_k: int) -> list[RetrievedChunk]:
        vector = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indexes = self._index.search(vector, min(top_k, len(self._chunks)))
        return [
            RetrievedChunk(chunk=self._chunks[int(index)], score=float(score))
            for score, index in zip(scores[0], indexes[0], strict=True)
            if index >= 0
        ]
