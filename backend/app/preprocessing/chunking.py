"""Overlapping, source-addressable transcript chunks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_id: str
    text: str
    start_word: int
    end_word: int


def split_into_chunks(
    text: str,
    chunk_words: int = 600,
    overlap: int = 90,
    max_chunks: int = 20,
) -> tuple[list[TranscriptChunk], bool]:
    words = text.split()
    if not words:
        raise ValueError("Transcript text is empty.")
    if chunk_words <= 0 or overlap < 0 or overlap >= chunk_words:
        raise ValueError("Chunk size must be positive and overlap must be smaller than the chunk size.")

    stride = chunk_words - overlap
    total_count = (len(words) + stride - 1) // stride
    chunks: list[TranscriptChunk] = []
    for index in range(min(total_count, max_chunks)):
        start = index * stride
        end = min(start + chunk_words, len(words))
        chunks.append(
            TranscriptChunk(
                chunk_id=f"chunk-{index + 1:03d}",
                text=" ".join(words[start:end]),
                start_word=start,
                end_word=end,
            )
        )
    return chunks, total_count > max_chunks
