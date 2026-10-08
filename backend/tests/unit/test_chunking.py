import pytest

from backend.app.preprocessing.chunking import split_into_chunks


def test_chunks_have_stable_ids_and_configured_overlap() -> None:
    chunks, truncated = split_into_chunks(" ".join(str(i) for i in range(14)), chunk_words=8, overlap=3)

    assert not truncated
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-001", "chunk-002", "chunk-003"]
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[0].start_word == 0
    assert chunks[1].start_word == 5
    assert chunks[-1].text.split()[-4:] == ["10", "11", "12", "13"]


def test_chunking_caps_long_transcripts() -> None:
    chunks, truncated = split_into_chunks("word " * 50, chunk_words=10, overlap=2, max_chunks=3)

    assert len(chunks) == 3
    assert truncated


def test_empty_transcript_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_into_chunks(" \n ")


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_into_chunks("some transcript words", chunk_words=5, overlap=5)
