import pytest

from backend.app.loaders.transcripts import (
    TranscriptInputError,
    clean_subtitles,
    extract_video_id,
    load_transcript_file,
    normalize_transcript,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abcdefghijk&t=10", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk", "abcdefghijk"),
        ("https://youtube.com/shorts/abcdefghijk", "abcdefghijk"),
        ("https://m.youtube.com/embed/abcdefghijk", "abcdefghijk"),
    ],
)
def test_extract_video_id(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


def test_rejects_invalid_youtube_urls() -> None:
    with pytest.raises(TranscriptInputError, match="valid YouTube"):
        extract_video_id("https://youtube.com.evil.invalid/watch?v=abcdefghijk")


def test_normalize_transcript_cleans_whitespace_and_html_entities() -> None:
    assert normalize_transcript("  Maya&nbsp;:   hello  \n\n\nOmar &amp; Sara ") == "Maya : hello\n\nOmar & Sara"


def test_clean_subtitles_removes_timing_and_markup() -> None:
    captions = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<v Maya>Hello &amp; welcome</v>\n"
    assert clean_subtitles(captions, ".vtt") == "Hello & welcome"


@pytest.mark.parametrize("suffix", [".srt", ".vtt"])
def test_malformed_subtitles_are_rejected(suffix: str) -> None:
    with pytest.raises(TranscriptInputError, match="no valid caption timestamps"):
        load_transcript_file(f"meeting{suffix}", b"just words, no timestamp", 1024)


def test_empty_upload_and_unsupported_extension_are_rejected() -> None:
    with pytest.raises(TranscriptInputError, match="empty"):
        load_transcript_file("meeting.txt", b"", 1024)
    with pytest.raises(TranscriptInputError, match="PDF, TXT"):
        load_transcript_file("meeting.docx", b"some content", 1024)


def test_oversized_upload_is_rejected() -> None:
    with pytest.raises(TranscriptInputError, match="too large"):
        load_transcript_file("meeting.txt", b"12345", 4)
