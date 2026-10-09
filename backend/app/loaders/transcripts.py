"""Load and normalize transcripts from YouTube, text, subtitle, and PDF inputs."""

from html import unescape
from io import BytesIO
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".vtt", ".srt"}
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
TIMESTAMP_PATTERN = re.compile(
    r"^(?:\d{2}:)?\d{2}:\d{2}[,.]\d{3}\s+-->\s+(?:\d{2}:)?\d{2}:\d{2}[,.]\d{3}"
)


class TranscriptInputError(ValueError):
    """Raised when source input cannot be converted into readable transcript text."""


def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    video_id = ""

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id:
            segments = parsed.path.strip("/").split("/")
            if len(segments) >= 2 and segments[0] in {"shorts", "embed", "live", "v"}:
                video_id = segments[1]

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise TranscriptInputError("Enter a valid YouTube video link.")
    return video_id


def fetch_youtube_transcript(url: str) -> str:
    video_id = extract_video_id(url)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        text = "\n".join(segment.text for segment in transcript)
    except Exception as error:
        raise TranscriptInputError(
            "Captions could not be retrieved. Check that the video is public and has English captions."
        ) from error
    return normalize_transcript(text)


def clean_subtitles(text: str, suffix: str) -> str:
    lines = text.splitlines()
    has_timing = any(TIMESTAMP_PATTERN.match(line.strip()) for line in lines)
    if not has_timing:
        raise TranscriptInputError(f"The {suffix[1:].upper()} file has no valid caption timestamps.")

    cleaned_lines: list[str] = []
    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.upper().startswith("WEBVTT") or candidate.startswith("NOTE "):
            continue
        if candidate.isdigit() or TIMESTAMP_PATTERN.match(candidate):
            continue
        candidate = unescape(candidate)
        candidate = re.sub(r"<[^>]*>", "", candidate)
        candidate = re.sub(r"\{\\[^}]*\}", "", candidate)
        if candidate:
            cleaned_lines.append(candidate)
    return "\n".join(cleaned_lines)


def load_transcript_file(filename: str, content: bytes, max_bytes: int) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise TranscriptInputError("Upload a PDF, TXT, VTT, or SRT meeting transcript.")
    if not content:
        raise TranscriptInputError("The uploaded file is empty.")
    if len(content) > max_bytes:
        raise TranscriptInputError(f"The file is too large. Maximum upload size is {max_bytes // (1024 * 1024)} MB.")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise TranscriptInputError("This PDF is password-protected. Upload an unlocked PDF.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except TranscriptInputError:
            raise
        except Exception as error:
            raise TranscriptInputError("This PDF could not be read. It may be corrupted or image-only.") from error
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise TranscriptInputError("The text file must use UTF-8 encoding.") from error
        if suffix in {".vtt", ".srt"}:
            text = clean_subtitles(text, suffix)

    normalized = normalize_transcript(unescape(text))
    if not normalized:
        raise TranscriptInputError("No readable transcript text was found in this file.")
    return normalized


def normalize_transcript(text: str) -> str:
    text = unescape(text).replace("\x00", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
