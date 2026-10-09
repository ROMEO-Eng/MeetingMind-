import json
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from backend.app.api.main import app
from backend.app.api import routes
from backend.app.config import Settings
from backend.app.embeddings.vector_store import RetrievedChunk
from backend.app.pipeline.service import MeetingService

TRANSCRIPT = """The product team met to review the upcoming release.

Alex will prepare the launch checklist by Friday.

Sarah will coordinate the final UI review with the design team.

The team agreed to release version 2.1 next Monday.

The team will review the release status again next week.

There was no final decision about the marketing campaign."""


class FakeEmbeddings:
    def __init__(self, *_args, **_kwargs):
        pass


class FakeVectorStore:
    def __init__(self, _embedding_model, chunks):
        self.chunks = chunks

    def search(self, _question, _embedding_model, top_k):
        return [
            RetrievedChunk(chunk=chunk, score=0.9)
            for chunk in self.chunks[:top_k]
        ]


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def test_meeting_analysis_and_grounded_qa_remote_flow(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeEmbeddings),
    )
    monkeypatch.setattr(
        "backend.app.pipeline.service.FaissVectorStore",
        FakeVectorStore,
    )

    def fake_urlopen(request, **_kwargs):
        path = urlsplit(request.full_url).path
        if path == "/health":
            return FakeResponse(
                {
                    "model_status": "ready",
                    "model_ready": True,
                    "model_name": "Test instruction model",
                    "gpu_available": True,
                    "detail": "Ready.",
                }
            )
        body = json.loads(request.data)
        if path == "/analyse":
            if "Transcript excerpt:" in body["prompt"]:
                return FakeResponse(
                    {
                        "text": json.dumps(
                            {
                                "meeting_title": "Upcoming release planning",
                                "meeting_summary": "The team reviewed the upcoming release and agreed on the version 2.1 launch.",
                                "decisions": [
                                    {
                                        "Decision": "Release version 2.1 next Monday.",
                                        "Context": "The team agreed on the release date.",
                                    }
                                ],
                                "tasks": [
                                    {
                                        "Task": "Prepare the launch checklist",
                                        "Owner": "Alex",
                                        "Deadline": "Friday",
                                        "Priority": None,
                                    },
                                    {
                                        "Task": "Coordinate the final UI review with the design team",
                                        "Owner": "Sarah",
                                        "Deadline": None,
                                        "Priority": None,
                                    },
                                ],
                                "Key Points": [
                                    "The team reviewed the upcoming release.",
                                    "The team will review the release status again next week.",
                                ],
                                "open_questions": [
                                    "The marketing campaign decision remains open."
                                ],
                            }
                        )
                    }
                )
            return FakeResponse(
                {
                    "text": json.dumps(
                        {
                            "Meeting Title": "Upcoming release planning",
                            "Meeting Summary": "The team reviewed the upcoming release and agreed to release version 2.1 next Monday.",
                        }
                    )
                }
            )
        if path == "/qa":
            if body["question"] == "What is Alex responsible for?":
                answer = "Alex is responsible for preparing the launch checklist by Friday."
            else:
                answer = "I couldn't find this information in the meeting."
            return FakeResponse({"answer": answer})
        raise AssertionError(f"Unexpected remote route: {path}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    meeting_service = MeetingService(
        Settings(llm_base_url="https://colab.example", llm_api_key="test-key")
    )
    monkeypatch.setattr(routes, "_service", lambda: meeting_service)

    with TestClient(app) as client:
        analysis_response = client.post(
            "/api/analyse",
            data={"source_type": "text", "transcript": TRANSCRIPT},
        )
        assert analysis_response.status_code == 200
        analysis = analysis_response.json()
        assert analysis["executive_summary"]
        assert analysis["decisions"][0]["decision"] == "Release version 2.1 next Monday."
        assert analysis["tasks"][0]["owner"] == "Alex"
        assert analysis["tasks"][0]["deadline"] == "Friday"
        assert analysis["tasks"][1]["owner"] == "Sarah"
        assert analysis["tasks"][1]["deadline"] is None
        assert "release" in analysis["key_points"][0].casefold()
        assert any(
            "review the release status again next week" in point.casefold()
            for point in analysis["key_points"]
        )
        assert any(
            "marketing campaign" in question.casefold()
            for question in analysis["open_questions"]
        )
        assert analysis["chunk_count"] == 1

        supported = client.post(
            "/api/qa",
            json={
                "meeting_id": analysis["meeting_id"],
                "question": "What is Alex responsible for?",
            },
        )
        assert supported.status_code == 200
        assert supported.json()["answer"] == (
            "Alex is responsible for preparing the launch checklist by Friday."
        )
        assert supported.json()["found"] is True
        assert supported.json()["sources"][0]["chunk_id"] == "chunk-001"
        assert "Alex will prepare the launch checklist by Friday." in supported.json()["sources"][0]["excerpt"]

        unsupported = client.post(
            "/api/qa",
            json={
                "meeting_id": analysis["meeting_id"],
                "question": "What was the company's revenue last quarter?",
            },
        )
        assert unsupported.status_code == 200
        assert unsupported.json()["answer"] == "I couldn't find this information in the meeting."
        assert unsupported.json()["found"] is False
        assert unsupported.json()["sources"] == []


def test_scalar_decisions_from_remote_extraction_return_safe_bad_gateway(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeEmbeddings),
    )

    def fake_urlopen(request, **_kwargs):
        path = urlsplit(request.full_url).path
        if path == "/health":
            return FakeResponse(
                {
                    "model_status": "ready",
                    "model_ready": True,
                    "model_name": "Test instruction model",
                    "gpu_available": True,
                    "detail": "Ready.",
                }
            )
        return FakeResponse(
            {
                "text": json.dumps(
                    {
                        "meeting_title": "Upcoming release",
                        "meeting_summary": "The team reviewed the upcoming release.",
                        "decisions": "The team agreed to release version 2.1 next Monday.",
                        "tasks": [],
                        "key_points": [],
                        "open_questions": [],
                    }
                )
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "backend.app.pipeline.service.FaissVectorStore",
        FakeVectorStore,
    )
    meeting_service = MeetingService(
        Settings(llm_base_url="https://colab.example", llm_api_key="test-key")
    )
    monkeypatch.setattr(routes, "_service", lambda: meeting_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/analyse",
            data={"source_type": "text", "transcript": TRANSCRIPT},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The AI could not structure the meeting notes. Please retry the analysis."
    )
