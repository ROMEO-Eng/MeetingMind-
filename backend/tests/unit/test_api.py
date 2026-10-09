from fastapi.testclient import TestClient

from backend.app.api.main import app
from backend.app.config import Settings
from backend.app.chains.errors import RemoteResponseError
from backend.app.pipeline.service import MeetingService, service

client = TestClient(app)


def test_health_reports_remote_unavailable_without_loading_model(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "refresh_remote_status",
        lambda: ("unavailable", "The Colab inference model is unavailable."),
    )
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is False
    assert response.json()["status"] == "degraded"
    assert response.json()["api_status"] == "connected"
    assert response.json()["model_status"] == "unavailable"
    assert response.json()["model_status_detail"] == "The Colab inference model is unavailable."


def test_health_with_unconfigured_colab_stays_reachable_and_never_claims_ready(monkeypatch) -> None:
    from backend.app.api import routes

    meeting_service = MeetingService(Settings(llm_base_url=""))
    monkeypatch.setattr(routes, "_service", lambda: meeting_service)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["api_status"] == "connected"
    assert response.json()["model_status"] == "unavailable"
    assert response.json()["model_ready"] is False


def test_health_only_reports_ready_when_remote_model_confirms_ready(monkeypatch) -> None:
    monkeypatch.setattr(service, "_gpu_available", True)
    monkeypatch.setattr(
        service,
        "refresh_remote_status",
        lambda: setattr(service, "_model_state", ("ready", "Remote model ready."))
        or service.model_state,
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["gpu_available"] is True
    assert response.json()["model_status"] == "ready"
    assert response.json()["model_ready"] is True


def test_analysis_reports_unconfigured_remote_without_loading_local_model(monkeypatch) -> None:
    from backend.app.api import routes

    monkeypatch.setattr(
        routes,
        "_service",
        lambda: MeetingService(Settings(llm_base_url="")),
    )
    response = client.post(
        "/api/analyse",
        data={
            "source_type": "text",
            "transcript": "The team reviewed the release plan and assigned owners for follow-up work.",
        },
    )

    assert response.status_code == 503
    assert "Colab AI service is not configured" in response.json()["detail"]


def test_sample_is_typed_and_preserves_missing_metadata() -> None:
    response = client.post("/api/sample")

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "November product launch planning"
    unassigned = next(task for task in body["tasks"] if task["task"] == "Obtain legal approval for the launch copy")
    assert unassigned["owner"] is None
    assert unassigned["deadline"] is None
    assert unassigned["source_ids"] == ["chunk-002"]
    assert service.get_record(body["meeting_id"]) is not None


def test_unknown_meeting_qa_returns_safe_not_found() -> None:
    response = client.post(
        "/api/qa",
        json={"meeting_id": "not-a-session", "question": "What was decided?"},
    )

    assert response.status_code == 404
    assert "session has expired" in response.json()["detail"]


def test_qa_invalid_remote_response_returns_safe_bad_gateway(monkeypatch) -> None:
    from backend.app.api import routes

    class InvalidRemoteService:
        def answer(self, _meeting_id: str, _question: str):
            raise RemoteResponseError("invalid remote response")

    monkeypatch.setattr(routes, "_service", lambda: InvalidRemoteService())
    response = client.post(
        "/api/qa",
        json={"meeting_id": "meeting-1", "question": "What was decided?"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The Colab AI service returned an invalid answer. Please retry."
    )


def test_empty_text_input_is_rejected_before_model_loading() -> None:
    response = client.post(
        "/api/analyse",
        data={"source_type": "text", "transcript": ""},
    )

    assert response.status_code == 422
    assert "empty" in response.json()["detail"].casefold()


def test_spreadsheet_export_neutralizes_formula_cells() -> None:
    from backend.app.api.routes import _spreadsheet_safe

    assert _spreadsheet_safe("=1+1") == "'=1+1"
    assert _spreadsheet_safe("Normal text") == "Normal text"
