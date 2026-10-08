from fastapi.testclient import TestClient

from backend.app.chains.errors import ModelUnavailableError
from backend.app.api.main import app
from backend.app.pipeline.service import MeetingService, service

client = TestClient(app)


def test_health_is_available_without_loading_model() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is False
    assert response.json()["status"] in {"ok", "degraded"}
    assert response.json()["api_status"] == "connected"
    assert response.json()["model_status"] == "unavailable"
    assert "not been loaded yet" in response.json()["model_status_detail"]


def test_health_reports_detected_gpu_separately_from_model_readiness(monkeypatch) -> None:
    from backend.app.api import routes

    monkeypatch.setattr(routes, "_gpu_available", lambda: True)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["gpu_available"] is True
    assert response.json()["model_status"] == "unavailable"
    assert response.json()["model_ready"] is False


def test_nvidia_gpu_detection_uses_driver_query(monkeypatch) -> None:
    from backend.app.api import routes

    monkeypatch.setattr(
        routes.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"stdout": "NVIDIA RTX 3050 Ti\n"})(),
    )

    assert routes._nvidia_gpu_detected() is True


def test_model_state_tracks_loader_lifecycle(monkeypatch) -> None:
    from backend.app.chains import local_llm
    from backend.app.pipeline import service as service_module

    meeting_service = MeetingService()
    observed_statuses = []

    def load_model(_settings):
        observed_statuses.append(meeting_service.model_state[0])
        return object(), object()

    monkeypatch.setattr(local_llm, "load_local_model", load_model)
    monkeypatch.setattr(service_module, "build_chains", lambda _llm: {"extract": object()})

    meeting_service._ensure_runtime()

    assert observed_statuses == ["loading"]
    assert meeting_service.model_state[0] == "ready"
    assert meeting_service.model_ready is True


def test_model_load_failure_reports_safe_vram_status(monkeypatch) -> None:
    from backend.app.chains import local_llm

    meeting_service = MeetingService()

    def fail_to_load(_settings):
        try:
            raise RuntimeError("CUDA out of memory while allocating model weights")
        except RuntimeError as cause:
            raise ModelUnavailableError("Local model load failed.") from cause

    monkeypatch.setattr(local_llm, "load_local_model", fail_to_load)

    try:
        meeting_service._ensure_runtime()
    except ModelUnavailableError:
        pass
    else:
        raise AssertionError("Expected the model loader failure to propagate.")

    status, detail = meeting_service.model_state
    assert status == "unavailable"
    assert "GPU memory requirement not met" in detail
    assert "CUDA out of memory" not in detail
    assert meeting_service.model_ready is False


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
