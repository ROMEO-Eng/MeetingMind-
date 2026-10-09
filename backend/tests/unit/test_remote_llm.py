from io import BytesIO
import socket
import urllib.error

import pytest

from backend.app.chains.errors import RemoteInferenceError, RemoteResponseError
from backend.app.chains.remote_llm import RemoteInferenceClient
from backend.app.config import Settings


def create_client(**overrides) -> RemoteInferenceClient:
    defaults = {
        "llm_backend": "remote",
        "llm_base_url": "https://colab.example",
        "llm_api_key": "local-test-key",
    }
    defaults.update(overrides)
    return RemoteInferenceClient(Settings(**defaults))


def test_remote_health_confirms_ready_model_and_gpu(monkeypatch) -> None:
    client = create_client()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "model_status": "ready",
            "model_ready": True,
            "model_name": "Qwen",
            "gpu_available": True,
            "detail": "Model loaded.",
        },
    )

    result = client.health()

    assert result["model_status"] == "ready"
    assert result["gpu_available"] is True


def test_remote_health_never_treats_unconfirmed_model_as_ready(monkeypatch) -> None:
    client = create_client()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "model_status": "ready",
            "model_ready": False,
            "model_name": "Qwen",
            "gpu_available": True,
            "detail": "",
        },
    )

    assert client.health()["model_status"] == "unavailable"


def test_remote_request_sends_configured_bearer_key(monkeypatch) -> None:
    client = create_client()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"text":"ok"}'

    def fake_urlopen(request, **_kwargs):
        captured["authorization"] = request.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert client.complete("prompt") == "ok"
    assert captured["authorization"] == "Bearer local-test-key"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://colab.exampleLLM_API_KEY=not-a-real-secret",
        "https://colab.example/api",
        "http://colab.example",
        "https://user:password@colab.example",
        "https://colab.example?token=not-a-real-secret",
    ],
)
def test_invalid_remote_base_url_is_rejected_without_request(
    monkeypatch,
    base_url: str,
) -> None:
    client = create_client(llm_base_url=base_url)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("Invalid URL must not make a request."),
    )

    with pytest.raises(RemoteInferenceError, match="LLM_BASE_URL"):
        client.health()


def test_remote_health_url_uses_origin_and_health_route(monkeypatch) -> None:
    client = create_client(llm_base_url="https://colab.example/")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"model_status":"ready","model_ready":true,'
                b'"model_name":"Qwen","gpu_available":true,"detail":"Ready."}'
            )

    def fake_urlopen(request, **_kwargs):
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client.health()

    assert captured["url"] == "https://colab.example/health"


def test_remote_dns_error_identifies_stale_or_invalid_tunnel_host(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError(socket.gaierror("name not resolved"))
        ),
    )

    with pytest.raises(RemoteInferenceError, match="hostname could not be resolved"):
        create_client().health()


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (401, b'{"detail":"Invalid API key."}', "Colab rejected the inference request"),
        (404, b'{"detail":"Not found"}', "Set LLM_BASE_URL to the tunnel origin"),
        (503, b'{"detail":"The model is loading."}', "model is loading"),
        (530, b"<html>Cloudflare cannot reach origin</html>", "Cloudflare cannot reach"),
    ],
)
def test_remote_http_errors_have_safe_actionable_diagnostics(
    monkeypatch,
    status: int,
    body: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError(
                "https://colab.example/health",
                status,
                "Remote error",
                {},
                BytesIO(body),
            )
        ),
    )

    with pytest.raises(RemoteInferenceError, match=message):
        create_client().health()


def test_remote_unconfigured_service_is_unavailable() -> None:
    client = create_client(llm_base_url="")

    with pytest.raises(RemoteInferenceError, match="not configured"):
        client.health()


def test_remote_timeout_is_reported_as_unavailable(monkeypatch) -> None:
    client = create_client()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    with pytest.raises(RemoteInferenceError, match="could not be reached"):
        client.health()


def test_invalid_remote_health_contract_is_rejected(monkeypatch) -> None:
    client = create_client()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "model_status": "ready",
            "model_ready": "yes",
            "model_name": "Qwen",
            "gpu_available": True,
        },
    )

    with pytest.raises(RemoteResponseError, match="invalid health response"):
        client.health()


def test_invalid_remote_analysis_response_is_rejected(monkeypatch) -> None:
    client = create_client()
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"text": None})

    with pytest.raises(RemoteResponseError, match="invalid analysis response"):
        client.complete("prompt")


def test_invalid_remote_qa_response_is_rejected(monkeypatch) -> None:
    client = create_client()
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"answer": None})

    with pytest.raises(RemoteResponseError, match="invalid answer response"):
        client.answer("meeting context", "What was decided?")


def test_invalid_remote_json_is_rejected(monkeypatch) -> None:
    client = create_client()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"not json"

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(RemoteResponseError, match="invalid response"):
        client.health()
