"""Small HTTP client and LangChain adapter for remote Colab inference."""

import http.client
import json
import re
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict, Field

from backend.app.chains.errors import RemoteInferenceError, RemoteResponseError
from backend.app.config import Settings


class RemoteInferenceClient:
    def __init__(self, settings: Settings):
        self.base_url, self.base_url_error = _normalize_base_url(settings.llm_base_url)
        self.api_key = settings.llm_api_key
        self.request_timeout = settings.llm_request_timeout
        self.health_timeout = settings.llm_health_timeout
        self.default_model_name = settings.remote_model_name

    def health(self) -> dict[str, Any]:
        data = self._request("GET", "/health", timeout=self.health_timeout)
        status = data.get("model_status")
        if status not in {"ready", "loading", "unavailable"}:
            raise RemoteResponseError("The Colab service returned an invalid health response.")
        model_ready = data.get("model_ready")
        if not isinstance(model_ready, bool) or (model_ready and status != "ready"):
            raise RemoteResponseError("The Colab service returned an invalid health response.")
        if not isinstance(data.get("gpu_available"), bool):
            raise RemoteResponseError("The Colab service returned an invalid health response.")
        model_name = data.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise RemoteResponseError("The Colab service returned an invalid health response.")
        detail = data.get("detail", "")
        if not isinstance(detail, str):
            raise RemoteResponseError("The Colab service returned an invalid health response.")
        if status == "ready" and not model_ready:
            status = "unavailable"
            detail = "The Colab model has not confirmed readiness."
        return {
            "model_status": status,
            "model_name": model_name,
            "gpu_available": data["gpu_available"],
            "gpu_name": data.get("gpu_name") if isinstance(data.get("gpu_name"), str) else None,
            "gpu_vram_gb": data.get("gpu_vram_gb"),
            "detail": _safe_detail(detail),
        }

    def complete(self, prompt: str) -> str:
        data = self._request("POST", "/analyse", {"prompt": prompt})
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RemoteResponseError("The Colab service returned an invalid analysis response.")
        return text

    def answer(self, context: str, question: str) -> str:
        data = self._request("POST", "/qa", {"context": context, "question": question})
        text = data.get("answer")
        if not isinstance(text, str) or not text.strip():
            raise RemoteResponseError("The Colab service returned an invalid answer response.")
        return text.strip()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        if self.base_url_error:
            raise RemoteInferenceError(self.base_url_error)
        if not self.base_url:
            raise RemoteInferenceError(
                "The Colab AI service is not configured. Start the Colab notebook and set LLM_BASE_URL."
            )
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout if timeout is None else timeout,
            ) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            safe_message = _http_error_message(error)
            raise RemoteInferenceError(safe_message) from error
        except (TimeoutError, socket.timeout) as error:
            raise RemoteInferenceError(
                "The Colab AI service could not be reached before the request timed out. "
                "Check that the runtime and tunnel are active."
            ) from error
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, socket.gaierror):
                message = (
                    "The Colab tunnel hostname could not be resolved. "
                    "Copy the current HTTPS endpoint from the active notebook."
                )
            elif isinstance(reason, (TimeoutError, socket.timeout)):
                message = (
                    "The Colab AI service could not be reached before the request timed out. "
                    "Check that the runtime and tunnel are active."
                )
            elif isinstance(reason, ConnectionRefusedError):
                message = (
                    "The Colab tunnel connection was refused. "
                    "Check that the inference server and tunnel are running."
                )
            elif isinstance(reason, ssl.SSLError):
                message = (
                    "The Colab tunnel TLS connection failed. "
                    "Copy the current HTTPS endpoint from the active notebook."
                )
            else:
                message = (
                    "The Colab AI service could not be reached. "
                    "Check that the runtime and tunnel are active."
                )
            raise RemoteInferenceError(message) from error
        except (OSError, http.client.HTTPException) as error:
            raise RemoteInferenceError(
                "The Colab AI service could not be reached. Check that the runtime and tunnel are active."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RemoteResponseError(
                "The Colab service returned an invalid response."
            ) from error
        if not isinstance(result, dict):
            raise RemoteResponseError("The Colab service returned an invalid response.")
        return result


def _normalize_base_url(value: str) -> tuple[str, str | None]:
    base_url = value.strip()
    if not base_url:
        return "", None
    if re.search(r"LLM_API_KEY\s*=", base_url, re.IGNORECASE):
        return "", (
            "LLM_BASE_URL appears to contain another environment-variable assignment. "
            "Put LLM_BASE_URL and LLM_API_KEY on separate lines, and set the URL to "
            "the HTTPS Colab tunnel origin only."
        )
    if any(character.isspace() for character in base_url):
        return "", (
            "Invalid LLM_BASE_URL. Set it to the HTTPS Colab tunnel origin only, "
            "without spaces, a path, or a query string."
        )
    parsed = None
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        hostname = None
    if (
        parsed is None
        or parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return "", (
            "Invalid LLM_BASE_URL. Set it to the HTTPS Colab tunnel origin only, "
            "without credentials, a path, or a query string."
        )
    return f"https://{parsed.netloc}".rstrip("/"), None


def _safe_detail(value: str) -> str:
    if (
        len(value) > 300
        or "\n" in value
        or "\r" in value
        or re.search(r"traceback|stack trace|exception:|file \".*\", line \d+", value, re.I)
    ):
        return ""
    return value.strip()


def _http_error_message(error: urllib.error.HTTPError) -> str:
    if error.code in {401, 403}:
        return (
            "Colab rejected the inference request. Check that LLM_API_KEY matches "
            "the key printed by the active notebook."
        )
    if error.code == 404:
        return (
            "The Colab service returned 404. Set LLM_BASE_URL to the tunnel origin "
            "without an API path; MeetingMind calls /health, /analyse, and /qa."
        )

    detail = _http_error_detail(error)
    if error.code == 503:
        if detail:
            return f"The Colab inference service is not ready: {detail}"
        return (
            "The Colab inference service is temporarily unavailable. "
            "Check model readiness and the notebook logs."
        )
    if error.code in {502, 530, 521, 522, 523, 524}:
        return (
            "Cloudflare cannot reach the Colab inference server. Confirm the runtime, "
            "HTTP server, and tunnel are active, then copy the current tunnel URL."
        )
    if 500 <= error.code:
        return (
            f"The Colab inference service returned HTTP {error.code}. "
            "Check the notebook server logs."
        )
    return detail or f"The Colab inference service returned HTTP {error.code}."


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return _safe_detail(detail) if isinstance(detail, str) else ""


class RemoteTransformersLLM(LLM):
    """Adapts the Colab text-completion endpoint to existing LangChain chains."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(exclude=True)
    model_name: str

    @property
    def _llm_type(self) -> str:
        return "remote-colab-transformers"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        _ = kwargs
        answer = self.client.complete(prompt)
        for stop_text in stop or []:
            if stop_text in answer:
                answer = answer.split(stop_text, 1)[0].rstrip()
        return answer


class RemoteQAChain:
    def __init__(self, client: RemoteInferenceClient):
        self.client = client

    def invoke(self, inputs: dict[str, str]) -> dict[str, str]:
        return {
            "text": self.client.answer(inputs["context"], inputs["question"]),
        }
