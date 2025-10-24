from pathlib import Path
from typing import Iterable, List, Optional

import httpx
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import orchestrate
from config import LLM_ALLOW_FALLBACK, LLM_ROUTE


class DummyHealthClient:
    def __init__(self, responses: Iterable[httpx.Response]):
        self._responses = list(responses)
        self._index = 0
        self.requests: List[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _next(self) -> httpx.Response:
        if self._index >= len(self._responses):
            return self._responses[-1]
        response = self._responses[self._index]
        self._index += 1
        return response

    def get(self, url, headers=None, **kwargs):
        self.requests.append({"method": "GET", "url": url, "headers": headers})
        return self._next()

    def post(self, url, json=None, headers=None, **kwargs):
        self.requests.append({"method": "POST", "url": url, "json": json, "headers": headers})
        return self._next()


@pytest.fixture(autouse=True)
def _force_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _response(
    status_code: int,
    payload: Optional[dict] = None,
    text: str = "",
    *,
    method: str = "POST",
    url: str = orchestrate.RESPONSES_API_URL,
) -> httpx.Response:
    request = httpx.Request(method, url)
    if payload is not None:
        return httpx.Response(status_code, request=request, json=payload)
    return httpx.Response(status_code, request=request, text=text)


def test_health_ping_success(monkeypatch):
    payload = {"status": "completed", "output": [{"content": [{"type": "text", "text": "PONG"}]}]}
    responses = [
        _response(200, {"id": orchestrate.HEALTH_MODEL}, method="GET", url=f"https://api.openai.com/v1/models/{orchestrate.HEALTH_MODEL}"),
        _response(200, payload),
    ]
    client = DummyHealthClient(responses)
    monkeypatch.setattr(orchestrate.httpx, "Client", lambda timeout=None: client)

    result = orchestrate._run_health_ping()

    assert result["ok"] is True
    assert result["route"] == LLM_ROUTE
    assert result["fallback_used"] is LLM_ALLOW_FALLBACK
    expected_label = f"Responses OK (gpt-5, {orchestrate.HEALTH_INITIAL_MAX_TOKENS} токена)"
    assert expected_label in result["message"]

    assert client.requests[0]["method"] == "GET"
    request_payload = client.requests[1]["json"]
    assert request_payload["max_output_tokens"] == orchestrate.HEALTH_INITIAL_MAX_TOKENS
    assert request_payload["text"]["format"]["type"] == "text"
    assert "temperature" not in request_payload


def test_health_ping_incomplete_max_tokens(monkeypatch):
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    }
    responses = [
        _response(200, {"id": orchestrate.HEALTH_MODEL}, method="GET", url=f"https://api.openai.com/v1/models/{orchestrate.HEALTH_MODEL}"),
        _response(200, payload),
    ]
    client = DummyHealthClient(responses)
    monkeypatch.setattr(orchestrate.httpx, "Client", lambda timeout=None: client)

    result = orchestrate._run_health_ping()

    assert result["ok"] is True
    assert "incomplete по лимиту — норм для health" in result["message"]


def test_health_ping_auto_bump(monkeypatch):
    monkeypatch.setattr(orchestrate, "HEALTH_INITIAL_MAX_TOKENS", 12)
    error_payload = {
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid 'max_output_tokens': Expected a value >= 16",
        }
    }
    success_payload = {"status": "completed", "output": [{"content": [{"type": "text", "text": "PONG"}]}]}
    responses = [
        _response(200, {"id": orchestrate.HEALTH_MODEL}, method="GET", url=f"https://api.openai.com/v1/models/{orchestrate.HEALTH_MODEL}"),
        _response(400, error_payload),
        _response(200, success_payload),
    ]
    client = DummyHealthClient(responses)
    monkeypatch.setattr(orchestrate.httpx, "Client", lambda timeout=None: client)

    result = orchestrate._run_health_ping()

    assert result["ok"] is True
    assert f"auto-bump до {orchestrate.HEALTH_MIN_BUMP_TOKENS}" in result["message"]
    assert len(client.requests) == 3
    assert client.requests[1]["json"]["max_output_tokens"] == 12
    assert client.requests[2]["json"]["max_output_tokens"] >= orchestrate.HEALTH_MIN_BUMP_TOKENS


def test_health_ping_5xx_failure(monkeypatch):
    responses = [
        _response(200, {"id": orchestrate.HEALTH_MODEL}, method="GET", url=f"https://api.openai.com/v1/models/{orchestrate.HEALTH_MODEL}"),
        _response(502, None, text="Bad gateway"),
    ]
    client = DummyHealthClient(responses)
    monkeypatch.setattr(orchestrate.httpx, "Client", lambda timeout=None: client)

    result = orchestrate._run_health_ping()

    assert result["ok"] is False
    assert "HTTP 502" in result["message"]
