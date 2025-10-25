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
        self.timeout: Optional[httpx.Timeout] = None

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
    responses = [_response(200, payload)]
    client = DummyHealthClient(responses)

    def _client_factory(timeout=None):
        client.timeout = timeout
        return client

    monkeypatch.setattr(orchestrate.httpx, "Client", _client_factory)

    result = orchestrate._run_health_ping()

    assert result["ok"] is True
    assert result["route"] == LLM_ROUTE
    assert result["fallback_used"] is LLM_ALLOW_FALLBACK
    expected_label = f"Responses OK (gpt-5, {orchestrate.HEALTH_INITIAL_MAX_TOKENS} токенов)"
    assert expected_label in result["message"]

    assert isinstance(client.timeout, httpx.Timeout)
    assert client.timeout.read == 40.0
    assert client.timeout.connect == 10.0
    assert client.timeout.write == 10.0
    assert client.timeout.pool == 10.0

    assert len(client.requests) == 1
    request_payload = client.requests[0]["json"]
    assert request_payload["max_output_tokens"] == orchestrate.HEALTH_INITIAL_MAX_TOKENS
    assert request_payload["input"] == orchestrate.HEALTH_PROMPT
    assert request_payload["text"]["format"]["type"] == "text"
    assert "temperature" not in request_payload


def test_health_ping_incomplete_max_tokens(monkeypatch):
    payload = {
        "status": "incomplete",
        "output_text": "partial",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    responses = [_response(200, payload)]
    client = DummyHealthClient(responses)

    def _client_factory(timeout=None):
        client.timeout = timeout
        return client

    monkeypatch.setattr(orchestrate.httpx, "Client", _client_factory)

    result = orchestrate._run_health_ping()

    assert result["ok"] is True
    assert result["status"] == "incomplete"
    assert result["incomplete_reason"] == "max_output_tokens"
    assert "с обрезкой по max_output_tokens" in result["message"]


def test_health_ping_5xx_failure(monkeypatch):
    responses = [_response(502, None, text="Bad gateway")]
    client = DummyHealthClient(responses)
    def _client_factory(timeout=None):
        client.timeout = timeout
        return client

    monkeypatch.setattr(orchestrate.httpx, "Client", _client_factory)

    result = orchestrate._run_health_ping()

    assert result["ok"] is False
    assert "HTTP 502" in result["message"]
    assert len(client.requests) == 1


def test_health_ping_400_invalid_max_tokens(monkeypatch):
    payload = {
        "error": {
            "message": "max_output_tokens too small; expected >= 64",
        }
    }
    responses = [_response(400, payload)]
    client = DummyHealthClient(responses)

    def _client_factory(timeout=None):
        client.timeout = timeout
        return client

    monkeypatch.setattr(orchestrate.httpx, "Client", _client_factory)

    result = orchestrate._run_health_ping()

    assert result["ok"] is False
    assert result["status"] == "degraded"
    assert result["message"] == "LLM degraded: 400 invalid max_output_tokens (raised to >=64)"
    assert len(client.requests) == 1


def test_health_ping_timeout_degraded(monkeypatch):
    class TimeoutClient:
        def __init__(self):
            self.calls = 0
            self.timeout: Optional[httpx.Timeout] = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, **kwargs):
            self.calls += 1
            raise httpx.ReadTimeout("timeout", request=httpx.Request("POST", url))

    timeout_client = TimeoutClient()

    def _client_factory(timeout=None):
        timeout_client.timeout = timeout
        return timeout_client

    monkeypatch.setattr(orchestrate.httpx, "Client", _client_factory)
    monkeypatch.setattr(orchestrate.random, "uniform", lambda a, b: 0.4)
    monkeypatch.setattr(orchestrate.time, "sleep", lambda _: None)

    result = orchestrate._run_health_ping()

    assert result["ok"] is False
    assert result["status"] == "degraded"
    assert result["attempts"] == 2
    assert result["message"] == "LLM degraded: timeout on health ping (2 attempts, read=40s)."
    assert timeout_client.calls == 2
