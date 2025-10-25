import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import llm_client as llm_client_module  # noqa: E402

from config import LLM_ALLOW_FALLBACK, LLM_MODEL, LLM_ROUTE
from llm_client import (  # noqa: E402
    DEFAULT_RESPONSES_TEXT_FORMAT,
    FALLBACK_RESPONSES_PLAIN_OUTLINE_FORMAT,
    G5_ESCALATION_LADDER,
    GenerationResult,
    generate,
    reset_http_client_cache,
)


@pytest.fixture(autouse=True)
def _force_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _reset_http_clients():
    reset_http_client_cache()
    yield
    reset_http_client_cache()


class DummyResponse:
    def __init__(self, payload=None, *, status_code=200, text="", raise_for_status_exc=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text
        self._raise_for_status_exc = raise_for_status_exc

    def raise_for_status(self):
        if self._raise_for_status_exc:
            raise self._raise_for_status_exc
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, *, responses=None, polls=None, availability=None):
        self.responses = responses or []
        self.polls = polls or []
        self.availability = availability or []
        self.requests = []
        self.probes = []
        self.poll_requests = []
        self._response_index = 0
        self._poll_index = 0
        self._probe_index = 0

    def post(self, url, headers=None, json=None, **kwargs):
        request = {"url": url, "headers": headers, "json": json}
        self.requests.append(request)
        payload = {}
        if self._response_index < len(self.responses):
            payload = self.responses[self._response_index]
        self._response_index += 1
        if isinstance(payload, dict) and payload.get("__error__") == "http":
            status = int(payload.get("status", 400))
            data = payload.get("payload", {})
            request_obj = httpx.Request("POST", url)
            response_obj = httpx.Response(status, request=request_obj, json=data)
            error = httpx.HTTPStatusError("HTTP error", request=request_obj, response=response_obj)
            return DummyResponse(data, status_code=status, text=payload.get("text", ""), raise_for_status_exc=error)
        return DummyResponse(payload)

    def get(self, url, headers=None, **kwargs):
        if url.startswith("https://api.openai.com/v1/responses/"):
            entry = {}
            if self._poll_index < len(self.polls):
                entry = self.polls[self._poll_index]
            self._poll_index += 1
            self.poll_requests.append({"url": url, "headers": headers})
            return DummyResponse(entry)
        entry = self.availability[self._probe_index] if self._probe_index < len(self.availability) else {"status": 200}
        self._probe_index += 1
        self.probes.append({"url": url, "headers": headers})
        if isinstance(entry, dict):
            status = entry.get("status", 200)
            text = entry.get("text", "")
        else:
            status = int(entry)
            text = ""
        return DummyResponse({"object": "model"}, status_code=status, text=text)

    def close(self):
        return None


def _generate_with_dummy(
    *,
    responses=None,
    polls=None,
    availability=None,
    model="gpt-5",
    max_tokens=64,
):
    dummy_client = DummyClient(responses=responses, polls=polls, availability=availability)
    with patch("llm_client.httpx.Client", return_value=dummy_client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model=model,
            max_tokens=max_tokens,
        )
    return result, dummy_client


def test_generate_rejects_non_gpt5_model():
    with pytest.raises(RuntimeError):
        _generate_with_dummy(model="gpt-4o")


def test_generate_uses_responses_payload_for_gpt5():
    payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    result, client = _generate_with_dummy(responses=[payload])
    request_payload = client.requests[-1]["json"]
    assert request_payload["model"] == "gpt-5"
    assert request_payload["input"] == "ping"
    assert request_payload["max_output_tokens"] == 64
    assert request_payload["text"]["format"] == DEFAULT_RESPONSES_TEXT_FORMAT
    assert "temperature" not in request_payload
    metadata = result.metadata or {}
    assert metadata.get("model_effective") == LLM_MODEL
    assert metadata.get("api_route") == LLM_ROUTE
    assert metadata.get("allow_fallback") is LLM_ALLOW_FALLBACK
    assert metadata.get("temperature_applied") is False
    assert metadata.get("escalation_caps") == list(G5_ESCALATION_LADDER)
    assert metadata.get("max_output_tokens_applied") == 64


def test_generate_polls_until_completion():
    in_progress = {"status": "in_progress", "id": "resp-1", "output": []}
    final_payload = {
        "status": "completed",
        "output": [
            {
                "content": [
                    {"type": "text", "text": "done"},
                ]
            }
        ],
    }
    with patch("llm_client.time.sleep", return_value=None):
        result, client = _generate_with_dummy(responses=[in_progress], polls=[final_payload])
    assert result.text == "done"
    assert client.poll_requests
    assert client.poll_requests[0]["url"].endswith("/responses/resp-1")


def test_generate_retries_unknown_parameter():
    error_entry = {"__error__": "http", "status": 400, "payload": {"error": {"message": "Unknown parameter: 'modalities'"}}}
    success_entry = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    with patch("llm_client.LOGGER") as mock_logger:
        result, client = _generate_with_dummy(responses=[error_entry, success_entry])
    assert isinstance(result, GenerationResult)
    assert len(client.requests) == 2
    stripped = any("modalities" in call.args[1] if call.args else False for call in mock_logger.warning.call_args_list)
    assert stripped


def test_generate_raises_when_responses_empty_after_escalation():
    empty_payload = {"output": [{"content": [{"type": "text", "text": "   "}]}]}
    with patch("llm_client.LOGGER"):
        with pytest.raises(RuntimeError) as excinfo:
            _generate_with_dummy(responses=[empty_payload, empty_payload, empty_payload])
    assert "Попробуйте повторить" in str(excinfo.value)


def test_generate_retries_empty_completion_with_fallback():
    empty_payload = {"output": [{"content": [{"type": "text", "text": "   "}]}]}
    success_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "Готовый текст"},
                ]
            }
        ],
    }
    responses = [empty_payload, empty_payload, success_payload]
    result, client = _generate_with_dummy(
        responses=responses,
        max_tokens=100,
    )
    assert isinstance(result, GenerationResult)
    assert result.retry_used is True
    assert result.fallback_used == "plain_outline"
    assert result.fallback_reason == "empty_completion_fallback"
    assert len(client.requests) == 3
    primary_request = client.requests[0]["json"]
    retry_request = client.requests[1]["json"]
    fallback_request = client.requests[2]["json"]
    assert primary_request["max_output_tokens"] == 100
    assert retry_request["max_output_tokens"] == 85
    assert fallback_request["max_output_tokens"] == 76
    assert "previous_response_id" not in retry_request
    assert fallback_request["text"]["format"] == FALLBACK_RESPONSES_PLAIN_OUTLINE_FORMAT


def test_generate_accepts_incomplete_with_text():
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "content": [
                    {"type": "text", "text": "{\"intro\": \"Hello\"}"},
                ]
            }
        ],
    }
    result, client = _generate_with_dummy(responses=[payload], max_tokens=120)
    assert isinstance(result, GenerationResult)
    assert result.text.strip() == '{"intro": "Hello"}'
    metadata = result.metadata or {}
    assert metadata.get("status") == "completed"
    assert metadata.get("incomplete_reason") in (None, "")
    assert metadata.get("completion_warning") == "max_output_tokens"
    flags = metadata.get("degradation_flags") or []
    assert "draft_max_tokens" in flags
    assert len(client.requests) == 1


def test_responses_continue_includes_model_and_tokens(monkeypatch):
    monkeypatch.setattr("llm_client.G5_MAX_OUTPUT_TOKENS_BASE", 64, raising=False)
    monkeypatch.setattr("llm_client.G5_MAX_OUTPUT_TOKENS_STEP1", 96, raising=False)
    monkeypatch.setattr("llm_client.G5_MAX_OUTPUT_TOKENS_STEP2", 128, raising=False)
    monkeypatch.setattr("llm_client.G5_MAX_OUTPUT_TOKENS_MAX", 128, raising=False)
    monkeypatch.setattr("llm_client.G5_ESCALATION_LADDER", (64, 96, 128), raising=False)

    incomplete_payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "id": "resp-1",
        "output": [],
    }
    final_payload = {
        "status": "completed",
        "output": [
            {
                "content": [
                    {"type": "text", "text": "Готовый текст"},
                ]
            }
        ],
    }

    with patch("llm_client.LOGGER"):
        result, client = _generate_with_dummy(
            responses=[incomplete_payload, final_payload],
            max_tokens=64,
        )

    assert len(client.requests) == 2
    primary_payload = client.requests[0]["json"]
    continue_payload = client.requests[1]["json"]

    assert continue_payload["previous_response_id"] == "resp-1"
    assert continue_payload["model"] == primary_payload["model"]
    assert continue_payload["text"]["format"] == primary_payload["text"]["format"]
    expected_tokens = llm_client_module.G5_ESCALATION_LADDER[1]
    assert continue_payload["max_output_tokens"] == expected_tokens

    metadata = result.metadata or {}
    flags = metadata.get("degradation_flags") or []
    assert "draft_max_tokens" in flags
    assert metadata.get("completion_warning") == "max_output_tokens"
