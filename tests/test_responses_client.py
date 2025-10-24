from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_client import (
    RESPONSES_FORMAT_DEFAULT_NAME,
    build_responses_payload,
    generate,
    sanitize_payload_for_responses,
)


class DummyResponse:
    def __init__(self, *, payload=None, status_code=200, raise_for_status_exc=None):
        if payload is None:
            payload = {
                "output": [
                    {
                        "content": [
                            {"type": "text", "text": "ok"},
                        ]
                    }
                ]
            }
        self._payload = payload
        self.status_code = status_code
        self._raise_for_status_exc = raise_for_status_exc

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        self._response = httpx.Response(
            status_code,
            request=request,
            json=payload,
        )

    def raise_for_status(self):
        if self._raise_for_status_exc:
            raise self._raise_for_status_exc
        if self.status_code >= 400:
            self._response.raise_for_status()
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, payloads=None):
        self.payloads = payloads or []
        self.requests = []
        self.call_count = 0
        self.probe_count = 0

    def post(self, url, headers=None, json=None, **kwargs):
        self.requests.append({"url": url, "headers": headers, "json": json})
        index = min(self.call_count, len(self.payloads) - 1)
        payload = self.payloads[index] if self.payloads else None
        self.call_count += 1

        if isinstance(payload, dict) and payload.get("__error__") == "http":
            status = int(payload.get("status", 400))
            body = payload.get("payload", {})
            request_obj = httpx.Request("POST", url)
            response_obj = httpx.Response(status, request=request_obj, json=body)
            error = httpx.HTTPStatusError("HTTP error", request=request_obj, response=response_obj)
            return DummyResponse(payload=body, status_code=status, raise_for_status_exc=error)

        return DummyResponse(payload=payload)

    def get(self, url, headers=None, **kwargs):
        self.probe_count += 1
        if url.startswith("https://api.openai.com/v1/models/"):
            return DummyResponse(payload={"object": "model"})
        return DummyResponse(payload={})

    def close(self):
        return None


@pytest.fixture(autouse=True)
def _force_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_build_responses_payload_for_gpt5_includes_required_fields():
    payload = build_responses_payload(
        "gpt-5",
        "system message",
        "user message",
        64,
        text_format={"type": "json_schema", "json_schema": {"name": "stub", "schema": {}}},
    )

    assert payload["model"] == "gpt-5"
    assert payload["input"].count("system message") == 1
    assert payload["input"].count("user message") == 1
    assert payload["max_output_tokens"] == 64
    assert "temperature" not in payload
    format_block = payload["text"]["format"]
    assert format_block["type"] == "json_schema"
    assert "json_schema" not in format_block
    assert isinstance(format_block.get("schema"), dict)


def test_sanitize_payload_converts_legacy_json_schema():
    legacy_schema = {
        "type": "object",
        "properties": {
            "intro": {"type": "string"},
        },
        "required": ["intro"],
    }
    payload = {
        "model": "gpt-5",
        "input": "hello",
        "max_output_tokens": 256,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "legacy",
                "json_schema": {"schema": legacy_schema, "strict": True},
            }
        },
    }

    sanitized, _ = sanitize_payload_for_responses(payload)
    format_block = sanitized["text"]["format"]

    assert "json_schema" not in format_block
    sanitized_schema = format_block["schema"]
    assert sanitized_schema is not legacy_schema
    assert sanitized_schema["additionalProperties"] is False
    assert sanitized_schema["properties"] == legacy_schema["properties"]
    assert legacy_schema.get("additionalProperties") is None
    assert format_block["strict"] is True


def test_generate_retries_with_min_token_bump(monkeypatch):
    error_payload = {
        "__error__": "http",
        "status": 400,
        "payload": {
            "error": {
                "message": "Invalid 'max_output_tokens': Expected a value >= 16",
                "type": "invalid_request_error",
            }
        },
    }
    success_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    dummy_client = DummyClient(payloads=[error_payload, success_payload])

    with patch("llm_client.httpx.Client", return_value=dummy_client), patch("llm_client.LOGGER") as mock_logger:
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",
            max_tokens=8,
        )

    assert result.text == "ok"
    assert dummy_client.call_count == 2
    first_request = dummy_client.requests[0]["json"]
    second_request = dummy_client.requests[1]["json"]
    assert first_request["max_output_tokens"] == 8
    assert second_request["max_output_tokens"] >= 24
    mock_logger.warning.assert_any_call("LOG:RESP_RETRY_REASON=max_tokens_min_bump")


def test_generate_retries_on_missing_format_name(monkeypatch):
    error_payload = {
        "__error__": "http",
        "status": 400,
        "payload": {
            "error": {
                "message": "Missing required field text.format.name",
                "type": "invalid_request_error",
            }
        },
    }
    success_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    dummy_client = DummyClient(payloads=[error_payload, success_payload])

    with patch("llm_client.httpx.Client", return_value=dummy_client), patch(
        "llm_client.LOGGER"
    ) as mock_logger:
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",
            max_tokens=64,
        )

    assert result.text == "ok"
    assert dummy_client.call_count == 2
    first_format = dummy_client.requests[0]["json"]["text"]["format"]
    second_format = dummy_client.requests[1]["json"]["text"]["format"]
    assert first_format["name"] == RESPONSES_FORMAT_DEFAULT_NAME
    assert second_format["name"] == RESPONSES_FORMAT_DEFAULT_NAME
    assert "json_schema" not in first_format
    assert isinstance(first_format.get("schema"), dict)
    mock_logger.warning.assert_any_call(
        "RESP_RETRY_REASON=format_name_missing route=responses attempt=%d",
        1,
    )
