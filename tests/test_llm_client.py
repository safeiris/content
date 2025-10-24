# -*- coding: utf-8 -*-
from pathlib import Path
from unittest.mock import patch
import httpx
import pytest
import sys

import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_client import (
    DEFAULT_RESPONSES_TEXT_FORMAT,
    G5_MAX_OUTPUT_TOKENS_MAX,
    GenerationResult,
    generate,
)


@pytest.fixture(autouse=True)
def _force_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class DummyResponse:
    def __init__(self, payload=None, *, status_code=200, text="", raise_for_status_exc=None):
        if payload is None:
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                        }
                    }
                ]
            }
        self._json = payload
        self.status_code = status_code
        self.text = text
        self._raise_for_status_exc = raise_for_status_exc

    def raise_for_status(self):
        if self._raise_for_status_exc:
            raise self._raise_for_status_exc
        return None

    def json(self):
        return self._json


class DummyClient:
    def __init__(self, payloads=None, availability=None, poll_payloads=None):
        self.last_request = None
        self.requests = []
        self.last_probe = None
        self.last_poll = None
        self.payloads = payloads or []
        self.availability = availability or []
        self.poll_payloads = poll_payloads or []
        self.call_count = 0
        self.probe_count = 0
        self.poll_count = 0

    def post(self, url, headers=None, json=None, **kwargs):
        request = {
            "url": url,
            "headers": headers,
            "json": json,
        }
        self.last_request = request
        self.requests.append(request)
        payload = None
        if self.payloads:
            index = min(self.call_count, len(self.payloads) - 1)
            payload = self.payloads[index]
        self.call_count += 1
        if isinstance(payload, dict) and payload.get("__error__") == "http":
            status = int(payload.get("status", 400))
            response_payload = payload.get("payload", {})
            text = payload.get("text", "")
            request_obj = httpx.Request("POST", url)
            response_obj = httpx.Response(status, request=request_obj, json=response_payload)
            error = httpx.HTTPStatusError("HTTP error", request=request_obj, response=response_obj)
            return DummyResponse(response_payload, status_code=status, text=text, raise_for_status_exc=error)
        return DummyResponse(payload)

    def get(self, url, headers=None, **kwargs):
        if url.startswith("https://api.openai.com/v1/responses"):
            self.last_poll = {
                "url": url,
                "headers": headers,
            }
            payload = {}
            if self.poll_payloads:
                index = min(self.poll_count, len(self.poll_payloads) - 1)
                payload = self.poll_payloads[index]
            self.poll_count += 1
            return DummyResponse(payload)

        self.last_probe = {
            "url": url,
            "headers": headers,
        }
        status_code = 200
        text = ""
        if self.availability:
            entry = self.availability[min(self.probe_count, len(self.availability) - 1)]
            if isinstance(entry, dict):
                status_code = entry.get("status", 200)
                text = entry.get("text", "")
            else:
                status_code = int(entry)
        self.probe_count += 1
        return DummyResponse({"object": "model"}, status_code=status_code, text=text)

    def close(self):
        return None


def _run_and_capture_request(model_name: str, *, payloads=None, availability=None, poll_payloads=None):
    dummy_client = DummyClient(
        payloads=payloads,
        availability=availability,
        poll_payloads=poll_payloads,
    )
    with patch("llm_client.httpx.Client", return_value=dummy_client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model=model_name,
            temperature=0,
            max_tokens=42,
        )
    return result, dummy_client.last_request


def test_generate_uses_max_tokens_for_non_gpt5():
    result, request_payload = _run_and_capture_request("gpt-4o")
    assert isinstance(result, GenerationResult)
    assert result.text == "ok"
    assert result.api_route == "chat"
    assert request_payload["json"]["max_tokens"] == 42
    assert "max_completion_tokens" not in request_payload["json"]
    assert request_payload["json"]["temperature"] == 0
    assert "tool_choice" not in request_payload["json"]
    assert "modalities" not in request_payload["json"]
    assert "response_format" not in request_payload["json"]
    assert request_payload["url"].endswith("/chat/completions")


def test_generate_uses_responses_payload_for_gpt5():
    responses_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    result, request_payload = _run_and_capture_request("gpt-5-preview", payloads=[responses_payload])
    assert isinstance(result, GenerationResult)
    assert result.api_route == "responses"
    assert result.schema == "responses.output_text"
    payload = request_payload["json"]
    assert payload["max_output_tokens"] == 42
    assert "modalities" not in payload
    assert "messages" not in payload
    assert payload["model"] == "gpt-5-preview"
    assert payload["input"] == "ping"
    assert "text" in payload
    assert payload["text"]["format"] == DEFAULT_RESPONSES_TEXT_FORMAT
    assert "temperature" not in payload
    assert set(payload.keys()) == {"input", "max_output_tokens", "model", "text"}
    assert request_payload["url"].endswith("/responses")


def test_generate_logs_about_temperature_for_gpt5():
    responses_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    dummy_client = DummyClient(payloads=[responses_payload])
    with patch("llm_client.httpx.Client", return_value=dummy_client), patch("llm_client.LOGGER") as mock_logger:
        generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5-super",
            temperature=0.7,
            max_tokens=42,
        )

    mock_logger.info.assert_any_call(
        "responses payload_keys=%s",
        ["input", "max_output_tokens", "model", "text"],
    )
    mock_logger.info.assert_any_call("responses input_len=%d", 4)
    mock_logger.info.assert_any_call("responses max_output_tokens=%s", 42)
    mock_logger.info.assert_any_call(
        "LOG:RESPONSES_PARAM_OMITTED omitted=['temperature'] model=%s",
        "gpt-5-super",
    )


def test_generate_polls_for_incomplete_responses_status():
    initial_payload = {
        "status": "in_progress",
        "id": "resp-123",
        "output": [
            {
                "content": [
                    {"type": "text", "text": ""},
                ]
            }
        ],
    }
    final_payload = {
        "status": "completed",
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "done"},
                ]
            }
        ],
    }
    dummy_client = DummyClient(payloads=[initial_payload], poll_payloads=[final_payload])
    with patch("llm_client.httpx.Client", return_value=dummy_client), patch(
        "llm_client.time.sleep",
        return_value=None,
    ) as mock_sleep:
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",
            temperature=0.2,
            max_tokens=42,
        )

    assert isinstance(result, GenerationResult)
    assert result.text == "done"
    assert result.api_route == "responses"
    assert result.schema == "responses.output_text"
    assert dummy_client.poll_count == 1
    assert dummy_client.last_poll["url"].endswith("/responses/resp-123")
    mock_sleep.assert_called()


def test_generate_sends_minimal_payload_for_gpt5():
    responses_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "ok"},
                ]
            }
        ]
    }
    _, request_payload = _run_and_capture_request("gpt-5-turbo", payloads=[responses_payload])
    payload = request_payload["json"]
    assert payload["model"] == "gpt-5-turbo"
    assert "modalities" not in payload
    assert payload["max_output_tokens"] == 42
    assert payload["input"] == "ping"
    assert "temperature" not in payload
    assert set(payload.keys()) == {
        "model",
        "input",
        "max_output_tokens",
        "text",
    }


def test_generate_retries_when_unknown_parameter_reported():
    error_entry = {
        "__error__": "http",
        "status": 400,
        "payload": {"error": {"message": "Unknown parameter: 'modalities'"}},
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
    dummy_client = DummyClient(payloads=[error_entry, success_payload])
    with patch("llm_client.httpx.Client", return_value=dummy_client), patch(
        "llm_client.LOGGER"
    ) as mock_logger:
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",  # ensure responses route
            temperature=0.2,
            max_tokens=42,
        )

    assert isinstance(result, GenerationResult)
    assert result.retry_used is True
    assert dummy_client.call_count == 2
    mock_logger.warning.assert_any_call(
        "retry=shim_unknown_param stripped='%s'",
        "modalities",
    )


def test_generate_logs_responses_error_and_artifacts():
    error_entry = {
        "__error__": "http",
        "status": 400,
        "payload": {"error": {"type": "invalid_request_error", "message": "invalid field"}},
    }
    fallback_payload = {
        "choices": [
            {
                "message": {
                    "content": "fallback ok",
                }
            }
        ]
    }
    dummy_client = DummyClient(payloads=[error_entry, fallback_payload])
    with patch("llm_client.httpx.Client", return_value=dummy_client), patch(
        "llm_client._store_responses_request_snapshot"
    ) as mock_store_request, patch(
        "llm_client._store_responses_response_snapshot"
    ) as mock_store_response, patch("llm_client.LOGGER") as mock_logger:
        with pytest.raises(RuntimeError) as excinfo:
            generate(
                messages=[{"role": "user", "content": "ping"}],
                model="gpt-5",  # ensure Responses route
                temperature=0.2,
                max_tokens=42,
            )

    assert "HTTP 400" in str(excinfo.value)
    mock_store_request.assert_called()
    mock_store_response.assert_called()
    logged_errors = [call for call in mock_logger.error.call_args_list if "Responses API error" in call[0][0]]
    assert logged_errors, "Expected Responses API error log entry"


def test_generate_collects_text_from_content_parts():
    payloads = [
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Part1"},
                            {"type": "text", "text": "Part2"},
                        ]
                    }
                }
            ]
        }
    ]
    result, _ = _run_and_capture_request("gpt-4o", payloads=payloads)
    assert result.text == "Part1\n\nPart2"


def test_generate_falls_back_to_gpt4_when_gpt5_empty():
    empty_payload = {
        "output": [
            {
                "content": [
                    {"type": "text", "text": "   "},
                ]
            }
        ]
    }
    fallback_payload = {
        "choices": [
            {
                "message": {
                    "content": "from fallback",
                }
            }
        ]
    }
    client = DummyClient(payloads=[empty_payload, fallback_payload])
    with patch("llm_client.httpx.Client", return_value=client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5-large",
            temperature=0,
            max_tokens=42,
        )

    assert result.model_used == "gpt-4o"
    assert result.fallback_used == "gpt-4o"
    assert result.retry_used is False
    assert result.text == "from fallback"
    assert result.fallback_reason == "empty_completion_gpt5_responses"
    assert result.api_route == "chat"


def test_generate_falls_back_when_gpt5_unavailable(monkeypatch):
    monkeypatch.setattr("llm_client.FORCE_MODEL", False)
    fallback_payload = {
        "choices": [
            {
                "message": {
                    "content": "fallback ok",
                }
            }
        ]
    }
    client = DummyClient(payloads=[fallback_payload], availability=[403])
    with patch("llm_client.httpx.Client", return_value=client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",
            temperature=0.2,
            max_tokens=42,
        )

    assert result.model_used == "gpt-4o"
    assert result.fallback_used == "gpt-4o"
    assert result.retry_used is False
    assert result.text == "fallback ok"
    assert result.fallback_reason == "model_unavailable"


def test_generate_escalates_max_tokens_when_truncated():
    initial_payload = {
        "id": "resp-1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "content": [
                    {"type": "text", "text": ""},
                ]
            }
        ],
    }
    final_payload = {
        "status": "completed",
        "output": [
            {
                "content": [
                    {"type": "text", "text": "expanded"},
                ]
            }
        ],
    }
    client = DummyClient(payloads=[initial_payload, final_payload])
    with patch("llm_client.httpx.Client", return_value=client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",
            temperature=0.1,
        )

    assert isinstance(result, GenerationResult)
    assert result.text == "expanded"
    assert result.retry_used is True
    assert client.call_count == 2
    first_tokens = client.requests[0]["json"]["max_output_tokens"]
    second_tokens = client.requests[1]["json"]["max_output_tokens"]
    cap = G5_MAX_OUTPUT_TOKENS_MAX if G5_MAX_OUTPUT_TOKENS_MAX > 0 else None
    expected_second = max(first_tokens + 600, int(first_tokens * 1.5))
    if cap is not None:
        expected_second = min(expected_second, cap)
    assert second_tokens == expected_second
    assert client.requests[1]["json"].get("previous_response_id") == "resp-1"
    assert client.requests[0]["json"].get("previous_response_id") is None


def test_generate_accepts_incomplete_with_valid_json():
    skeleton_payload = {
        "intro": "Вступление",
        "main": ["Блок 1", "Блок 2", "Блок 3"],
        "faq": [
            {"q": f"Вопрос {idx}", "a": f"Ответ {idx}. Детали."}
            for idx in range(1, 6)
        ],
        "conclusion": "Вывод",
    }
    skeleton_text = json.dumps(skeleton_payload, ensure_ascii=False)
    incomplete_payload = {
        "id": "resp-accept",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "content": [
                    {"type": "text", "text": skeleton_text},
                ]
            }
        ],
    }
    client = DummyClient(payloads=[incomplete_payload])
    with patch("llm_client.httpx.Client", return_value=client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5",
            temperature=0.1,
        )

    assert client.call_count == 1
    assert isinstance(result, GenerationResult)
    assert result.text == skeleton_text
    assert result.metadata is not None
    assert result.metadata.get("status") == "completed"
    assert result.metadata.get("incomplete_reason") in {None, ""}


def test_generate_does_not_shrink_prompt_after_content_started():
    skeleton_payload = {
        "intro": "Вступление",
        "main": ["Блок 1", "Блок 2", "Блок 3"],
        "faq": [
            {"q": f"Вопрос {idx}", "a": f"Ответ {idx}. Детали."}
            for idx in range(1, 6)
        ],
        "conclusion": "Вывод",
    }
    skeleton_text = json.dumps(skeleton_payload, ensure_ascii=False)
    first_payload = {
        "id": "resp-partial",
        "status": "incomplete",
        "incomplete_details": {"reason": "safety"},
        "output": [
            {
                "content": [
                    {"type": "text", "text": "partial draft"},
                ]
            }
        ],
    }
    final_payload = {
        "status": "completed",
        "output": [
            {
                "content": [
                    {"type": "text", "text": skeleton_text},
                ]
            }
        ],
    }
    messages = [
        {
            "role": "system",
            "content": "A\nB\nA",
        },
        {
            "role": "user",
            "content": "C\nC\nD",
        },
    ]
    client = DummyClient(payloads=[first_payload, final_payload])
    with patch("llm_client.httpx.Client", return_value=client):
        result = generate(
            messages=messages,
            model="gpt-5",
            temperature=0.2,
        )

    assert client.call_count == 2
    assert isinstance(result, GenerationResult)
    assert result.text == skeleton_text
    first_input = client.requests[0]["json"]["input"]
    second_input = client.requests[1]["json"]["input"]
    assert second_input == first_input


def test_generate_reports_empty_completion_when_incomplete_at_cap(monkeypatch):
    monkeypatch.setattr("llm_client.G5_MAX_OUTPUT_TOKENS_MAX", 1800)
    initial_payload = {
        "id": "resp-init",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    second_payload = {
        "id": "resp-second",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    client = DummyClient(payloads=[initial_payload, second_payload])
    with patch("llm_client.httpx.Client", return_value=client):
        with pytest.raises(RuntimeError) as excinfo:
            generate(
                messages=[{"role": "user", "content": "ping"}],
                model="gpt-5",
                temperature=0.1,
            )

    message = str(excinfo.value)
    assert "Ответ не помещается" not in message
    assert "Модель не вернула варианты ответа." in message


def test_generate_raises_when_forced_and_gpt5_unavailable(monkeypatch):
    monkeypatch.setattr("llm_client.FORCE_MODEL", True)
    client = DummyClient(payloads=[], availability=[403])
    with patch("llm_client.httpx.Client", return_value=client):
        with pytest.raises(RuntimeError) as excinfo:
            generate(
                messages=[{"role": "user", "content": "ping"}],
                model="gpt-5",
                temperature=0.1,
                max_tokens=42,
            )

    assert "Model GPT-5 not available for this key/plan" in str(excinfo.value)
