# -*- coding: utf-8 -*-
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_client import GenerationResult, generate


class DummyResponse:
    def __init__(self, payload=None):
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

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


class DummyClient:
    def __init__(self, payloads=None):
        self.last_request = None
        self.payloads = payloads or []
        self.call_count = 0

    def post(self, url, headers=None, json=None):
        self.last_request = {
            "url": url,
            "headers": headers,
            "json": json,
        }
        payload = None
        if self.payloads:
            index = min(self.call_count, len(self.payloads) - 1)
            payload = self.payloads[index]
        self.call_count += 1
        return DummyResponse(payload)

    def close(self):
        return None


def _run_and_capture_request(model_name: str, *, payloads=None):
    dummy_client = DummyClient(payloads=payloads)
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
    assert request_payload["json"]["max_tokens"] == 42
    assert "max_completion_tokens" not in request_payload["json"]
    assert request_payload["json"]["temperature"] == 0


def test_generate_uses_max_completion_tokens_for_gpt5():
    result, request_payload = _run_and_capture_request("gpt-5-preview")
    assert isinstance(result, GenerationResult)
    assert request_payload["json"]["max_completion_tokens"] == 42
    assert "max_tokens" not in request_payload["json"]
    assert "temperature" not in request_payload["json"]


def test_generate_logs_about_temperature_for_gpt5():
    dummy_client = DummyClient()
    with patch("llm_client.httpx.Client", return_value=dummy_client), patch("llm_client.LOGGER") as mock_logger:
        generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5-super",
            temperature=0.7,
            max_tokens=42,
        )

    mock_logger.info.assert_called_once_with("temperature is ignored for GPT-5; using default")


def test_generate_disables_tools_for_gpt5():
    _, request_payload = _run_and_capture_request("gpt-5-turbo")
    payload = request_payload["json"]
    assert payload["tool_choice"] == "none"
    assert payload["response_format"] == {"type": "text"}
    assert payload["modalities"] == ["text"]


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
        "choices": [
            {
                "message": {
                    "content": "   ",
                }
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
    client = DummyClient(payloads=[empty_payload, empty_payload, fallback_payload])
    with patch("llm_client.httpx.Client", return_value=client), patch("llm_client.random.uniform", return_value=0.0), patch("llm_client.time.sleep", lambda _seconds: None):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-5-large",
            temperature=0,
            max_tokens=42,
        )

    assert result.model_used == "gpt-4o"
    assert result.fallback_used == "gpt-4o"
    assert result.retry_used is True
    assert result.text == "from fallback"
