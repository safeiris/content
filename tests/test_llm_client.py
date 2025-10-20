# -*- coding: utf-8 -*-
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_client import generate


class DummyResponse:
    def __init__(self):
        self._json = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ]
        }

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


class DummyClient:
    def __init__(self):
        self.last_request = None

    def post(self, url, headers=None, json=None):
        self.last_request = {
            "url": url,
            "headers": headers,
            "json": json,
        }
        return DummyResponse()

    def close(self):
        return None


def _run_and_capture_request(model_name: str):
    dummy_client = DummyClient()
    with patch("llm_client.httpx.Client", return_value=dummy_client):
        result = generate(
            messages=[{"role": "user", "content": "ping"}],
            model=model_name,
            temperature=0,
            max_tokens=42,
        )
    return result, dummy_client.last_request


def test_generate_uses_max_tokens_for_non_gpt5():
    _, request_payload = _run_and_capture_request("gpt-4o")
    assert request_payload["json"]["max_tokens"] == 42
    assert "max_completion_tokens" not in request_payload["json"]
    assert request_payload["json"]["temperature"] == 0


def test_generate_uses_max_completion_tokens_for_gpt5():
    _, request_payload = _run_and_capture_request("gpt-5-preview")
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
