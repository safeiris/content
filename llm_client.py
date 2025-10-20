# -*- coding: utf-8 -*-
"""Simple wrapper around chat completion providers with retries and sane defaults."""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

import httpx

from config import OPENAI_API_KEY


DEFAULT_MODEL = "gpt-5"
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [0.5, 1.0, 2.0]

MODEL_PROVIDER_MAP = {
    "gpt-5": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
}

PROVIDER_API_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
}


def _resolve_model_name(model: Optional[str]) -> str:
    env_model = os.getenv("LLM_MODEL")
    candidate = (model or env_model or DEFAULT_MODEL).strip()
    return candidate or DEFAULT_MODEL


def _resolve_provider(model_name: str) -> str:
    return MODEL_PROVIDER_MAP.get(model_name, "openai")


def _resolve_api_key(provider: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if api_key:
        api_key = api_key.strip()
    if not api_key:
        raise RuntimeError(
            "Не задан API-ключ для OpenAI. Установите переменную окружения OPENAI_API_KEY."
        )
    return api_key


def _resolve_backoff_schedule(override: Optional[List[float]]) -> List[float]:
    if override:
        return override
    raw = os.getenv("LLM_RETRY_BACKOFF")
    if raw:
        values: List[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                continue
        if values:
            return values
    return BACKOFF_SCHEDULE


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    return False


def _describe_error(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status:
        return str(status)
    if isinstance(exc, httpx.HTTPStatusError):  # pragma: no cover - depends on response type
        return str(exc.response.status_code)
    return exc.__class__.__name__


def generate(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1400,
    timeout_s: int = 60,
    backoff_schedule: Optional[List[float]] = None,
) -> str:
    """Call the configured LLM and return the article text."""

    if not messages:
        raise ValueError("messages must not be empty")

    model_name = _resolve_model_name(model)
    provider = _resolve_provider(model_name)
    api_key = _resolve_api_key(provider)
    api_url = PROVIDER_API_URLS.get(provider)
    if not api_url:
        raise RuntimeError(f"Неизвестный провайдер для модели '{model_name}'")

    timeout = httpx.Timeout(timeout_s)
    http_client = httpx.Client(timeout=timeout)

    last_error: Optional[BaseException] = None
    schedule = _resolve_backoff_schedule(backoff_schedule)
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = http_client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("Модель не вернула варианты ответа.")
                choice = choices[0]
                message = choice.get("message") if isinstance(choice, dict) else None
                content = None
                if isinstance(message, dict):
                    content = message.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    content = "".join(text_parts)
                if not isinstance(content, str):
                    raise RuntimeError("Модель вернула неожиданный формат ответа.")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, KeyboardInterrupt):  # pragma: no cover - respect interrupts
                    raise
                last_error = exc
                should_retry = _should_retry(exc)
                if attempt >= MAX_RETRIES or not should_retry:
                    break
                sleep_for = schedule[min(attempt - 1, len(schedule) - 1)]
                reason = _describe_error(exc)
                print(f"[llm_client] retry #{attempt} reason: {reason}; sleeping {sleep_for}s", file=sys.stderr)
                time.sleep(sleep_for)
        if last_error:
            if isinstance(last_error, httpx.HTTPStatusError):
                status_code = last_error.response.status_code
                detail = ""
                try:
                    payload = last_error.response.json()
                    if isinstance(payload, dict):
                        detail = (
                            payload.get("error", {}).get("message")
                            if isinstance(payload.get("error"), dict)
                            else payload.get("message", "")
                        ) or ""
                except ValueError:
                    detail = last_error.response.text.strip()
                message = f"Ошибка сервиса OpenAI: HTTP {status_code}"
                if detail:
                    message = f"{message} — {detail}"
                raise RuntimeError(message) from last_error
            if isinstance(last_error, httpx.TimeoutException):
                raise RuntimeError(
                    "Сетевой таймаут при обращении к модели. Проверьте соединение и повторите попытку."
                ) from last_error
            if isinstance(last_error, httpx.TransportError):
                raise RuntimeError(
                    "Сетевой сбой при обращении к модели. Проверьте соединение и повторите попытку."
                ) from last_error
            raise RuntimeError(f"Не удалось получить ответ модели: {last_error}") from last_error
        raise RuntimeError("Модель не вернула ответ.")
    finally:
        http_client.close()


__all__ = ["generate", "DEFAULT_MODEL"]
