# -*- coding: utf-8 -*-
"""Simple wrapper around the OpenAI client with retries and sane defaults."""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [0.5, 1.0, 2.0]


def _resolve_model_name(model: Optional[str]) -> str:
    env_model = os.getenv("LLM_MODEL")
    candidate = (model or env_model or DEFAULT_MODEL).strip()
    return candidate or DEFAULT_MODEL


def _ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не задан API-ключ. Установите переменную окружения OPENAI_API_KEY."
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
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
    if isinstance(exc, httpx.HTTPError):
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

    api_key = _ensure_api_key()
    model_name = _resolve_model_name(model)

    timeout = httpx.Timeout(timeout_s)
    http_client = httpx.Client(timeout=timeout)
    client = OpenAI(api_key=api_key, http_client=http_client)

    last_error: Optional[BaseException] = None
    schedule = _resolve_backoff_schedule(backoff_schedule)
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = response.choices[0].message
                content = choice.get("content") if isinstance(choice, dict) else getattr(choice, "content", None)
                if isinstance(content, list):
                    # Newer SDKs may return list of content parts.
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
            if isinstance(last_error, RateLimitError):
                raise RuntimeError(
                    "Превышен лимит запросов к модели. Попробуйте позже или уменьшите частоту обращений."
                ) from last_error
            if isinstance(last_error, (APIConnectionError, APITimeoutError, httpx.HTTPError)):
                raise RuntimeError("Сетевой сбой при обращении к модели. Проверьте соединение и повторите попытку.") from last_error
            if isinstance(last_error, APIError):
                message = getattr(last_error, "message", str(last_error))
                raise RuntimeError(f"Ошибка сервиса OpenAI: {message}") from last_error
            raise RuntimeError(f"Не удалось получить ответ модели: {last_error}") from last_error
        raise RuntimeError("Модель не вернула ответ.")
    finally:
        http_client.close()


__all__ = ["generate", "DEFAULT_MODEL"]
