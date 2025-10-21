# -*- coding: utf-8 -*-
"""Simple wrapper around chat completion providers with retries and sane defaults."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from config import FORCE_MODEL, OPENAI_API_KEY


DEFAULT_MODEL = "gpt-5"
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [0.5, 1.0, 2.0]
FALLBACK_MODEL = "gpt-4o"

MODEL_PROVIDER_MAP = {
    "gpt-5": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
}

PROVIDER_API_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
}


LOGGER = logging.getLogger(__name__)
RAW_RESPONSE_PATH = Path("artifacts/debug/last_raw_response.json")


@dataclass(frozen=True)
class GenerationResult:
    """Container describing the outcome of a text generation call."""

    text: str
    model_used: str
    retry_used: bool
    fallback_used: Optional[str]
    fallback_reason: Optional[str] = None


class EmptyCompletionError(RuntimeError):
    """Raised when the model responds without any textual content."""

    status_code = 502

    def __init__(
        self,
        message: str,
        *,
        raw_response: Optional[Dict[str, object]] = None,
        parse_flags: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response or {}
        self.parse_flags = parse_flags or {}


def _collect_text_parts(parts: List[object]) -> str:
    collected: List[str] = []
    for part in parts:
        candidate: Optional[str] = None
        if isinstance(part, dict):
            text_value = part.get("text")
            if isinstance(text_value, str):
                candidate = text_value
            else:
                content_value = part.get("content")
                if isinstance(content_value, str):
                    candidate = content_value
                elif isinstance(content_value, list):
                    nested = _collect_text_parts(content_value)
                    if nested:
                        candidate = nested
                value_value = part.get("value")
                if not candidate and isinstance(value_value, str):
                    candidate = value_value
                if not candidate and part.get("type") == "text":
                    alt_value = part.get("content") or part.get("value")
                    if isinstance(alt_value, str):
                        candidate = alt_value
        elif isinstance(part, str):
            candidate = part
        if candidate:
            stripped = candidate.strip()
            if stripped:
                collected.append(stripped)
    return "\n\n".join(collected)


def _describe_type(value: object) -> str:
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value.keys())
        if not keys:
            return "dict(empty)"
        preview = ",".join(keys[:4])
        if len(keys) > 4:
            preview += ",…"
        return f"dict(keys={preview})"
    if value is None:
        return "none"
    return value.__class__.__name__


def _extract_choice_content(choice: Dict[str, object]) -> Tuple[str, Dict[str, object], str]:
    """Return textual content from a completion choice along with parse statistics."""

    parse_flags: Dict[str, object] = {
        "content_str": 0,
        "parts": 0,
        "content_dict": 0,
        "choices_text": 0,
        "output_text": 0,
    }
    schema_label = "choice:unknown"

    def _extract_from_dict(container: Dict[str, object]) -> Tuple[str, bool]:
        text_value = container.get("text")
        if isinstance(text_value, str):
            stripped = text_value.strip()
            if stripped:
                return stripped, False
        for key in ("content", "value"):
            alt_value = container.get(key)
            if isinstance(alt_value, str):
                stripped = alt_value.strip()
                if stripped:
                    return stripped, False
            elif isinstance(alt_value, list):
                joined = _collect_text_parts(alt_value)
                if joined:
                    return joined, True
            elif isinstance(alt_value, dict):
                nested, nested_parts = _extract_from_dict(alt_value)
                if nested:
                    return nested, nested_parts
        return "", False

    # Primary schema – chat message
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        schema_label = f"message.content:{_describe_type(content)}"
        if isinstance(content, str):
            stripped = content.strip()
            if stripped:
                parse_flags["content_str"] = 1
                parse_flags["schema"] = schema_label
                return stripped, parse_flags, schema_label
        elif isinstance(content, list):
            joined = _collect_text_parts(content)
            if joined:
                parse_flags["parts"] = 1
                parse_flags["schema"] = schema_label
                return joined, parse_flags, schema_label
        elif isinstance(content, dict):
            extracted, used_parts = _extract_from_dict(content)
            if extracted:
                parse_flags["content_dict"] = 1
                if used_parts:
                    parse_flags["parts"] = 1
                parse_flags["schema"] = schema_label
                return extracted, parse_flags, schema_label
        alt_text = message.get("text")
        schema_label = f"message.text:{_describe_type(alt_text)}"
        if isinstance(alt_text, str):
            stripped = alt_text.strip()
            if stripped:
                parse_flags["content_str"] = 1
                parse_flags["schema"] = schema_label
                return stripped, parse_flags, schema_label
        elif isinstance(alt_text, list):
            joined = _collect_text_parts(alt_text)
            if joined:
                parse_flags["parts"] = 1
                parse_flags["schema"] = schema_label
                return joined, parse_flags, schema_label
        elif isinstance(alt_text, dict):
            extracted, used_parts = _extract_from_dict(alt_text)
            if extracted:
                parse_flags["content_dict"] = 1
                if used_parts:
                    parse_flags["parts"] = 1
                parse_flags["schema"] = schema_label
                return extracted, parse_flags, schema_label

    # Alternate schemas – top-level text fields
    candidate = choice.get("text") if isinstance(choice, dict) else None
    schema_label = f"choice.text:{_describe_type(candidate)}"
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if stripped:
            parse_flags["choices_text"] = 1
            parse_flags["schema"] = schema_label
            return stripped, parse_flags, schema_label
    elif isinstance(candidate, list):
        joined = _collect_text_parts(candidate)
        if joined:
            parse_flags["choices_text"] = 1
            parse_flags["parts"] = 1
            parse_flags["schema"] = schema_label
            return joined, parse_flags, schema_label
    elif isinstance(candidate, dict):
        extracted, used_parts = _extract_from_dict(candidate)
        if extracted:
            parse_flags["choices_text"] = 1
            parse_flags["content_dict"] = 1
            if used_parts:
                parse_flags["parts"] = 1
            parse_flags["schema"] = schema_label
            return extracted, parse_flags, schema_label

    output_candidate = choice.get("output_text") if isinstance(choice, dict) else None
    schema_label = f"choice.output_text:{_describe_type(output_candidate)}"
    if isinstance(output_candidate, str):
        stripped = output_candidate.strip()
        if stripped:
            parse_flags["output_text"] = 1
            parse_flags["schema"] = schema_label
            return stripped, parse_flags, schema_label
    elif isinstance(output_candidate, list):
        joined = _collect_text_parts(output_candidate)
        if joined:
            parse_flags["output_text"] = 1
            parse_flags["parts"] = 1
            parse_flags["schema"] = schema_label
            return joined, parse_flags, schema_label
    elif isinstance(output_candidate, dict):
        extracted, used_parts = _extract_from_dict(output_candidate)
        if extracted:
            parse_flags["output_text"] = 1
            parse_flags["content_dict"] = 1
            if used_parts:
                parse_flags["parts"] = 1
            parse_flags["schema"] = schema_label
            return extracted, parse_flags, schema_label

    exotic_content = choice.get("content") if isinstance(choice, dict) else None
    schema_label = f"choice.content:{_describe_type(exotic_content)}"
    if isinstance(exotic_content, list):
        joined = _collect_text_parts(exotic_content)
        if joined:
            parse_flags["output_text"] = 1
            parse_flags["parts"] = 1
            parse_flags["schema"] = schema_label
            return joined, parse_flags, schema_label
    elif isinstance(exotic_content, dict):
        extracted, used_parts = _extract_from_dict(exotic_content)
        if extracted:
            parse_flags["output_text"] = 1
            parse_flags["content_dict"] = 1
            if used_parts:
                parse_flags["parts"] = 1
            parse_flags["schema"] = schema_label
            return extracted, parse_flags, schema_label

    parse_flags["schema"] = schema_label
    return "", parse_flags, schema_label


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


def _check_model_availability(
    http_client: httpx.Client,
    *,
    provider: str,
    headers: Dict[str, str],
    model_name: str,
) -> bool:
    """Return True if the requested model looks available for the current credentials."""

    if provider != "openai":
        return True

    url = f"https://api.openai.com/v1/models/{model_name}"
    try:
        response = http_client.get(url, headers=headers)
    except httpx.HTTPError as exc:  # pragma: no cover - network dependent
        LOGGER.warning("model availability probe failed: %s", exc)
        return True

    if response.status_code == 200:
        return True

    if response.status_code in {401, 403, 404}:
        LOGGER.error(
            "model %s is not available for current credentials (HTTP %s)",
            model_name,
            response.status_code,
        )
        return False

    if response.status_code >= 500:
        LOGGER.warning(
            "model availability endpoint temporary failure (HTTP %s) — proceeding",
            response.status_code,
        )
        return True

    LOGGER.warning(
        "unexpected status from model availability endpoint for %s: HTTP %s",
        model_name,
        response.status_code,
    )
    return False


def _persist_raw_response(payload: Dict[str, object]) -> None:
    try:
        RAW_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_RESPONSE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - diagnostic helper
        LOGGER.debug("failed to persist raw response: %s", exc)


def _summarize_payload(payload: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for key, value in payload.items():
        if key == "messages":
            if isinstance(value, list):
                summary[key] = f"<{len(value)} messages>"
            else:
                summary[key] = "<messages>"
            continue
        summary[key] = value
    return summary


def _log_parse_chain(parse_flags: Dict[str, object], *, retry: int, fallback: str) -> None:
    LOGGER.warning(
        "parse_chain: content_str=%d; parts=%d; content_dict=%d; choices_text=%d; output_text=%d; retry=%d; fallback=%s; schema=%s",
        parse_flags.get("content_str", 0),
        parse_flags.get("parts", 0),
        parse_flags.get("content_dict", 0),
        parse_flags.get("choices_text", 0),
        parse_flags.get("output_text", 0),
        retry,
        fallback,
        parse_flags.get("schema", "unknown"),
    )


def _should_retry_empty_gpt5(parse_flags: Dict[str, object]) -> bool:
    schema = str(parse_flags.get("schema", ""))
    if not schema:
        return True
    for marker in (":none", "len=0", "dict(empty)"):
        if marker in schema:
            return False
    return True


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


def _raise_for_last_error(last_error: BaseException) -> None:
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


def _make_request(
    http_client: httpx.Client,
    *,
    api_url: str,
    headers: Dict[str, str],
    payload: Dict[str, object],
    schedule: List[float],
) -> Dict[str, object]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = http_client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            raise RuntimeError("Модель вернула неожиданный формат ответа.")
        except EmptyCompletionError:
            raise
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):  # pragma: no cover - respect interrupts
                raise
            last_error = exc
            if attempt >= MAX_RETRIES or not _should_retry(exc):
                break
            sleep_for = schedule[min(attempt - 1, len(schedule) - 1)]
            reason = _describe_error(exc)
            print(
                f"[llm_client] retry #{attempt} reason: {reason}; sleeping {sleep_for}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    if last_error:
        _raise_for_last_error(last_error)
    raise RuntimeError("Модель не вернула ответ.")


def _extract_response_text(data: Dict[str, object]) -> Tuple[str, Dict[str, object], str]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Модель не вернула варианты ответа.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("Модель вернула неожиданный формат ответа.")
    return _extract_choice_content(choice)


def generate(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1400,
    timeout_s: int = 60,
    backoff_schedule: Optional[List[float]] = None,
    ) -> GenerationResult:
    """Call the configured LLM and return a structured generation result."""

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

    schedule = _resolve_backoff_schedule(backoff_schedule)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _prepare_payload(target_model: str) -> Dict[str, object]:
        lower = target_model.lower()
        payload: Dict[str, object] = {
            "model": target_model,
            "messages": messages,
        }
        if "gpt-5" in lower:
            payload["max_completion_tokens"] = max_tokens
            payload["response_format"] = {"type": "text"}
            payload["modalities"] = ["text"]
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        LOGGER.info("openai payload blueprint: %s", _summarize_payload(payload))
        return payload

    def _call_model(target_model: str) -> Tuple[str, Dict[str, object], Dict[str, object], str]:
        lower_target = target_model.lower()
        payload = _prepare_payload(target_model)
        if "gpt-5" in lower_target:
            LOGGER.info("dispatching request to GPT-5 model %s", target_model)
        data = _make_request(
            http_client,
            api_url=api_url,
            headers=headers,
            payload=payload,
            schedule=schedule,
        )
        text, parse_flags, schema_label = _extract_response_text(data)
        if "gpt-5" in lower_target:
            LOGGER.info("gpt-5 response schema=%s", schema_label)
        if not text:
            LOGGER.warning("Пустой ответ модели %s (schema=%s)", target_model, schema_label)
            raise EmptyCompletionError(
                "Модель вернула пустой ответ",
                raw_response=data,
                parse_flags=parse_flags,
            )
        return text, parse_flags, data, schema_label

    lower_model = model_name.lower()
    is_gpt5_model = "gpt-5" in lower_model
    if is_gpt5_model:
        LOGGER.info("temperature is ignored for GPT-5; using default")

    retry_used = False
    fallback_used: Optional[str] = None
    fallback_reason: Optional[str] = None

    try:
        if is_gpt5_model:
            available = _check_model_availability(
                http_client,
                provider=provider,
                headers=headers,
                model_name=model_name,
            )
            if not available:
                error_message = "Model GPT-5 not available for this key/plan"
                LOGGER.warning("primary model %s unavailable — considering fallback", model_name)
                if FORCE_MODEL:
                    raise RuntimeError(error_message)
                fallback_used = FALLBACK_MODEL
                fallback_reason = "model_unavailable"
                LOGGER.warning(
                    "switching to fallback model %s (primary=%s, reason=%s: %s)",
                    fallback_used,
                    model_name,
                    fallback_reason,
                    error_message,
                )
                try:
                    text, _, _, _ = _call_model(fallback_used)
                    return GenerationResult(
                        text=text,
                        model_used=fallback_used,
                        retry_used=False,
                        fallback_used=fallback_used,
                        fallback_reason=fallback_reason,
                    )
                except EmptyCompletionError as fallback_error:
                    _persist_raw_response(fallback_error.raw_response)
                    _log_parse_chain(fallback_error.parse_flags, retry=0, fallback=fallback_used)
                    raise

        text, _, _, _ = _call_model(model_name)
        return GenerationResult(
            text=text,
            model_used=model_name,
            retry_used=False,
            fallback_used=None,
            fallback_reason=None,
        )
    except EmptyCompletionError as initial_error:
        _persist_raw_response(initial_error.raw_response)
        _log_parse_chain(initial_error.parse_flags, retry=0, fallback="none")
        schema_initial = str(initial_error.parse_flags.get("schema", "unknown"))
        LOGGER.warning(
            "empty completion from %s (schema=%s, attempt=initial)",
            model_name,
            schema_initial,
        )
        if not is_gpt5_model:
            raise

        retry_used = False
        error_for_fallback = initial_error
        if _should_retry_empty_gpt5(initial_error.parse_flags):
            jitter = random.uniform(0.2, 0.4)
            time.sleep(jitter)
            retry_used = True
            try:
                text, _, _, schema_retry = _call_model(model_name)
                LOGGER.info(
                    "gpt-5 retry succeeded after empty completion (schema=%s)",
                    schema_retry,
                )
                return GenerationResult(
                    text=text,
                    model_used=model_name,
                    retry_used=True,
                    fallback_used=None,
                    fallback_reason=None,
                )
            except EmptyCompletionError as retry_error:
                _persist_raw_response(retry_error.raw_response)
                _log_parse_chain(retry_error.parse_flags, retry=1, fallback="gpt-4o")
                LOGGER.warning(
                    "empty completion from %s (schema=%s, attempt=retry)",
                    model_name,
                    retry_error.parse_flags.get("schema", "unknown"),
                )
                error_for_fallback = retry_error
        else:
            LOGGER.info(
                "Skipping GPT-5 retry because schema indicates no textual payload (schema=%s)",
                schema_initial,
            )
            _log_parse_chain(initial_error.parse_flags, retry=0, fallback="gpt-4o")

        fallback_used = FALLBACK_MODEL
        fallback_reason = "empty_completion"
        LOGGER.warning(
            "switching to fallback model %s (primary=%s, reason=%s, schema=%s, retry=%s)",
            fallback_used,
            model_name,
            fallback_reason,
            error_for_fallback.parse_flags.get("schema", "unknown"),
            "yes" if retry_used else "no",
        )
        try:
            text, _, _, _ = _call_model(fallback_used)
            return GenerationResult(
                text=text,
                model_used=fallback_used,
                retry_used=retry_used,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        except EmptyCompletionError as fallback_error:
            _persist_raw_response(fallback_error.raw_response)
            _log_parse_chain(
                fallback_error.parse_flags,
                retry=1 if retry_used else 0,
                fallback=fallback_used,
            )
            raise
    finally:
        http_client.close()


__all__ = ["generate", "DEFAULT_MODEL", "GenerationResult"]
