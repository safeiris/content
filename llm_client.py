# -*- coding: utf-8 -*-
"""Simple wrapper around chat completion providers with retries and sane defaults."""
from __future__ import annotations

import json
import logging
import os
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
RESPONSES_API_URL = "https://api.openai.com/v1/responses"
RESPONSES_ALLOWED_KEYS = ("model", "input", "text", "max_output_tokens")
RESPONSES_POLL_DELAY = 0.25
GPT5_TEXT_ONLY_SUFFIX = "Ответь обычным текстом, без tool_calls и без структурированных форматов."

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
RESPONSES_ERROR_PATH = Path("artifacts/debug/last_gpt5_responses_error.json")
RESPONSES_REQUEST_PATH = Path("artifacts/debug/last_gpt5_responses_request.json")


@dataclass(frozen=True)
class GenerationResult:
    """Container describing the outcome of a text generation call."""

    text: str
    model_used: str
    retry_used: bool
    fallback_used: Optional[str]
    fallback_reason: Optional[str] = None
    api_route: str = "chat"
    schema: str = "none"


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


def build_responses_payload(
    model: str,
    system_text: Optional[str],
    user_text: Optional[str],
    max_tokens: int,
) -> Dict[str, object]:
    """Construct a minimal Responses API payload for GPT-5 models."""

    sections: List[str] = []

    system_block = (system_text or "").strip()
    if system_block:
        sections.append(f"SYSTEM\n{system_block}")

    user_block = (user_text or "").strip()
    if user_block:
        sections.append(f"USER\n{user_block}")

    joined_input = "\n\n".join(section for section in sections if section)

    payload: Dict[str, object] = {
        "model": str(model).strip(),
        "input": joined_input.strip(),
        "max_output_tokens": int(max_tokens),
    }
    return payload


def sanitize_payload_for_responses(payload: Dict[str, object]) -> Tuple[Dict[str, object], int]:
    """Restrict Responses payload to the documented whitelist and types."""

    sanitized: Dict[str, object] = {}
    for key in RESPONSES_ALLOWED_KEYS:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            if key == "model":
                sanitized[key] = trimmed
                continue
            if key == "input":
                sanitized[key] = trimmed
                continue
        if key == "input" and not isinstance(value, str):
            if isinstance(value, (list, dict)):
                converted = json.dumps(value, ensure_ascii=False)
            else:
                converted = str(value)
            converted = converted.strip()
            if converted:
                sanitized[key] = converted
            continue
        if key == "max_output_tokens":
            try:
                sanitized[key] = int(value)
            except (TypeError, ValueError):
                continue
            continue
        if key == "text":
            if isinstance(value, dict):
                text_block = {}
                format_section = value.get("format")
                if isinstance(format_section, dict):
                    type_value = format_section.get("type")
                    if type_value:
                        text_block["format"] = {"type": str(type_value).strip()}
                if text_block:
                    sanitized[key] = text_block
            continue
    input_value = sanitized.get("input", "")
    input_length = len(input_value) if isinstance(input_value, str) else 0
    return sanitized, input_length


def _store_responses_debug_artifacts(
    error_payload: Dict[str, object],
    request_payload: Dict[str, object],
) -> None:
    """Persist diagnostic artifacts for Responses API failures."""

    try:
        RESPONSES_ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESPONSES_ERROR_PATH.write_text(
            json.dumps(error_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        snapshot = dict(request_payload)
        input_value = snapshot.pop("input", "")
        if isinstance(input_value, str):
            preview = input_value[:200]
        else:
            preview = str(input_value)[:200]
        snapshot["input_preview"] = preview
        RESPONSES_REQUEST_PATH.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        LOGGER.debug("failed to persist Responses debug artifacts: %s", exc)


def _handle_responses_http_error(
    error: httpx.HTTPStatusError,
    payload_snapshot: Dict[str, object],
) -> None:
    """Log a concise message and persist diagnostics for Responses failures."""

    response = error.response
    status = response.status_code if response is not None else "unknown"
    error_payload: Dict[str, object] = {}
    error_type = ""
    message = ""
    if response is not None:
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            error_payload = parsed
            error_block = parsed.get("error")
            if isinstance(error_block, dict):
                error_type = str(error_block.get("type", ""))
                message = str(error_block.get("message", ""))
        if not message:
            message = response.text.strip()
    truncated = (message or "")[:200]
    LOGGER.error(
        'Responses API error: status=%s type=%s message="%s"',
        status,
        error_type or "unknown",
        truncated,
    )
    if not error_payload and response is not None:
        # Ensure we persist at least the textual payload
        error_payload = {
            "status": status,
            "error": {
                "type": error_type or "unknown",
                "message": message,
            },
        }
    _store_responses_debug_artifacts(error_payload, payload_snapshot)


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


def _categorize_schema(parse_flags: Dict[str, object]) -> str:
    if parse_flags.get("parts"):
        return "parts"
    if (
        parse_flags.get("content_str")
        or parse_flags.get("choices_text")
        or parse_flags.get("output_text")
    ):
        return "text"
    return "none"


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


def _extract_responses_text(data: Dict[str, object]) -> Tuple[str, Dict[str, object], str]:
    parse_flags: Dict[str, object] = {
        "content_str": 0,
        "parts": 0,
        "content_dict": 0,
        "choices_text": 0,
        "output_text": 0,
    }

    resp_keys = sorted(str(key) for key in data.keys())
    parse_flags["resp_keys"] = resp_keys

    def _collect_segments(container: object) -> List[str]:
        segments: List[str] = []
        if isinstance(container, list):
            for item in container:
                segments.extend(_collect_segments(item))
        elif isinstance(container, dict):
            content_value = container.get("content")
            if isinstance(content_value, list):
                for part in content_value:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type", ""))
                    if part_type not in {"output_text", "text"}:
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        stripped = text_value.strip()
                        if stripped:
                            segments.append(stripped)
            for key in ("output", "outputs", "response", "message"):
                nested_value = container.get(key)
                if nested_value is not None:
                    segments.extend(_collect_segments(nested_value))
        return segments

    segments: List[str] = []
    root_used: Optional[str] = None

    for root_key in ("output", "outputs"):
        root_value = data.get(root_key)
        if root_value is None:
            continue
        candidate_segments = _collect_segments(root_value)
        if candidate_segments:
            segments.extend(candidate_segments)
            root_used = root_key
    if not segments:
        for root_key in ("response", "message"):
            root_value = data.get(root_key)
            if root_value is None:
                continue
            candidate_segments = _collect_segments(root_value)
            if candidate_segments:
                segments.extend(candidate_segments)
                root_used = root_key
                break

    schema_label = "responses.output_text" if segments else "responses.none"
    parse_flags["schema"] = schema_label
    parse_flags["segments"] = len(segments)
    if segments:
        parse_flags["output_text"] = 1
        parse_flags["parts"] = 1

    text = "\n\n".join(segments) if segments else ""
    text_len = len(text)
    LOGGER.info(
        "responses parse resp_keys=%s root=%s segments=%d len=%d schema=%s",
        resp_keys,
        root_used,
        parse_flags.get("segments", 0),
        text_len,
        schema_label,
    )
    return text, parse_flags, schema_label


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


def _extract_unknown_parameter_name(response: httpx.Response) -> Optional[str]:
    message: str = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error_block = payload.get("error")
        if isinstance(error_block, dict):
            message = str(error_block.get("message", ""))
    if not message:
        message = response.text or ""
    message = message.strip()
    if not message:
        return None
    lowered = message.lower()
    if "unknown parameter" not in lowered and "unsupported parameter" not in lowered:
        return None
    marker = ":"
    if marker in message:
        remainder = message.split(marker, 1)[1].strip()
    else:
        remainder = message
    if remainder.startswith("'") and "'" in remainder[1:]:
        return remainder.split("'", 2)[1].strip()
    return remainder.split()[0].strip("'\"") or None


def _make_request(
    http_client: httpx.Client,
    *,
    api_url: str,
    headers: Dict[str, str],
    payload: Dict[str, object],
    schedule: List[float],
) -> Tuple[Dict[str, object], bool]:
    last_error: Optional[BaseException] = None
    shimmed_param = False
    stripped_param: Optional[str] = None
    current_payload: Dict[str, object] = dict(payload)
    attempt_index = 0
    while attempt_index < MAX_RETRIES:
        attempt_index += 1
        try:
            response = http_client.post(api_url, headers=headers, json=current_payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data, shimmed_param
            raise RuntimeError("Модель вернула неожиданный формат ответа.")
        except EmptyCompletionError:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if (
                status == 400
                and not shimmed_param
                and exc.response is not None
            ):
                param_name = _extract_unknown_parameter_name(exc.response)
                if param_name:
                    if param_name in current_payload:
                        current_payload = dict(current_payload)
                        current_payload.pop(param_name, None)
                    shimmed_param = True
                    stripped_param = param_name
                    LOGGER.warning(
                        "retry=shim_unknown_param: stripped '%s' from payload",
                        param_name,
                    )
                    continue
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):  # pragma: no cover - respect interrupts
                raise
            last_error = exc
        if attempt_index >= MAX_RETRIES or not _should_retry(last_error):
            break
        sleep_for = schedule[min(attempt_index - 1, len(schedule) - 1)]
        reason = _describe_error(last_error)
        print(
            f"[llm_client] retry #{attempt_index} reason: {reason}; sleeping {sleep_for}s",
            file=sys.stderr,
        )
        time.sleep(sleep_for)
    if stripped_param and last_error and isinstance(last_error, httpx.HTTPStatusError):
        LOGGER.debug(
            "param shim exhausted for '%s' with HTTP %s", stripped_param, last_error.response.status_code
        )
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

    def _augment_gpt5_messages(original: List[Dict[str, object]]) -> List[Dict[str, object]]:
        augmented: List[Dict[str, object]] = []
        appended_suffix = False
        for message in original:
            cloned = dict(message)
            if not appended_suffix and cloned.get("role") == "system":
                content = str(cloned.get("content", ""))
                if GPT5_TEXT_ONLY_SUFFIX not in content:
                    content = f"{content.rstrip()}\n\n{GPT5_TEXT_ONLY_SUFFIX}".strip()
                cloned["content"] = content
                appended_suffix = True
            augmented.append(cloned)
        return augmented

    gpt5_messages_cache: Optional[List[Dict[str, object]]] = None

    def _messages_for_model(target_model: str) -> List[Dict[str, object]]:
        nonlocal gpt5_messages_cache
        if target_model.lower().startswith("gpt-5"):
            if gpt5_messages_cache is None:
                gpt5_messages_cache = _augment_gpt5_messages(messages)
            return gpt5_messages_cache
        return messages

    def _prepare_chat_payload(target_model: str) -> Dict[str, object]:
        payload_messages = _messages_for_model(target_model)
        payload: Dict[str, object] = {
            "model": target_model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "tools" not in payload:
            LOGGER.info("chat payload: tool_choice omitted (no tools)")
        LOGGER.info("openai payload blueprint: %s", _summarize_payload(payload))
        return payload

    def _call_chat_model(target_model: str) -> Tuple[str, Dict[str, object], Dict[str, object], str]:
        nonlocal retry_used
        payload = _prepare_chat_payload(target_model)
        LOGGER.info("dispatch route=chat model=%s", target_model)
        data, shimmed = _make_request(
            http_client,
            api_url=api_url,
            headers=headers,
            payload=payload,
            schedule=schedule,
        )
        if shimmed:
            retry_used = True
        text, parse_flags, schema_label = _extract_response_text(data)
        if not text:
            LOGGER.warning("Пустой ответ модели %s (schema=%s)", target_model, schema_label)
            raise EmptyCompletionError(
                "Модель вернула пустой ответ",
                raw_response=data,
                parse_flags=parse_flags,
            )
        _persist_raw_response(data)
        return text, parse_flags, data, schema_label

    def _call_responses_model(target_model: str) -> Tuple[str, Dict[str, object], Dict[str, object], str]:
        nonlocal retry_used

        payload_messages = _messages_for_model(target_model)
        system_segments: List[str] = []
        user_segments: List[str] = []
        for item in payload_messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_segments.append(content)
            elif role == "user":
                user_segments.append(content)
            else:
                user_segments.append(f"{role.upper()}:\n{content}")

        system_text = "\n\n".join(system_segments)
        user_text = "\n\n".join(user_segments)

        payload = build_responses_payload(target_model, system_text, user_text, max_tokens)
        sanitized_payload, _ = sanitize_payload_for_responses(payload)

        LOGGER.info("dispatch route=responses model=%s", target_model)

        def _log_payload_state(payload_snapshot: Dict[str, object]) -> None:
            keys = sorted(payload_snapshot.keys())
            LOGGER.info("payload keys: %s", keys)
            input_candidate = payload_snapshot.get("input", "")
            length = len(input_candidate) if isinstance(input_candidate, str) else 0
            LOGGER.info("responses input_len=%d", length)

        current_payload = dict(sanitized_payload)
        base_payload = dict(sanitized_payload)
        attempt_index = 0
        shimmed_param: Optional[str] = None
        last_error: Optional[BaseException] = None
        hint_retry_used = False
        text_format_retry_used = False

        while attempt_index < MAX_RETRIES:
            attempt_index += 1
            if attempt_index > 1:
                retry_used = True
            _log_payload_state(current_payload)
            try:
                response = http_client.post(
                    RESPONSES_API_URL,
                    headers=headers,
                    json=current_payload,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Модель вернула неожиданный формат ответа.")
                status_value = data.get("status")
                if isinstance(status_value, str) and status_value and status_value.lower() != "completed":
                    response_id = data.get("id")
                    if isinstance(response_id, str) and response_id:
                        LOGGER.info(
                            "responses status=%s; polling id=%s",
                            status_value,
                            response_id,
                        )
                        time.sleep(RESPONSES_POLL_DELAY)
                        try:
                            poll_response = http_client.get(
                                f"{RESPONSES_API_URL}/{response_id}",
                                headers=headers,
                            )
                            poll_response.raise_for_status()
                            polled_data = poll_response.json()
                            if isinstance(polled_data, dict):
                                data = polled_data
                        except Exception as poll_error:  # noqa: BLE001
                            LOGGER.warning(
                                "responses poll failed id=%s error=%s",
                                response_id,
                                poll_error,
                            )
                text, parse_flags, schema_label = _extract_responses_text(data)
                if not text:
                    LOGGER.warning(
                        "Пустой ответ от Responses API %s (schema=%s)",
                        target_model,
                        schema_label,
                    )
                    raise EmptyCompletionError(
                        "Responses API вернул пустой ответ",
                        raw_response=data,
                        parse_flags=parse_flags,
                    )
                _persist_raw_response(data)
                return text, parse_flags, data, schema_label
            except EmptyCompletionError:
                raise
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                error_message = ""
                if exc.response is not None:
                    try:
                        payload_json = exc.response.json()
                    except ValueError:
                        payload_json = None
                    if isinstance(payload_json, dict):
                        error_block = payload_json.get("error")
                        if isinstance(error_block, dict):
                            error_message = str(error_block.get("message", ""))
                    if not error_message:
                        error_message = exc.response.text or ""
                _handle_responses_http_error(exc, current_payload)
                lowered_message = error_message.lower()
                if status_code == 400:
                    if shimmed_param is None:
                        param_name = _extract_unknown_parameter_name(exc.response)
                        if param_name:
                            next_payload = dict(current_payload)
                            if param_name in next_payload:
                                next_payload.pop(param_name, None)
                            current_payload, _ = sanitize_payload_for_responses(next_payload)
                            shimmed_param = param_name
                            retry_used = True
                            LOGGER.warning(
                                "retry=shim_unknown_param stripped='%s'",
                                param_name,
                            )
                            continue
                    if (
                        not hint_retry_used
                        and (
                            "response_format" in lowered_message
                            or "text.format" in lowered_message
                        )
                    ):
                        current_payload = dict(base_payload)
                        hint_retry_used = True
                        LOGGER.warning("retry=shim_hint_response_format")
                        continue
                    if (
                        not text_format_retry_used
                        and (
                            "must specify text.format" in lowered_message
                            or "expected a text format" in lowered_message
                            or "specify text.format" in lowered_message
                        )
                    ):
                        next_payload = dict(base_payload)
                        next_payload["text"] = {"format": {"type": "plain"}}
                        current_payload, _ = sanitize_payload_for_responses(next_payload)
                        text_format_retry_used = True
                        LOGGER.warning("retry=shim_text_format_object")
                        continue
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, KeyboardInterrupt):  # pragma: no cover - respect interrupts
                    raise
                last_error = exc
            if attempt_index >= MAX_RETRIES or not _should_retry(last_error):
                break
            sleep_for = schedule[min(attempt_index - 1, len(schedule) - 1)]
            reason = _describe_error(last_error)
            print(
                f"[llm_client] retry #{attempt_index} reason: {reason}; sleeping {sleep_for}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)

        if last_error:
            if isinstance(last_error, httpx.HTTPStatusError):
                _raise_for_last_error(last_error)
            if isinstance(last_error, (httpx.TimeoutException, httpx.TransportError)):
                _raise_for_last_error(last_error)
            raise last_error

        raise RuntimeError("Модель не вернула ответ.")

    lower_model = model_name.lower()
    is_gpt5_model = lower_model.startswith("gpt-5")
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
                    text, parse_flags_fallback, _, schema_fallback = _call_chat_model(
                        fallback_used
                    )
                    schema_category = _categorize_schema(parse_flags_fallback)
                    LOGGER.info(
                        "completion schema category=%s (schema=%s, route=chat)",
                        schema_category,
                        schema_fallback,
                    )
                    return GenerationResult(
                        text=text,
                        model_used=fallback_used,
                        retry_used=False,
                        fallback_used=fallback_used,
                        fallback_reason=fallback_reason,
                        api_route="chat",
                        schema=schema_category,
                    )
                except EmptyCompletionError as fallback_error:
                    _persist_raw_response(fallback_error.raw_response)
                    _log_parse_chain(fallback_error.parse_flags, retry=0, fallback=fallback_used)
                    raise

            try:
                text, parse_flags_initial, _, schema_initial_call = _call_responses_model(
                    model_name
                )
                schema_category = schema_initial_call
                LOGGER.info(
                    "completion schema category=%s (schema=%s, route=responses)",
                    schema_category,
                    schema_initial_call,
                )
                return GenerationResult(
                    text=text,
                    model_used=model_name,
                    retry_used=retry_used,
                    fallback_used=None,
                    fallback_reason=None,
                    api_route="responses",
                    schema=schema_category,
                )
            except EmptyCompletionError as responses_empty:
                _persist_raw_response(responses_empty.raw_response)
                _log_parse_chain(responses_empty.parse_flags, retry=0, fallback="responses")
                LOGGER.warning(
                    "empty completion from Responses API %s (schema=%s)",
                    model_name,
                    responses_empty.parse_flags.get("schema", "unknown"),
                )
                if FORCE_MODEL:
                    raise RuntimeError(
                        "Responses API вернул пустой ответ, а fallback отключён (FORCE_MODEL)"
                    ) from responses_empty
                fallback_reason = "empty_completion_gpt5_responses"
            except Exception as responses_error:  # noqa: BLE001
                LOGGER.warning("Responses API call failed: %s", responses_error)
                if FORCE_MODEL:
                    raise RuntimeError(
                        "Responses API вернул ошибку, а fallback отключён (FORCE_MODEL)"
                    ) from responses_error
                fallback_reason = "api_error_gpt5_responses"
            fallback_used = FALLBACK_MODEL
            LOGGER.warning(
                "switching to fallback model %s (primary=%s, reason=%s)",
                fallback_used,
                model_name,
                fallback_reason,
            )
            text, parse_flags_fallback, _, schema_fallback = _call_chat_model(fallback_used)
            schema_category = _categorize_schema(parse_flags_fallback)
            LOGGER.info(
                "fallback completion schema category=%s (schema=%s, route=chat)",
                schema_category,
                schema_fallback,
            )
            return GenerationResult(
                text=text,
                model_used=fallback_used,
                retry_used=retry_used,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                api_route="chat",
                schema=schema_category,
            )

        text, parse_flags_initial, _, schema_initial_call = _call_chat_model(model_name)
        schema_category = _categorize_schema(parse_flags_initial)
        LOGGER.info(
            "completion schema category=%s (schema=%s, route=chat)",
            schema_category,
            schema_initial_call,
        )
        return GenerationResult(
            text=text,
            model_used=model_name,
            retry_used=False,
            fallback_used=None,
            fallback_reason=None,
            api_route="chat",
            schema=schema_category,
        )
    finally:
        http_client.close()


__all__ = ["generate", "DEFAULT_MODEL", "GenerationResult"]
