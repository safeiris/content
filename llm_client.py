# -*- coding: utf-8 -*-
"""Simple wrapper around chat completion providers with retries and sane defaults."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from config import (
    FORCE_MODEL,
    OPENAI_API_KEY,
    G5_ENABLE_PREVIOUS_ID_FETCH,
    G5_MAX_OUTPUT_TOKENS_BASE,
    G5_MAX_OUTPUT_TOKENS_MAX,
    G5_MAX_OUTPUT_TOKENS_STEP1,
    G5_MAX_OUTPUT_TOKENS_STEP2,
    G5_POLL_INTERVALS,
    G5_POLL_MAX_ATTEMPTS,
    G5_ESCALATION_LADDER,
    LLM_ALLOW_FALLBACK,
    LLM_MODEL,
    LLM_ROUTE,
)


DEFAULT_MODEL = LLM_MODEL
MAX_RETRIES = 2
BACKOFF_SCHEDULE = [0.75, 1.5]
RESPONSES_API_URL = "https://api.openai.com/v1/responses"
RESPONSES_ALLOWED_KEYS = (
    "model",
    "input",
    "max_output_tokens",
    "text",
    "previous_response_id",
)
RESPONSES_POLL_SCHEDULE = G5_POLL_INTERVALS
RESPONSES_MAX_ESCALATIONS = 2
MAX_RESPONSES_POLL_ATTEMPTS = (
    G5_POLL_MAX_ATTEMPTS if G5_POLL_MAX_ATTEMPTS > 0 else len(RESPONSES_POLL_SCHEDULE)
)
if MAX_RESPONSES_POLL_ATTEMPTS <= 0:
    MAX_RESPONSES_POLL_ATTEMPTS = len(RESPONSES_POLL_SCHEDULE)
GPT5_TEXT_ONLY_SUFFIX = "Ответь обычным текстом, без tool_calls и без структурированных форматов."
LIVING_STYLE_INSTRUCTION = (
    "Стиль текста: живой, человечный, уверенный.\n"
    "Пиши так, как будто объясняешь это умному человеку, но без канцелярита.\n"
    "Избегай сухих определений, добавляй лёгкие переходы и короткие фразы.\n"
    "Разбивай длинные абзацы, вставляй мини-примеры и пояснения своими словами.\n"
    "Тон — дружелюбный, экспертный, без лишней официальности."
)
_PROMPT_CACHE: "OrderedDict[Tuple[Tuple[str, str], ...], List[Dict[str, str]]]" = OrderedDict()
_PROMPT_CACHE_LIMIT = 16

_HTTP_CLIENT_LIMITS = httpx.Limits(
    max_connections=16,
    max_keepalive_connections=16,
    keepalive_expiry=120.0,
)
_HTTP_CLIENTS: "OrderedDict[float, httpx.Client]" = OrderedDict()


RESPONSES_MAX_OUTPUT_TOKENS_MIN = 16
RESPONSES_MAX_OUTPUT_TOKENS_MAX = 256


def clamp_responses_max_output_tokens(value: object) -> int:
    """Clamp max_output_tokens to the supported Responses bounds."""

    try:
        numeric_value = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        numeric_value = RESPONSES_MAX_OUTPUT_TOKENS_MIN
    return max(
        RESPONSES_MAX_OUTPUT_TOKENS_MIN,
        min(numeric_value, RESPONSES_MAX_OUTPUT_TOKENS_MAX),
    )


def reset_http_client_cache() -> None:
    """Close and clear pooled HTTP clients.

    Intended for test code to avoid state leaking between invocations when
    mocked clients keep internal counters (e.g. DummyClient instances)."""

    while _HTTP_CLIENTS:
        _, pooled_client = _HTTP_CLIENTS.popitem(last=False)
        try:
            pooled_client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


_JSON_STYLE_GUARD: Set[str] = {"json_schema", "json_object"}


def _should_apply_living_style(format_type: str) -> bool:
    normalized = str(format_type or "").strip().lower()
    return not normalized or normalized not in _JSON_STYLE_GUARD


def _apply_living_style_instruction(system_text: str, *, format_type: str) -> str:
    instruction = LIVING_STYLE_INSTRUCTION.strip()
    if not instruction:
        return system_text
    if not _should_apply_living_style(format_type):
        return system_text

    normalized_text = system_text.replace("\r\n", "\n")
    if instruction in normalized_text:
        return system_text

    base = system_text.rstrip()
    if not base:
        return instruction
    return f"{base}\n\n{instruction}"


def _cache_augmented_messages(messages: List[Dict[str, object]]) -> List[Dict[str, object]]:
    key = tuple((str(item.get("role", "")), str(item.get("content", ""))) for item in messages)
    cached = _PROMPT_CACHE.get(key)
    if cached is not None:
        _PROMPT_CACHE.move_to_end(key)
        return [dict(message) for message in cached]
    augmented: List[Dict[str, object]] = []
    appended_suffix = False
    for message in messages:
        cloned = dict(message)
        if not appended_suffix and cloned.get("role") == "system":
            content = str(cloned.get("content", ""))
            if GPT5_TEXT_ONLY_SUFFIX not in content:
                content = f"{content.rstrip()}\n\n{GPT5_TEXT_ONLY_SUFFIX}".strip()
            cloned["content"] = content
            appended_suffix = True
        augmented.append(cloned)
    _PROMPT_CACHE[key] = augmented
    while len(_PROMPT_CACHE) > _PROMPT_CACHE_LIMIT:
        _PROMPT_CACHE.popitem(last=False)
    return [dict(message) for message in augmented]


def _acquire_http_client(timeout_value: float) -> httpx.Client:
    key = round(timeout_value, 1)
    client = _HTTP_CLIENTS.get(key)
    if client is not None:
        _HTTP_CLIENTS.move_to_end(key)
        return client

    timeout = httpx.Timeout(
        timeout=timeout_value,
        connect=min(20.0, timeout_value),
        read=timeout_value,
        write=timeout_value,
    )
    client = httpx.Client(
        timeout=timeout,
        limits=_HTTP_CLIENT_LIMITS,
        headers={"Connection": "keep-alive"},
        http2=True,
    )
    _HTTP_CLIENTS[key] = client
    while len(_HTTP_CLIENTS) > 4:
        _, old_client = _HTTP_CLIENTS.popitem(last=False)
        try:
            old_client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
    return client
def is_min_tokens_error(response: Optional[httpx.Response]) -> bool:
    """Detect the specific 400 error about max_output_tokens being too small."""

    if response is None:
        return False

    message = ""
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

    normalized = re.sub(r"\s+", " ", message).lower()
    if "max_output_tokens" not in normalized:
        return False
    return "expected" in normalized and ">=" in normalized and "16" in normalized

RESPONSES_FORMAT_DEFAULT_NAME = "seo_article_skeleton"


DEFAULT_RESPONSES_TEXT_FORMAT: Dict[str, object] = {
    "type": "json_schema",
    "name": RESPONSES_FORMAT_DEFAULT_NAME,
    "schema": {
        "type": "object",
        "properties": {
            "intro": {"type": "string"},
            "main": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
            },
            "faq": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "a": {"type": "string"},
                    },
                    "required": ["q", "a"],
                    "additionalProperties": False,
                },
                "minItems": 5,
                "maxItems": 5,
            },
            "conclusion": {"type": "string"},
        },
        "required": ["intro", "main", "faq", "conclusion"],
        "additionalProperties": False,
    },
    "strict": True,
}

FALLBACK_RESPONSES_PLAIN_OUTLINE_FORMAT: Dict[str, object] = {
    "type": "json_schema",
    "name": "seo_article_plain_outline",
    "schema": {
        "type": "object",
        "properties": {
            "plain": {"type": "string"},
            "outline": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 7,
            },
        },
        "required": ["plain"],
        "additionalProperties": False,
    },
    "strict": False,
}


class SchemaValidationError(ValueError):
    """Raised when the provided schema cannot be normalized."""


def _iter_schema_children(schema: Dict[str, Any], path: str) -> List[Tuple[str, Dict[str, Any]]]:
    children: List[Tuple[str, Dict[str, Any]]] = []
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, value in properties.items():
            if isinstance(value, dict):
                child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
                children.append((child_path, value))
    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, dict):
        for key, value in pattern_properties.items():
            if isinstance(value, dict):
                child_path = f"{path}.patternProperties[{key!r}]"
                children.append((child_path, value))
    items = schema.get("items")
    if isinstance(items, dict):
        children.append((f"{path}['items']", items))
    elif isinstance(items, list):
        for index, value in enumerate(items):
            if isinstance(value, dict):
                children.append((f"{path}['items'][{index}]", value))
    for keyword in ("allOf", "anyOf", "oneOf"):
        collection = schema.get(keyword)
        if isinstance(collection, list):
            for index, value in enumerate(collection):
                if isinstance(value, dict):
                    children.append((f"{path}.{keyword}[{index}]", value))
    for keyword in ("$defs", "definitions"):
        collection = schema.get(keyword)
        if isinstance(collection, dict):
            for key, value in collection.items():
                if isinstance(value, dict):
                    child_path = f"{path}.{keyword}[{key!r}]"
                    children.append((child_path, value))
    return children


def _normalize_json_schema(schema: Dict[str, Any], *, path: str = "$") -> Tuple[int, List[str]]:
    enforced = 0
    errors: List[str] = []

    def _walk(node: Dict[str, Any], current_path: str) -> None:
        nonlocal enforced
        node_type = node.get("type")
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else None
        has_pattern_properties = bool(node.get("patternProperties"))
        if node_type == "object" and properties is not None and not has_pattern_properties:
            if "additionalProperties" not in node:
                node["additionalProperties"] = False
                enforced += 1
        required = node.get("required")
        if isinstance(required, list):
            defined = set(properties.keys()) if isinstance(properties, dict) else set()
            missing = [str(name) for name in required if name not in defined]
            if missing:
                missing_sorted = ", ".join(sorted(missing))
                errors.append(
                    f"{current_path}: required fields missing from properties: {missing_sorted}"
                )
        for child_path, child in _iter_schema_children(node, current_path):
            _walk(child, child_path)

    _walk(schema, path)
    return enforced, errors

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
RESPONSES_RESPONSE_PATH = Path("artifacts/debug/last_gpt5_responses_response.json")
RESPONSES_REQUEST_PATH = Path("artifacts/debug/last_gpt5_responses_request.json")


@dataclass(frozen=True)
class GenerationResult:
    """Container describing the outcome of a text generation call."""

    text: str
    model_used: str
    retry_used: bool
    fallback_used: Optional[str]
    fallback_reason: Optional[str] = None
    api_route: str = LLM_ROUTE
    schema: str = "none"
    metadata: Optional[Dict[str, object]] = None


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


def _format_diagnostics(details: Dict[str, object]) -> str:
    components: List[str] = []
    for key, value in details.items():
        if value is None or value == "":
            continue
        if isinstance(value, (list, dict)):
            try:
                value_repr = json.dumps(value, ensure_ascii=False)
            except TypeError:
                value_repr = str(value)
        else:
            value_repr = str(value)
        components.append(f"{key}={value_repr}")
    return ", ".join(components)


def _build_force_model_error(reason: str, details: Dict[str, object]) -> RuntimeError:
    diagnostics = _format_diagnostics(details)
    message = f"FORCE_MODEL active: {reason}"
    if diagnostics:
        message += f"; diagnostics: {diagnostics}"
    return RuntimeError(message)


def build_responses_payload(
    model: str,
    system_text: Optional[str],
    user_text: Optional[str],
    max_tokens: int,
    *,
    text_format: Optional[Dict[str, object]] = None,
    previous_response_id: Optional[str] = None,
) -> Dict[str, object]:
    """Construct a minimal Responses API payload for GPT-5 models."""

    sections: List[str] = []

    system_block = (system_text or "").strip()
    if system_block:
        sections.append(system_block)

    user_block = (user_text or "").strip()
    if user_block:
        sections.append(user_block)

    joined_input = "\n\n".join(section for section in sections if section)
    joined_input = re.sub(r"[ ]{2,}", " ", joined_input)
    joined_input = re.sub(r"\n{3,}", "\n\n", joined_input)

    format_block, _, _ = _prepare_text_format_for_request(
        text_format or DEFAULT_RESPONSES_TEXT_FORMAT,
        context="build_payload",
        log_on_migration=False,
    )

    payload: Dict[str, object] = {
        "model": str(model).strip(),
        "input": joined_input.strip(),
        "max_output_tokens": clamp_responses_max_output_tokens(max_tokens),
        "text": {"format": format_block},
    }
    if previous_response_id and previous_response_id.strip():
        payload["previous_response_id"] = previous_response_id.strip()
    return payload


def _shrink_responses_input(text_value: str) -> str:
    """Return a slightly condensed version of the Responses input payload."""

    if not text_value:
        return text_value

    normalized_lines: List[str] = []
    seen: set[str] = set()
    for raw_line in text_value.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        fingerprint = re.sub(r"\s+", " ", stripped.lower())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        normalized_lines.append(stripped)

    condensed = "\n\n".join(normalized_lines)
    if len(condensed) < len(text_value):
        return condensed
    target = max(1000, int(len(text_value) * 0.9))
    return text_value[:target]


def _parse_schema_instance(schema: Dict[str, Any], text: str) -> Tuple[Optional[object], bool]:
    if not schema or not text:
        return None, False

    try:
        instance = json.loads(text)
    except json.JSONDecodeError:
        return None, False

    if not isinstance(instance, (dict, list)):
        return None, False

    try:
        Draft7Validator.check_schema(schema)
    except JSONSchemaError:
        LOGGER.warning("RESP_INCOMPLETE_SCHEMA_INVALID schema=invalid")
        return None, False

    try:
        Draft7Validator(schema).validate(instance)
    except JSONSchemaValidationError as exc:
        LOGGER.warning("RESP_INCOMPLETE_SCHEMA_INVALID message=%s", exc.message)
        return None, False

    return instance, True


def _has_non_empty_content(node: object) -> bool:
    if isinstance(node, str):
        return bool(node.strip())
    if isinstance(node, list):
        return any(_has_non_empty_content(item) for item in node)
    if isinstance(node, dict):
        return any(_has_non_empty_content(value) for value in node.values())
    return False


def _is_valid_json_schema_instance(schema: Dict[str, Any], text: str) -> bool:
    """Validate the provided JSON text against the supplied schema."""

    _, valid = _parse_schema_instance(schema, text)
    return valid


def _coerce_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _sanitize_text_format_in_place(
    format_block: Dict[str, object],
    *,
    context: str = "-",
    log_on_migration: bool = True,
) -> Tuple[bool, bool, int]:
    migrated = False
    if not isinstance(format_block, dict):
        return False, False, 0

    removed_keys: List[str] = []

    def _pop_key(key: str) -> None:
        if key in format_block:
            format_block.pop(key, None)
            removed_keys.append(key)

    allowed_keys = {"type", "name", "schema", "strict"}

    if not isinstance(format_block.get("schema"), dict):
        if "schema" in format_block:
            _pop_key("schema")
            migrated = True

    if "json_schema" in format_block:
        removed_keys.append("json_schema")
    legacy_block = format_block.pop("json_schema", None)
    if isinstance(legacy_block, dict):
        schema_candidate = legacy_block.get("schema")
        if isinstance(schema_candidate, dict):
            format_block["schema"] = deepcopy(schema_candidate)
        strict_candidate = legacy_block.get("strict")
        strict_value = _coerce_bool(strict_candidate)
        if strict_value is not None and "strict" not in format_block:
            format_block["strict"] = strict_value
        migrated = True

    type_value = format_block.get("type")
    if isinstance(type_value, str):
        trimmed = type_value.strip()
        if trimmed:
            normalized = trimmed.lower()
            if trimmed != type_value:
                format_block["type"] = trimmed
                migrated = True
            if normalized == "output_text":
                format_block["type"] = "text"
                migrated = True
        else:
            _pop_key("type")
            migrated = True
    elif type_value is not None:
        trimmed = str(type_value).strip()
        if trimmed:
            normalized = trimmed.lower()
            format_block["type"] = "text" if normalized == "output_text" else trimmed
            migrated = True
        else:
            _pop_key("type")
            migrated = True

    name_value = format_block.get("name")
    if isinstance(name_value, str):
        trimmed = name_value.strip()
        if trimmed:
            if trimmed != name_value:
                format_block["name"] = trimmed
                migrated = True
        else:
            _pop_key("name")
            migrated = True
    elif name_value is not None:
        trimmed = str(name_value).strip()
        if trimmed:
            format_block["name"] = trimmed
            migrated = True
        else:
            _pop_key("name")
            migrated = True

    strict_value = format_block.get("strict")
    strict_bool = _coerce_bool(strict_value)
    if strict_bool is None:
        if "strict" in format_block:
            _pop_key("strict")
            migrated = True
    else:
        if strict_value is not strict_bool:
            format_block["strict"] = strict_bool
            migrated = True

    fmt_type_normalized = str(format_block.get("type", "")).strip().lower()
    if fmt_type_normalized == "text":
        stripped_any = False
        for forbidden_key in ("name", "schema", "strict"):
            if forbidden_key in format_block:
                _pop_key(forbidden_key)
                stripped_any = True
        if stripped_any:
            migrated = True

    for key in list(format_block.keys()):
        if key not in allowed_keys:
            _pop_key(key)
            migrated = True

    if "type" not in format_block and isinstance(format_block.get("schema"), dict):
        format_block["type"] = "json_schema"
        migrated = True

    schema_dict = format_block.get("schema")
    enforced_count = 0
    if isinstance(schema_dict, dict):
        enforced_count, errors = _normalize_json_schema(schema_dict)
        if errors:
            details = "; ".join(errors)
            raise SchemaValidationError(
                f"Invalid schema for text.format ({context}): {details}"
            )

    has_schema = isinstance(format_block.get("schema"), dict)
    fmt_type = str(format_block.get("type", "")).strip() or "-"
    fmt_name = str(format_block.get("name", "")).strip() or "-"

    if migrated and log_on_migration:
        LOGGER.info(
            "LOG:RESP_SCHEMA_MIGRATION_APPLIED context=%s type=%s name=%s",
            context,
            fmt_type,
            fmt_name,
        )
    if removed_keys:
        LOGGER.warning(
            "LOG:RESP_SCHEMA_KEYS_REMOVED context=%s keys=%s",
            context,
            sorted(set(removed_keys)),
        )
    if enforced_count > 0:
        LOGGER.info(
            "LOG:RESP_SCHEMA_ADDITIONAL_PROPS_ENFORCED context=%s type=%s name=%s count=%d",
            context,
            fmt_type,
            fmt_name,
            enforced_count,
        )

    return migrated, has_schema, enforced_count


def _prepare_text_format_for_request(
    template: Optional[Dict[str, object]],
    *,
    context: str,
    log_on_migration: bool = True,
) -> Tuple[Dict[str, object], bool, bool]:
    if not isinstance(template, dict):
        return {}, False, False
    working_copy: Dict[str, object] = deepcopy(template)
    migrated, has_schema, enforced_count = _sanitize_text_format_in_place(
        working_copy,
        context=context,
        log_on_migration=log_on_migration,
    )
    if not migrated and enforced_count <= 0 and log_on_migration:
        fmt_type = str(working_copy.get("type", "")).strip() or "-"
        fmt_name = str(working_copy.get("name", "")).strip() or "-"
        LOGGER.debug(
            "LOG:RESP_SCHEMA_NORMALIZED context=%s type=%s name=%s has_schema=%s",
            context,
            fmt_type,
            fmt_name,
            has_schema,
        )
    return working_copy, migrated, has_schema


def _sanitize_text_format_block(
    format_value: Dict[str, object],
    *,
    context: str,
    log_on_migration: bool = True,
) -> Optional[Dict[str, object]]:
    if not isinstance(format_value, dict):
        return None
    sanitized_format, _, _ = _prepare_text_format_for_request(
        format_value,
        context=context,
        log_on_migration=log_on_migration,
    )
    if not sanitized_format:
        return None
    return sanitized_format


def _sanitize_text_block(text_value: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not isinstance(text_value, dict):
        return None
    format_block = text_value.get("format")
    sanitized_format = _sanitize_text_format_block(
        format_block,
        context="sanitize_payload.text",
    )
    if not sanitized_format:
        return None
    return {"format": sanitized_format}


def sanitize_payload_for_responses(payload: Dict[str, object]) -> Tuple[Dict[str, object], int]:
    """Restrict Responses payload to the documented whitelist and types."""

    sanitized: Dict[str, object] = {}
    unexpected_keys = [key for key in payload.keys() if key not in RESPONSES_ALLOWED_KEYS]
    if unexpected_keys:
        LOGGER.warning(
            "RESP_PAYLOAD_TRIMMED unknown_keys=%s",
            sorted(str(key) for key in unexpected_keys),
        )
    for key in RESPONSES_ALLOWED_KEYS:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if key == "input":
                sanitized[key] = trimmed
                continue
            if not trimmed:
                continue
            if key == "model":
                sanitized[key] = trimmed
                continue
            if key == "previous_response_id":
                sanitized[key] = trimmed
                continue
        if key == "input" and not isinstance(value, str):
            if isinstance(value, (list, dict)):
                converted = json.dumps(value, ensure_ascii=False)
            else:
                converted = str(value)
            converted = converted.strip()
            if converted or "input" not in sanitized:
                sanitized[key] = converted
            continue
        if key == "max_output_tokens":
            try:
                sanitized[key] = clamp_responses_max_output_tokens(value)
            except (TypeError, ValueError):
                continue
            continue
        if key == "text":
            if isinstance(value, dict):
                sanitized_text = _sanitize_text_block(value)
                if sanitized_text:
                    sanitized["text"] = sanitized_text
            continue
    if "input" not in sanitized and "input" in payload:
        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            sanitized["input"] = raw_input.strip()
        elif raw_input is None:
            sanitized["input"] = ""
        else:
            sanitized["input"] = str(raw_input).strip()

    input_value = sanitized.get("input", "")
    input_length = len(input_value) if isinstance(input_value, str) else 0
    return sanitized, input_length


def _store_responses_request_snapshot(payload: Dict[str, object]) -> None:
    """Persist a sanitized snapshot of the latest Responses API request."""

    try:
        RESPONSES_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot = dict(payload)
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
        LOGGER.debug("failed to persist Responses request snapshot: %s", exc)


def _store_responses_response_snapshot(payload: Dict[str, object]) -> None:
    """Persist the latest Responses API payload for diagnostics."""

    try:
        RESPONSES_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESPONSES_RESPONSE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        LOGGER.debug("failed to persist Responses response snapshot: %s", exc)


def _infer_responses_step(payload_snapshot: Dict[str, object]) -> str:
    if not isinstance(payload_snapshot, dict):
        return "unknown"
    format_block: Optional[Dict[str, object]] = None
    candidate = payload_snapshot.get("response_format")
    if isinstance(candidate, dict):
        format_block = candidate
    else:
        text_block = payload_snapshot.get("text")
        if isinstance(text_block, dict):
            inner = text_block.get("format")
            if isinstance(inner, dict):
                format_block = inner
    if isinstance(format_block, dict):
        name_value = format_block.get("name")
        if isinstance(name_value, str):
            trimmed = name_value.strip()
            if trimmed:
                step = trimmed
                if "_" in trimmed:
                    step = trimmed.rsplit("_", 1)[-1]
                return step.lower()
    return "unknown"


def _handle_responses_http_error(
    error: httpx.HTTPStatusError,
    payload_snapshot: Dict[str, object],
    *,
    step: Optional[str] = None,
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
    step_name = (step or _infer_responses_step(payload_snapshot) or "unknown").strip() or "unknown"
    LOGGER.error(
        'Responses API error: status=%s error.type=%s error.message="%s" step=%s',
        status,
        error_type or "unknown",
        truncated,
        step_name,
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
    _store_responses_request_snapshot(payload_snapshot)
    if error_payload:
        _store_responses_response_snapshot(error_payload)


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
    parse_flags: Dict[str, object] = {}

    resp_keys = sorted(str(key) for key in data.keys())
    filtered_resp_keys = [key for key in resp_keys if key != "temperature"]
    parse_flags["resp_keys"] = filtered_resp_keys

    output_text_raw = data.get("output_text")
    if isinstance(output_text_raw, str):
        legacy_output_text = output_text_raw.strip()
    else:
        legacy_output_text = ""

    def _collect_text_branch(value: object) -> List[str]:
        collected: List[str] = []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                collected.append(stripped)
            return collected
        if isinstance(value, list):
            joined = _collect_text_parts(value)
            if joined:
                collected.append(joined)
            else:
                for item in value:
                    collected.extend(_collect_text_branch(item))
            return collected
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    stripped = candidate.strip()
                    if stripped:
                        collected.append(stripped)
                elif isinstance(candidate, list):
                    joined = _collect_text_parts(candidate)
                    if joined:
                        collected.append(joined)
                    else:
                        for item in candidate:
                            collected.extend(_collect_text_branch(item))
                elif isinstance(candidate, dict):
                    collected.extend(_collect_text_branch(candidate))
            for key in ("output", "outputs", "segments", "parts"):
                nested = value.get(key)
                if isinstance(nested, list):
                    joined = _collect_text_parts(nested)
                    if joined:
                        collected.append(joined)
                    else:
                        for item in nested:
                            collected.extend(_collect_text_branch(item))
                elif isinstance(nested, dict):
                    collected.extend(_collect_text_branch(nested))
        return [segment for segment in collected if segment]

    text_branch = data.get("text")
    text_segments = _collect_text_branch(text_branch)
    text_branch_value = "\n\n".join(text_segments) if text_segments else ""

    def _iter_segments(container: object) -> List[str]:
        collected: List[str] = []
        if isinstance(container, list):
            for item in container:
                collected.extend(_iter_segments(item))
        elif isinstance(container, dict):
            content = container.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type", "")).strip()
                    if part_type not in {"text", "output_text"}:
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        stripped = text_value.strip()
                        if stripped:
                            collected.append(stripped)
            for key in ("output", "outputs"):
                nested = container.get(key)
                if nested is not None:
                    collected.extend(_iter_segments(nested))
        return collected

    segments: List[str] = []
    root_used: Optional[str] = None
    for root_key in ("output", "outputs"):
        root_value = data.get(root_key)
        if root_value is None:
            continue
        extracted = _iter_segments(root_value)
        if extracted:
            segments.extend(extracted)
            root_used = root_key

    content_text = "\n\n".join(segments) if segments else ""
    primary_text = text_branch_value or legacy_output_text
    schema_label = (
        "responses.text"
        if text_branch_value
        else ("responses.output_text" if legacy_output_text else "responses.none")
    )
    parse_flags["schema"] = schema_label
    parse_flags["segments"] = len(segments)
    parse_flags["text_segments"] = len(text_segments)
    parse_flags["text_len"] = len(text_branch_value)
    parse_flags["output_text_len"] = len(primary_text)
    parse_flags["content_text_len"] = len(content_text)

    if text_branch_value:
        parse_source = "text"
        parse_length = parse_flags["text_len"]
    elif legacy_output_text:
        parse_source = "output_text"
        parse_length = len(legacy_output_text)
    elif content_text:
        parse_source = "content_text"
        parse_length = parse_flags["content_text_len"]
    else:
        parse_source = "none"
        parse_length = 0

    LOGGER.info(
        "RESP_PARSE=%s len=%d output_len=%d content_len=%d",
        parse_source,
        parse_length,
        parse_flags["output_text_len"],
        parse_flags["content_text_len"],
    )
    LOGGER.info(
        "responses parse resp_keys=%s root=%s segments=%d schema=%s",
        filtered_resp_keys,
        root_used,
        parse_flags.get("segments", 0),
        schema_label,
    )

    text = primary_text or content_text
    if text:
        LOGGER.info("RESP_PARSE_OK schema=%s len=%d", schema_label, len(text))
    return text, parse_flags, schema_label


def _resolve_model_name(model: Optional[str]) -> str:
    requested = (model or "").strip()
    if requested and requested != DEFAULT_MODEL:
        LOGGER.warning(
            "model override '%s' ignored; using %s",
            requested,
            DEFAULT_MODEL,
        )
    return DEFAULT_MODEL


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
    message = _extract_error_message(response)
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


def _extract_error_message(response: httpx.Response) -> str:
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
    return (message or "").strip()


def _has_text_format_migration_hint(response: httpx.Response) -> bool:
    message: str = ""
    message = _extract_error_message(response)
    if not message:
        return False
    return "moved to 'text.format'" in message.lower()


def _needs_text_type_retry(response: httpx.Response) -> bool:
    message = _extract_error_message(response)
    if not message:
        return False
    lowered = message.lower()
    if "text.format.type" in lowered and "output_text" in lowered:
        return True
    if "response_format" in lowered and "output_text" in lowered and "type" in lowered:
        return True
    if "unsupported" in lowered and "output_text" in lowered and "type" in lowered:
        return True
    if "expected" in lowered and "'text'" in lowered and "output_text" in lowered:
        return True
    return False


def _needs_format_name_retry(response: httpx.Response) -> bool:
    message = _extract_error_message(response)
    if not message:
        return False
    lowered = message.lower()
    if "text.format.name" in lowered:
        return True
    if "text.format" in lowered and "missing" in lowered and "name" in lowered:
        return True
    if "unsupported parameter" in lowered and "text.format" in lowered:
        return True
    if "moved to text.format" in lowered and "name" in lowered:
        return True
    return False


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
            input_candidate = current_payload.get("input", "")
            input_len = len(input_candidate) if isinstance(input_candidate, str) else 0
            LOGGER.info("responses input_len=%d", input_len)
            if "max_output_tokens" in current_payload:
                current_payload = dict(current_payload)
                current_payload["max_output_tokens"] = clamp_responses_max_output_tokens(
                    current_payload.get("max_output_tokens")
                )
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
    max_tokens: int = 1400,
    timeout_s: int = 60,
    backoff_schedule: Optional[List[float]] = None,
    responses_text_format: Optional[Dict[str, object]] = None,
    previous_response_id: Optional[str] = None,
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

    raw_timeout = timeout_s if timeout_s is not None else 60
    try:
        timeout_value = float(raw_timeout)
    except (TypeError, ValueError):
        timeout_value = 60.0
    effective_timeout = min(max(timeout_value, 1.0), 120.0)
    http_client = _acquire_http_client(effective_timeout)

    schedule = _resolve_backoff_schedule(backoff_schedule)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    gpt5_messages_cache: Optional[List[Dict[str, object]]] = None

    def _messages_for_model(target_model: str) -> List[Dict[str, object]]:
        nonlocal gpt5_messages_cache
        if target_model.lower().startswith("gpt-5"):
            if gpt5_messages_cache is None:
                gpt5_messages_cache = _cache_augmented_messages(messages)
            return [dict(message) for message in gpt5_messages_cache]
        return [dict(message) for message in messages]

    _PREVIOUS_ID_SENTINEL = object()

    def _call_responses_model(
        target_model: str,
        *,
        max_tokens_override: Optional[int] = None,
        text_format_override: Optional[Dict[str, object]] = None,
        previous_id_override: object = _PREVIOUS_ID_SENTINEL,
        max_attempts_override: Optional[int] = None,
        allow_empty_retry: bool = True,
    ) -> Tuple[str, Dict[str, object], Dict[str, object], str]:
        nonlocal retry_used

        payload_messages = _messages_for_model(target_model)
        style_template, _, _ = _prepare_text_format_for_request(
            text_format_override
            or responses_text_format
            or DEFAULT_RESPONSES_TEXT_FORMAT,
            context="style_probe",
            log_on_migration=False,
        )
        style_format_type = str(style_template.get("type", "")).strip().lower()
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
        system_text = _apply_living_style_instruction(
            system_text,
            format_type=style_format_type,
        )
        user_text = "\n\n".join(user_segments)

        effective_max_tokens = max_tokens_override if max_tokens_override is not None else max_tokens
        effective_previous_id: Optional[str]
        if previous_id_override is _PREVIOUS_ID_SENTINEL:
            effective_previous_id = previous_response_id
        else:
            effective_previous_id = previous_id_override if isinstance(previous_id_override, str) else None
        base_payload = build_responses_payload(
            target_model,
            system_text,
            user_text,
            effective_max_tokens,
            text_format=text_format_override or responses_text_format,
            previous_response_id=effective_previous_id,
        )
        sanitized_payload, _ = sanitize_payload_for_responses(base_payload)
        base_model_name = str(sanitized_payload.get("model") or target_model).strip()
        if not base_model_name:
            base_model_name = target_model

        text_section = sanitized_payload.get("text")
        if not isinstance(text_section, dict):
            text_section = {}
        format_template_source = text_section.get("format")
        if not isinstance(format_template_source, dict) or not format_template_source:
            raw_format_template = (
                text_format_override
                or responses_text_format
                or DEFAULT_RESPONSES_TEXT_FORMAT
            )
            format_template_source, _, _ = _prepare_text_format_for_request(
                raw_format_template,
                context="template",
                log_on_migration=False,
            )
        format_template = (
            deepcopy(format_template_source)
            if isinstance(format_template_source, dict)
            else {}
        )
        _sanitize_text_format_in_place(
            format_template,
            context="template_normalize",
            log_on_migration=False,
        )
        fmt_template_type = str(format_template.get("type", "")).strip().lower()
        if fmt_template_type == "json_schema":
            current_name = str(format_template.get("name", "")).strip()
            allowed_names = {RESPONSES_FORMAT_DEFAULT_NAME}
            fallback_name = str(
                FALLBACK_RESPONSES_PLAIN_OUTLINE_FORMAT.get("name", "")
            ).strip()
            if fallback_name:
                allowed_names.add(fallback_name)
            if current_name not in allowed_names:
                format_template["name"] = RESPONSES_FORMAT_DEFAULT_NAME

        def _clone_text_format() -> Dict[str, object]:
            return deepcopy(format_template)

        def _apply_text_format(target: Dict[str, object]) -> Dict[str, object]:
            text_container = target.get("text")
            if not isinstance(text_container, dict):
                text_container = {}
            format_block = text_container.get("format")
            if not isinstance(format_block, dict):
                format_block = _clone_text_format()
            else:
                format_block = deepcopy(format_block)
            text_container["format"] = format_block
            target["text"] = text_container
            return format_block

        def _ensure_format_name(
            target: Dict[str, object]
        ) -> Tuple[Dict[str, object], str, str, bool, bool]:
            format_block = _apply_text_format(target)
            _sanitize_text_format_in_place(
                format_block,
                context="normalize_format_block",
                log_on_migration=False,
            )
            fmt_type = str(format_block.get("type", "")).strip() or "-"
            has_schema = isinstance(format_block.get("schema"), dict)
            fmt_name = str(format_block.get("name", "")).strip()
            fixed = False
            if fmt_type.lower() == "json_schema":
                allowed_names = {RESPONSES_FORMAT_DEFAULT_NAME}
                fallback_name = str(
                    FALLBACK_RESPONSES_PLAIN_OUTLINE_FORMAT.get("name", "")
                ).strip()
                if fallback_name:
                    allowed_names.add(fallback_name)
                desired = RESPONSES_FORMAT_DEFAULT_NAME
                if fmt_name not in allowed_names:
                    format_block["name"] = desired
                    fmt_name = desired
                    fixed = True
            if not fmt_name:
                fmt_name = "-"
            return format_block, fmt_type, fmt_name, has_schema, fixed

        sanitized_payload["text"] = {"format": deepcopy(format_template)}

        raw_max_tokens = sanitized_payload.get("max_output_tokens")
        try:
            max_tokens_value = int(raw_max_tokens)
        except (TypeError, ValueError):
            max_tokens_value = 0
        if max_tokens_value <= 0:
            fallback_default = G5_MAX_OUTPUT_TOKENS_BASE if G5_MAX_OUTPUT_TOKENS_BASE > 0 else 1500
            max_tokens_value = fallback_default
        upper_cap = G5_MAX_OUTPUT_TOKENS_MAX if G5_MAX_OUTPUT_TOKENS_MAX > 0 else None
        if upper_cap is not None and max_tokens_value > upper_cap:
            LOGGER.info(
                "responses max_output_tokens clamped requested=%s limit=%s",
                raw_max_tokens,
                upper_cap,
            )
            max_tokens_value = upper_cap
        sanitized_payload["max_output_tokens"] = max_tokens_value
        LOGGER.info(
            "resolved max_output_tokens=%s (requested=%s, cap=%s)",
            max_tokens_value,
            raw_max_tokens if raw_max_tokens is not None else "-",
            upper_cap if upper_cap is not None else "-",
        )

        if "temperature" in sanitized_payload:
            sanitized_payload.pop("temperature", None)

        def _log_payload(snapshot: Dict[str, object]) -> None:
            keys = sorted(snapshot.keys())
            LOGGER.info("responses payload_keys=%s", keys)
            input_candidate = snapshot.get("input", "")
            length = len(input_candidate) if isinstance(input_candidate, str) else 0
            LOGGER.info("responses input_len=%d", length)
            LOGGER.info("responses max_output_tokens=%s", snapshot.get("max_output_tokens"))
            format_block: Optional[Dict[str, object]] = None
            text_block = snapshot.get("text")
            if isinstance(text_block, dict):
                candidate = text_block.get("format")
                if isinstance(candidate, dict):
                    format_block = candidate
            format_type = "-"
            format_name = "-"
            has_schema = False
            if isinstance(format_block, dict):
                fmt = format_block.get("type")
                if isinstance(fmt, str) and fmt.strip():
                    format_type = fmt.strip()
                name_candidate = format_block.get("name")
                if isinstance(name_candidate, str) and name_candidate.strip():
                    format_name = name_candidate.strip()
                has_schema = isinstance(format_block.get("schema"), dict)
            LOGGER.info(
                "responses text_format type=%s name=%s has_schema=%s",
                format_type,
                format_name,
                has_schema,
            )

        def _extract_metadata(payload: Dict[str, object]) -> Dict[str, object]:
            status_value = payload.get("status")
            status = str(status_value).strip().lower() if isinstance(status_value, str) else ""
            incomplete_details = payload.get("incomplete_details")
            incomplete_reason = ""
            if isinstance(incomplete_details, dict):
                reason = incomplete_details.get("reason")
                if isinstance(reason, str):
                    incomplete_reason = reason.strip().lower()
            usage_block = payload.get("usage")
            usage_output_tokens: Optional[float] = None
            if isinstance(usage_block, dict):
                raw_usage = usage_block.get("output_tokens")
                if isinstance(raw_usage, (int, float)):
                    usage_output_tokens = float(raw_usage)
                elif isinstance(raw_usage, dict):
                    for value in raw_usage.values():
                        if isinstance(value, (int, float)):
                            usage_output_tokens = float(value)
                            break
            response_id = ""
            raw_response_id = payload.get("id")
            if isinstance(raw_response_id, str):
                response_id = raw_response_id.strip()
            prev_response_id = ""
            raw_prev = payload.get("previous_response_id")
            if isinstance(raw_prev, str):
                prev_response_id = raw_prev.strip()
            metadata_block = payload.get("metadata")
            if isinstance(metadata_block, dict):
                if not prev_response_id:
                    prev_candidate = metadata_block.get("previous_response_id")
                    if isinstance(prev_candidate, str):
                        prev_response_id = prev_candidate.strip()
            finish_reason = ""
            finish_block = payload.get("finish_reason")
            if isinstance(finish_block, str):
                finish_reason = finish_block.strip().lower()
            metadata: Dict[str, object] = {
                "status": status,
                "incomplete_reason": incomplete_reason,
                "usage_output_tokens": usage_output_tokens,
                "response_id": response_id,
                "previous_response_id": prev_response_id,
                "finish_reason": finish_reason,
            }
            metadata["model_effective"] = target_model
            metadata["api_route"] = LLM_ROUTE
            metadata["allow_fallback"] = LLM_ALLOW_FALLBACK
            metadata["temperature_applied"] = False
            metadata["escalation_caps"] = list(G5_ESCALATION_LADDER)
            return metadata

        attempts = 0
        if max_attempts_override is not None:
            try:
                parsed_attempts = int(max_attempts_override)
            except (TypeError, ValueError):
                parsed_attempts = 1
            max_attempts = max(1, parsed_attempts)
        else:
            max_attempts = max(1, RESPONSES_MAX_ESCALATIONS + 1)
        current_max = max_tokens_value
        last_error: Optional[BaseException] = None
        format_retry_done = False
        format_type_retry_done = False
        format_name_retry_done = False
        min_tokens_bump_done = False
        min_token_floor = 1
        base_input_text = str(sanitized_payload.get("input", ""))
        shrunken_input = _shrink_responses_input(base_input_text)
        shrink_next_attempt = False
        shrink_applied = False
        incomplete_retry_count = 0
        token_escalations = 0
        resume_from_response_id: Optional[str] = None
        content_started = False
        cap_retry_performed = False
        empty_retry_attempted = False
        empty_direct_retry_attempted = False
        pending_degradation_flags: List[str] = []
        pending_completion_warning: Optional[str] = None

        def _record_pending_degradation(reason: str) -> None:
            nonlocal pending_completion_warning
            normalized_reason = (reason or "").strip().lower()
            if not normalized_reason:
                return
            flag_map = {
                "max_output_tokens": "draft_max_tokens",
                "soft_timeout": "draft_soft_timeout",
            }
            flag = flag_map.get(normalized_reason)
            if flag and flag not in pending_degradation_flags:
                pending_degradation_flags.append(flag)
            if not pending_completion_warning:
                pending_completion_warning = normalized_reason

        def _apply_pending_degradation(metadata: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(metadata, dict):
                metadata = {}
            else:
                metadata = dict(metadata)
            if pending_degradation_flags:
                existing: List[str] = []
                raw_flags = metadata.get("degradation_flags")
                if isinstance(raw_flags, list):
                    existing.extend(
                        str(flag).strip()
                        for flag in raw_flags
                        if isinstance(flag, str) and str(flag).strip()
                    )
                for flag in pending_degradation_flags:
                    if flag not in existing:
                        existing.append(flag)
                if existing:
                    metadata["degradation_flags"] = existing
            if pending_completion_warning and not metadata.get("completion_warning"):
                metadata["completion_warning"] = pending_completion_warning
            return metadata

        def _compute_next_max_tokens(current: int, step_index: int, cap: Optional[int]) -> int:
            ladder: List[int] = []
            for value in G5_ESCALATION_LADDER:
                try:
                    normalized = int(value)
                except (TypeError, ValueError):
                    continue
                if normalized <= 0:
                    continue
                if normalized not in ladder:
                    ladder.append(normalized)
            for target in ladder:
                if target > current:
                    return target if cap is None else min(target, cap)
            if cap is not None and cap > current:
                return int(cap)
            return current

        def _poll_responses_payload(response_id: str) -> Optional[Dict[str, object]]:
            poll_attempt = 0
            while poll_attempt < MAX_RESPONSES_POLL_ATTEMPTS:
                poll_attempt += 1
                poll_url = f"{RESPONSES_API_URL}/{response_id}"
                LOGGER.info("responses poll attempt=%d id=%s", poll_attempt, response_id)
                if poll_attempt == 1:
                    initial_sleep = schedule[0] if schedule else 0.5
                    LOGGER.info("responses poll initial sleep=%.2f", initial_sleep)
                    time.sleep(initial_sleep)
                try:
                    poll_response = http_client.get(
                        poll_url,
                        headers=headers,
                    )
                    poll_response.raise_for_status()
                except httpx.HTTPStatusError as poll_error:
                    _handle_responses_http_error(poll_error, {"poll_id": response_id})
                    break
                except httpx.HTTPError as transport_error:  # pragma: no cover - defensive
                    LOGGER.warning("responses poll transport error: %s", transport_error)
                    break
                try:
                    payload = poll_response.json()
                except ValueError:
                    LOGGER.warning("responses poll returned invalid JSON")
                    break
                if not isinstance(payload, dict):
                    break
                text, poll_parse_flags, _ = _extract_responses_text(payload)
                metadata = _extract_metadata(payload)
                poll_status = metadata.get("status") or ""
                poll_reason = metadata.get("incomplete_reason") or ""
                segments = int(poll_parse_flags.get("segments", 0) or 0)
                LOGGER.info("RESP_POLL_STATUS=%s|%s", poll_status or "ok", poll_reason or "-")
                if poll_status == "completed" and (text or segments > 0):
                    return payload
                if poll_status == "incomplete" and poll_reason == "max_output_tokens":
                    LOGGER.info(
                        "RESP_STATUS=incomplete|max_output_tokens=%s",
                        sanitized_payload.get("max_output_tokens"),
                    )
                    break
                if poll_attempt >= MAX_RESPONSES_POLL_ATTEMPTS:
                    break
                sleep_for = schedule[min(poll_attempt - 1, len(schedule) - 1)] if schedule else 0.5
                LOGGER.info("responses poll sleep=%.2f", sleep_for)
                time.sleep(sleep_for)
            return None

        while attempts < max_attempts:
            attempts += 1
            if resume_from_response_id:
                current_payload = {
                    "model": base_model_name,
                    "previous_response_id": resume_from_response_id,
                    "max_output_tokens": max(min_token_floor, int(current_max)),
                }
                continue_prompt = base_input_text if base_input_text else "Continue generation"
                current_payload["input"] = continue_prompt
                _apply_text_format(current_payload)
                LOGGER.info(
                    "RESP_CONTINUE previous_response_id=%s model=%s max_output_tokens=%s",
                    resume_from_response_id,
                    base_model_name,
                    current_payload.get("max_output_tokens"),
                )
            else:
                current_payload = dict(sanitized_payload)
                _apply_text_format(current_payload)
                if not content_started:
                    if shrink_applied and shrunken_input:
                        current_payload["input"] = shrunken_input
                    elif shrink_next_attempt:
                        shrink_next_attempt = False
                        if shrunken_input and shrunken_input != base_input_text:
                            current_payload["input"] = shrunken_input
                            shrink_applied = True
                            LOGGER.info(
                                "RESP_PROMPT_SHRINK original_len=%d shrunk_len=%d",
                                len(base_input_text),
                                len(shrunken_input),
                            )
                else:
                    if shrink_applied:
                        LOGGER.info("RESP_PROMPT_SHRINK_DISABLED after_content_started")
                    shrink_applied = False
                    shrink_next_attempt = False
                current_payload["max_output_tokens"] = max(min_token_floor, int(current_max))
            if attempts > 1:
                retry_used = True
            format_block, fmt_type, fmt_name, has_schema, fixed_name = _ensure_format_name(current_payload)
            suffix = " (fixed=name)" if fixed_name else ""
            LOGGER.info(
                "LOG:RESP_PAYLOAD_FORMAT type=%s name=%s has_schema=%s%s",
                fmt_type,
                fmt_name or "-",
                has_schema,
                suffix,
            )
            updated_format: Optional[Dict[str, object]] = None
            if isinstance(format_block, dict):
                try:
                    updated_format = deepcopy(format_block)
                except (TypeError, ValueError):
                    updated_format = _clone_text_format()
            if isinstance(updated_format, dict):
                sanitized_payload["text"] = {"format": deepcopy(updated_format)}
                format_template = deepcopy(updated_format)
            if isinstance(updated_format, dict):
                try:
                    format_snapshot = json.dumps(updated_format, ensure_ascii=False, sort_keys=True)
                except (TypeError, ValueError):
                    format_snapshot = str(updated_format)
                LOGGER.debug("DEBUG:payload.text.format = %s", format_snapshot)
                current_payload["text"] = {"format": deepcopy(updated_format)}
            else:
                LOGGER.debug("DEBUG:payload.text.format = null")
                current_payload["text"] = {"format": _clone_text_format()}
            _log_payload(current_payload)
            try:
                _store_responses_request_snapshot(current_payload)
                if "max_output_tokens" in current_payload:
                    current_payload = dict(current_payload)
                    current_payload["max_output_tokens"] = clamp_responses_max_output_tokens(
                        current_payload.get("max_output_tokens")
                    )
                response = http_client.post(
                    RESPONSES_API_URL,
                    headers=headers,
                    json=current_payload,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Модель вернула неожиданный формат ответа.")
                _store_responses_response_snapshot(data)
                text, parse_flags, schema_label = _extract_responses_text(data)
                metadata = _extract_metadata(data)
                if isinstance(parse_flags, dict):
                    parse_flags["metadata"] = metadata
                content_lengths = 0
                if isinstance(parse_flags, dict):
                    output_len = int(parse_flags.get("output_text_len", 0) or 0)
                    content_len = int(parse_flags.get("content_text_len", 0) or 0)
                    content_lengths = output_len + content_len
                if content_lengths > 0 and not content_started:
                    content_started = True
                    shrink_applied = False
                    shrink_next_attempt = False
                    LOGGER.info("RESP_CONTENT_STARTED len=%d", content_lengths)
                status = metadata.get("status") or ""
                reason = metadata.get("incomplete_reason") or ""
                segments = int(parse_flags.get("segments", 0) or 0)
                LOGGER.info("RESP_STATUS=%s|%s", status or "ok", reason or "-")
                if status in {"in_progress", "queued"}:
                    response_id = data.get("id")
                    if isinstance(response_id, str) and response_id.strip():
                        polled_payload = _poll_responses_payload(response_id.strip())
                        if polled_payload is None:
                            last_error = RuntimeError("responses_incomplete")
                            continue
                        data = polled_payload
                        text, parse_flags, schema_label = _extract_responses_text(data)
                        metadata = _extract_metadata(data)
                        if isinstance(parse_flags, dict):
                            parse_flags["metadata"] = metadata
                        content_lengths = 0
                        if isinstance(parse_flags, dict):
                            output_len = int(parse_flags.get("output_text_len", 0) or 0)
                            content_len = int(parse_flags.get("content_text_len", 0) or 0)
                            content_lengths = output_len + content_len
                        if content_lengths > 0 and not content_started:
                            content_started = True
                            shrink_applied = False
                            shrink_next_attempt = False
                            LOGGER.info("RESP_CONTENT_STARTED len=%d", content_lengths)
                        status = metadata.get("status") or ""
                        reason = metadata.get("incomplete_reason") or ""
                        segments = int(parse_flags.get("segments", 0) or 0)
                        LOGGER.info("RESP_STATUS=%s|%s", status or "ok", reason or "-")
                if status == "incomplete":
                    response_id_value = metadata.get("response_id") or ""
                    prev_field_present = "previous_response_id" in data or (
                        isinstance(metadata.get("previous_response_id"), str)
                        and metadata.get("previous_response_id")
                    )
                    if (
                        response_id_value
                        and reason in {"max_output_tokens", "soft_timeout"}
                        and (G5_ENABLE_PREVIOUS_ID_FETCH or prev_field_present)
                    ):
                        resume_from_response_id = str(response_id_value)
                    if reason == "max_output_tokens":
                        LOGGER.info(
                            "RESP_STATUS=incomplete|max_output_tokens=%s",
                            current_payload.get("max_output_tokens"),
                        )
                        schema_dict: Optional[Dict[str, Any]] = None
                        if isinstance(format_block, dict):
                            candidate_schema = format_block.get("schema")
                            if isinstance(candidate_schema, dict):
                                schema_dict = candidate_schema
                        if (
                            not text
                            and schema_label == "responses.none"
                            and segments == 0
                        ):
                            _record_pending_degradation(reason)
                        if text:
                            if schema_dict:
                                schema_instance: Optional[object] = None
                                schema_valid = False
                                schema_has_content = False
                                schema_instance, schema_valid = _parse_schema_instance(
                                    schema_dict, text
                                )
                                if schema_valid:
                                    schema_has_content = _has_non_empty_content(schema_instance)
                                if not (schema_valid and schema_has_content):
                                    LOGGER.info(
                                        "RESP_INCOMPLETE_RETRY schema_invalid len=%d content=%s",
                                        len(text),
                                        schema_has_content,
                                    )
                                    last_error = RuntimeError("responses_incomplete_schema_invalid")
                                    if response_id_value:
                                        resume_from_response_id = str(response_id_value)
                                    shrink_next_attempt = False
                                    text = ""
                                else:
                                    metadata = dict(metadata)
                                    metadata["status"] = "completed"
                                    metadata["incomplete_reason"] = ""
                                    metadata["completion_warning"] = "max_output_tokens"
                                    degradation_flags: List[str] = []
                                    raw_flags = metadata.get("degradation_flags")
                                    if isinstance(raw_flags, list):
                                        degradation_flags.extend(
                                            str(flag).strip()
                                            for flag in raw_flags
                                            if isinstance(flag, str) and flag.strip()
                                        )
                                    if "draft_max_tokens" not in degradation_flags:
                                        degradation_flags.append("draft_max_tokens")
                                    metadata["degradation_flags"] = degradation_flags
                                    metadata["completion_schema_valid"] = True
                                    metadata["completion_schema_content"] = bool(schema_has_content)
                                    LOGGER.info(
                                        "RESP_INCOMPLETE_ACCEPT schema_valid=%s content=%s len=%d",
                                        schema_valid,
                                        schema_has_content,
                                        len(text),
                                    )
                                    metadata = _apply_pending_degradation(metadata)
                                    parse_flags["metadata"] = metadata
                                    updated_data = dict(data)
                                    updated_data["metadata"] = metadata
                                    _persist_raw_response(updated_data)
                                    return text, parse_flags, updated_data, schema_label
                            else:
                                metadata = dict(metadata)
                                metadata["status"] = "completed"
                                metadata["incomplete_reason"] = ""
                                metadata["completion_warning"] = "max_output_tokens"
                                degradation_flags: List[str] = []
                                raw_flags = metadata.get("degradation_flags")
                                if isinstance(raw_flags, list):
                                    degradation_flags.extend(
                                        str(flag).strip()
                                        for flag in raw_flags
                                        if isinstance(flag, str) and flag.strip()
                                    )
                                if "draft_max_tokens" not in degradation_flags:
                                    degradation_flags.append("draft_max_tokens")
                                metadata["degradation_flags"] = degradation_flags
                                LOGGER.info(
                                    "RESP_INCOMPLETE_ACCEPT text len=%d",
                                    len(text),
                                )
                                metadata["completion_schema_valid"] = False
                                metadata = _apply_pending_degradation(metadata)
                                parse_flags["metadata"] = metadata
                                updated_data = dict(data)
                                updated_data["metadata"] = metadata
                                _persist_raw_response(updated_data)
                                return text, parse_flags, updated_data, schema_label
                        last_error = RuntimeError("responses_incomplete")
                        cap_exhausted = (
                            upper_cap is not None and int(current_max) >= upper_cap
                        )
                        if not cap_exhausted and token_escalations >= RESPONSES_MAX_ESCALATIONS:
                            if (
                                upper_cap is not None
                                and int(current_max) < upper_cap
                                and upper_cap > 0
                            ):
                                LOGGER.info(
                                    "RESP_ESCALATE_TOKENS reason=max_output_tokens cap_force=%s",
                                    upper_cap,
                                )
                                token_escalations += 1
                                current_max = upper_cap
                                sanitized_payload["max_output_tokens"] = max(
                                    min_token_floor, int(current_max)
                                )
                                cap_retry_performed = True
                                shrink_next_attempt = False
                                continue
                            break
                        if not cap_exhausted:
                            next_max = _compute_next_max_tokens(
                                int(current_max), token_escalations, upper_cap
                            )
                            if next_max <= int(current_max):
                                if (
                                    upper_cap is not None
                                    and int(current_max) >= upper_cap
                                ):
                                    cap_exhausted = True
                                else:
                                    break
                            if not cap_exhausted:
                                cap_label = (
                                    upper_cap if upper_cap is not None else "-"
                                )
                                LOGGER.info(
                                    "RESP_ESCALATE_TOKENS reason=max_output_tokens from=%s to=%s cap=%s",
                                    current_payload.get("max_output_tokens"),
                                    next_max,
                                    cap_label,
                                )
                                token_escalations += 1
                                current_max = next_max
                                if (
                                    upper_cap is not None
                                    and int(current_max) == upper_cap
                                ):
                                    cap_retry_performed = True
                                sanitized_payload["max_output_tokens"] = max(
                                    min_token_floor, int(current_max)
                                )
                                shrink_next_attempt = False
                                continue
                        if cap_exhausted:
                            output_length = 0
                            content_length = 0
                            if isinstance(parse_flags, dict):
                                output_length = int(
                                    parse_flags.get("output_text_len", 0) or 0
                                )
                                content_length = int(
                                    parse_flags.get("content_text_len", 0) or 0
                                )
                            LOGGER.warning(
                                "LLM_WARN cap_reached limit=%s output_len=%d content_len=%d status=%s reason=%s",
                                upper_cap,
                                output_length,
                                content_length,
                                status or "",
                                reason or "",
                            )
                            cap_retry_performed = True
                            shrink_next_attempt = False
                            current_max_tokens = int(current_max)
                            final_cap_reached = cap_exhausted and (
                                reason == "max_output_tokens_final"
                                or (
                                    reason == "max_output_tokens"
                                    and upper_cap is not None
                                    and current_max_tokens >= int(upper_cap)
                                )
                                or current_max_tokens >= 3600
                            )
                            if final_cap_reached and output_length <= 0:
                                _record_pending_degradation("max_output_tokens")
                                metadata = dict(metadata)
                                metadata["cap_reached_final"] = True
                                metadata["step_status"] = "degraded"
                                terminal_reason = (
                                    reason or "max_output_tokens"
                                )
                                if (
                                    terminal_reason != "max_output_tokens_final"
                                    and current_max_tokens >= 3600
                                ):
                                    terminal_reason = "max_output_tokens_final"
                                metadata["incomplete_reason"] = terminal_reason
                                metadata["degradation_reason"] = terminal_reason
                                existing_flags: List[str] = []
                                raw_flags = metadata.get("degradation_flags")
                                if isinstance(raw_flags, list):
                                    existing_flags = [
                                        str(flag).strip()
                                        for flag in raw_flags
                                        if isinstance(flag, str) and str(flag).strip()
                                    ]
                                if "draft_max_tokens" not in existing_flags:
                                    existing_flags.append("draft_max_tokens")
                                metadata["degradation_flags"] = existing_flags
                                if not metadata.get("completion_warning"):
                                    metadata["completion_warning"] = "max_output_tokens"
                                metadata = _apply_pending_degradation(metadata)
                                parse_flags["metadata"] = metadata
                                updated_data = dict(data)
                                updated_data["metadata"] = metadata
                                _persist_raw_response(updated_data)
                                return text or "", parse_flags, updated_data, schema_label
                    last_error = RuntimeError("responses_incomplete")
                    incomplete_retry_count += 1
                    if incomplete_retry_count >= 2:
                        break
                    shrink_next_attempt = True
                    continue
                if not text:
                    if (
                        status == "incomplete"
                        and reason == "soft_timeout"
                        and schema_label == "responses.none"
                        and segments == 0
                    ):
                        _record_pending_degradation(reason)
                    if (
                        allow_empty_retry
                        and status == "incomplete"
                        and segments == 0
                        and not empty_direct_retry_attempted
                    ):
                        empty_direct_retry_attempted = True
                        resume_from_response_id = None
                        shrink_next_attempt = False
                        reduced = int(round(int(current_max) * 0.85)) if current_max else 0
                        if reduced <= 0 or reduced >= int(current_max):
                            reduced = int(current_max) - 1 if int(current_max) > 1 else 1
                        if reduced < 1:
                            reduced = 1
                        current_max = reduced
                        sanitized_payload["max_output_tokens"] = max(
                            min_token_floor, int(current_max)
                        )
                        LOGGER.warning(
                            "RESP_EMPTY direct retry without previous_response_id max_tokens=%s",
                            sanitized_payload.get("max_output_tokens"),
                        )
                        last_error = RuntimeError("responses_empty_direct_retry")
                        continue
                    response_id_value = metadata.get("response_id") or ""
                    if (
                        allow_empty_retry
                        and response_id_value
                        and not empty_retry_attempted
                    ):
                        empty_retry_attempted = True
                        resume_from_response_id = str(response_id_value)
                        LOGGER.warning(
                            "RESP_EMPTY retrying with previous_response_id=%s",
                            resume_from_response_id,
                        )
                        last_error = RuntimeError("responses_empty_retry")
                        continue
                    last_error = EmptyCompletionError(
                        "Модель вернула пустой ответ",
                        raw_response=data,
                        parse_flags=parse_flags,
                    )
                    LOGGER.info("RESP_STATUS=json_error|segments=%d", segments)
                    if not allow_empty_retry:
                        raise last_error
                    continue
                metadata = _apply_pending_degradation(metadata)
                parse_flags["metadata"] = metadata
                updated_data = dict(data)
                updated_data["metadata"] = metadata
                _persist_raw_response(updated_data)
                return text, parse_flags, updated_data, schema_label
            except EmptyCompletionError as exc:
                last_error = exc
                if not allow_empty_retry:
                    raise
            except httpx.HTTPStatusError as exc:
                response_obj = exc.response
                status = response_obj.status_code if response_obj is not None else None
                if response_obj is not None and _needs_format_name_retry(response_obj):
                    setattr(exc, "responses_no_fallback", True)
                if (
                    status == 400
                    and not format_retry_done
                    and response_obj is not None
                    and _has_text_format_migration_hint(response_obj)
                ):
                    format_retry_done = True
                    retry_used = True
                    LOGGER.warning("RESP_RETRY_REASON=response_format_moved")
                    _apply_text_format(sanitized_payload)
                    continue
                if (
                    status == 400
                    and not format_type_retry_done
                    and response_obj is not None
                    and _needs_text_type_retry(response_obj)
                ):
                    format_type_retry_done = True
                    retry_used = True
                    LOGGER.warning("RESP_RETRY_REASON=text_format_type_migrated")
                    _apply_text_format(sanitized_payload)
                    text_block = sanitized_payload.get("text")
                    fmt_block = None
                    if isinstance(text_block, dict):
                        candidate = text_block.get("format")
                        if isinstance(candidate, dict):
                            fmt_block = candidate
                    if isinstance(fmt_block, dict):
                        fmt_block["type"] = "text"
                        format_template = deepcopy(fmt_block)
                    continue
                if (
                    status == 400
                    and response_obj is not None
                    and _needs_format_name_retry(response_obj)
                ):
                    if not format_name_retry_done:
                        format_name_retry_done = True
                        retry_used = True
                        LOGGER.warning(
                            "RESP_RETRY_REASON=format_name_missing route=responses attempt=%d",
                            attempts,
                        )
                        _apply_text_format(sanitized_payload)
                        format_block, _, _, _, _ = _ensure_format_name(sanitized_payload)
                        if isinstance(format_block, dict):
                            format_template = deepcopy(format_block)
                        continue
                if (
                    status == 400
                    and not min_tokens_bump_done
                    and is_min_tokens_error(response_obj)
                ):
                    min_tokens_bump_done = True
                    retry_used = True
                    min_token_floor = max(min_token_floor, 24)
                    current_max = max(current_max, min_token_floor)
                    sanitized_payload["max_output_tokens"] = max(current_max, min_token_floor)
                    LOGGER.warning("LOG:RESP_RETRY_REASON=max_tokens_min_bump")
                    continue
                if status == 400 and response_obj is not None:
                    shim_param = _extract_unknown_parameter_name(response_obj)
                    if shim_param:
                        retry_used = True
                        if shim_param in sanitized_payload:
                            sanitized_payload.pop(shim_param, None)
                        LOGGER.warning(
                            "retry=shim_unknown_param stripped='%s'",
                            shim_param,
                        )
                        continue
                last_error = exc
                step_label = _infer_responses_step(current_payload)
                _handle_responses_http_error(exc, current_payload, step=step_label)
                break
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, KeyboardInterrupt):
                    raise
                last_error = exc
            if attempts >= max_attempts:
                break
            sleep_for = schedule[min(attempts - 1, len(schedule) - 1)] if schedule else 0.5
            LOGGER.warning("responses retry attempt=%d sleep=%.2f", attempts, sleep_for)
            time.sleep(sleep_for)

        if last_error:
            if (
                isinstance(last_error, RuntimeError)
                and str(last_error) == "responses_incomplete"
                and cap_retry_performed
                and upper_cap is not None
                and int(current_max) >= upper_cap
            ):
                LOGGER.warning(
                    "LLM_WARN cap_reached limit=%s status=incomplete reason=max_output_tokens_final",
                    upper_cap,
                )
                last_error = EmptyCompletionError(
                    "Модель вернула пустой ответ",
                    raw_response={},
                    parse_flags={},
                )
            if isinstance(last_error, httpx.HTTPStatusError):
                _raise_for_last_error(last_error)
            if isinstance(last_error, (httpx.TimeoutException, httpx.TransportError)):
                _raise_for_last_error(last_error)
            raise last_error

        raise RuntimeError("Модель не вернула ответ.")


    lower_model = model_name.lower()
    is_gpt5_model = lower_model.startswith("gpt-5")

    retry_used = False

    if is_gpt5_model:
        available = _check_model_availability(
            http_client,
            provider=provider,
            headers=headers,
            model_name=model_name,
        )
        if not available:
            error_message = "Model GPT-5 not available for this key/plan"
            LOGGER.error("primary model %s unavailable", model_name)
            raise RuntimeError(error_message)

        def _scale_tokens(base: int, factor: float) -> int:
            if base <= 0:
                return 1
            scaled = int(round(base * factor))
            if scaled < 1:
                scaled = 1
            if scaled >= base:
                scaled = max(1, base - 1)
            return scaled

        try:
            primary_tokens = int(max_tokens)
        except (TypeError, ValueError):
            primary_tokens = 1400
        if primary_tokens <= 0:
            primary_tokens = 1400

        first_retry_tokens = _scale_tokens(primary_tokens, 0.85)
        fallback_tokens = _scale_tokens(first_retry_tokens, 0.9)

        attempt_plan = [
            {
                "label": "primary",
                "max_tokens": primary_tokens,
                "previous_id": _PREVIOUS_ID_SENTINEL,
                "format": responses_text_format,
                "fallback": None,
                "reason": None,
            },
            {
                "label": "retry_trimmed",
                "max_tokens": first_retry_tokens,
                "previous_id": None,
                "format": responses_text_format,
                "fallback": None,
                "reason": "empty_completion_retry",
            },
            {
                "label": "fallback_plain_outline",
                "max_tokens": fallback_tokens,
                "previous_id": None,
                "format": FALLBACK_RESPONSES_PLAIN_OUTLINE_FORMAT,
                "fallback": "plain_outline",
                "reason": "empty_completion_fallback",
            },
        ]

        last_empty_error: Optional[EmptyCompletionError] = None
        last_empty_plan: Optional[Dict[str, object]] = None

        for attempt_index, attempt_cfg in enumerate(attempt_plan):
            label = str(attempt_cfg["label"])
            LOGGER.info(
                "RESP_ATTEMPT label=%s max_tokens=%s fallback=%s",
                label,
                attempt_cfg["max_tokens"],
                attempt_cfg["fallback"] or "-",
            )
            if attempt_index > 0:
                retry_used = True
            try:
                text, parse_flags_current, _, schema_current = _call_responses_model(
                    model_name,
                    max_tokens_override=int(attempt_cfg["max_tokens"]),
                    text_format_override=attempt_cfg["format"],
                    previous_id_override=attempt_cfg["previous_id"],
                    allow_empty_retry=False,
                )
                schema_category = schema_current
                LOGGER.info(
                    "completion schema category=%s (schema=%s, route=responses, attempt=%s)",
                    schema_category,
                    schema_current,
                    label,
                )
                metadata_block = None
                if isinstance(parse_flags_current, dict):
                    meta_candidate = parse_flags_current.get("metadata")
                    if isinstance(meta_candidate, dict):
                        metadata_block = dict(meta_candidate)
                        metadata_block["max_output_tokens_applied"] = int(attempt_cfg["max_tokens"])
                        if attempt_cfg["fallback"]:
                            metadata_block["fallback_used"] = attempt_cfg["fallback"]
                            metadata_block["fallback_reason"] = attempt_cfg["reason"]
                return GenerationResult(
                    text=text,
                    model_used=model_name,
                    retry_used=retry_used or attempt_index > 0,
                    fallback_used=attempt_cfg["fallback"],
                    fallback_reason=attempt_cfg["reason"],
                    api_route=LLM_ROUTE,
                    schema=schema_category,
                    metadata=metadata_block,
                )
            except EmptyCompletionError as responses_empty:
                _persist_raw_response(responses_empty.raw_response)
                _log_parse_chain(responses_empty.parse_flags, retry=attempt_index, fallback=label)
                LOGGER.error(
                    "empty completion attempt=%s model=%s max_tokens=%s",
                    label,
                    model_name,
                    attempt_cfg["max_tokens"],
                )
                last_empty_error = responses_empty
                last_empty_plan = attempt_cfg
                continue
            except Exception as responses_error:  # noqa: BLE001
                LOGGER.warning("Responses API call failed on attempt=%s: %s", label, responses_error)
                if FORCE_MODEL:
                    error_details: Dict[str, object] = {
                        "reason": "api_error_gpt5_responses",
                        "exception": str(responses_error),
                        "attempt": label,
                    }
                    if isinstance(responses_error, httpx.HTTPStatusError):
                        status_code = (
                            responses_error.response.status_code
                            if responses_error.response is not None
                            else None
                        )
                        error_details["status_code"] = status_code
                        if responses_error.response is not None:
                            try:
                                payload_json = responses_error.response.json()
                            except ValueError:
                                payload_json = None
                            if isinstance(payload_json, dict):
                                error_block = payload_json.get("error")
                                if isinstance(error_block, dict):
                                    error_details["error_type"] = error_block.get("type")
                                    error_details["error_message"] = error_block.get("message")
                    raise _build_force_model_error("responses_error", error_details) from responses_error
                raise

        if last_empty_error is not None:
            diagnostics: Dict[str, object] = {"reason": "empty_completion_gpt5_responses"}
            parse_flags = last_empty_error.parse_flags or {}
            diagnostics["schema"] = parse_flags.get("schema")
            diagnostics["segments"] = parse_flags.get("segments")
            metadata_block = parse_flags.get("metadata")
            if isinstance(metadata_block, dict):
                diagnostics.update(
                    {
                        "status": metadata_block.get("status"),
                        "incomplete_reason": metadata_block.get("incomplete_reason"),
                        "usage_output_tokens": metadata_block.get("usage_output_tokens"),
                        "finish_reason": metadata_block.get("finish_reason"),
                        "previous_response_id": metadata_block.get("previous_response_id"),
                    }
                )
            if last_empty_plan:
                diagnostics["attempt"] = last_empty_plan.get("label")
                diagnostics["max_output_tokens"] = last_empty_plan.get("max_tokens")
                diagnostics["fallback"] = last_empty_plan.get("fallback")
            response_id = last_empty_error.raw_response.get("id")
            if isinstance(response_id, str) and response_id.strip():
                diagnostics["response_id"] = response_id.strip()
            if FORCE_MODEL:
                raise _build_force_model_error("responses_empty", diagnostics) from last_empty_error
            raise RuntimeError("Модель вернула пустой ответ. Попробуйте повторить генерацию.") from last_empty_error
    raise RuntimeError(f"Поддерживается только модель {LLM_MODEL.upper()} для маршрута {LLM_ROUTE}")


__all__ = ["generate", "DEFAULT_MODEL", "GenerationResult", "reset_http_client_cache"]
