from __future__ import annotations

import argparse
import json
import httpx
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

from assemble_messages import ContextBundle, assemble_messages, retrieve_context
from artifacts_store import _atomic_write_text as store_atomic_write_text, register_artifact
from config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    LLM_ALLOW_FALLBACK,
    LLM_ROUTE,
    MAX_CUSTOM_CONTEXT_CHARS,
    OPENAI_API_KEY,
)
from deterministic_pipeline import DeterministicPipeline, PipelineStep, PipelineStepError
from llm_client import (
    DEFAULT_MODEL,
    RESPONSES_API_URL,
    build_responses_payload,
    is_min_tokens_error,
    sanitize_payload_for_responses,
)
from keywords import parse_manual_keywords
from length_limits import ResolvedLengthLimits, resolve_length_limits
from validators import ValidationResult, length_no_spaces

BELGRADE_TZ = ZoneInfo("Europe/Belgrade")
TARGET_LENGTH_RANGE: Tuple[int, int] = (DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH)
LATEST_SCHEMA_VERSION = "2024-06"

HEALTH_MODEL = DEFAULT_MODEL
HEALTH_PROMPT = "Ответь ровно словом: PONG"
HEALTH_INITIAL_MAX_TOKENS = 10
HEALTH_MIN_BUMP_TOKENS = 24


@dataclass
class GenerationContext:
    data: Dict[str, Any]
    context_bundle: ContextBundle
    messages: List[Dict[str, Any]]
    clip_texts: List[str]
    style_profile_applied: bool = False
    style_profile_source: Optional[str] = None
    style_profile_variant: Optional[str] = None
    keywords_manual: List[str] = field(default_factory=list)
    context_source: str = "index.json"
    custom_context_text: Optional[str] = None
    custom_context_len: int = 0
    custom_context_filename: Optional[str] = None
    custom_context_hash: Optional[str] = None
    custom_context_truncated: bool = False
    jsonld_requested: bool = False
    length_limits: Optional[ResolvedLengthLimits] = None
    faq_questions: int = 0


def _local_now() -> datetime:
    return datetime.now(tz=BELGRADE_TZ)


def _ensure_artifacts_dir() -> Path:
    base = Path("artifacts").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _slugify(value: str) -> str:
    allowed = [ch if ch.isalnum() else "_" for ch in value]
    slug = "".join(allowed).strip("_")
    slug = "_".join(filter(None, slug.split("_")))
    if not slug:
        return "article"
    return slug[:80].lower()


def _strip_control_characters(text: str) -> str:
    cleaned: List[str] = []
    for char in text:
        if char == "\r":
            continue
        if ord(char) < 32 and char not in {"\n", "\t"}:
            continue
        cleaned.append(char)
    return "".join(cleaned)


def _collapse_blank_lines(text: str) -> str:
    lines = text.splitlines()
    collapsed: List[str] = []
    blank_pending = False
    for line in lines:
        if line.strip():
            collapsed.append(line.rstrip())
            blank_pending = False
            continue
        if not blank_pending and collapsed:
            collapsed.append("")
        blank_pending = True
    return "\n".join(collapsed).strip()


def _normalize_custom_context_text(
    raw_text: Any,
    *,
    max_chars: int = MAX_CUSTOM_CONTEXT_CHARS,
) -> Tuple[str, bool]:
    if not isinstance(raw_text, str):
        return "", False
    normalized = _strip_control_characters(raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    collapsed = _collapse_blank_lines(normalized)
    truncated = False
    if len(collapsed) > max_chars:
        collapsed = collapsed[:max_chars]
        truncated = True
    return collapsed, truncated


def _hash_context_snippet(text: str, *, byte_limit: int = 4096) -> Optional[str]:
    if not text:
        return None
    snippet = text.encode("utf-8")[:byte_limit]
    if not snippet:
        return None
    import hashlib

    return hashlib.sha256(snippet).hexdigest()


def make_generation_context(
    *,
    theme: str,
    data: Dict[str, Any],
    k: int,
    append_style_profile: Optional[bool] = None,
    context_source: Optional[str] = None,
    custom_context_text: Optional[str] = None,
    context_filename: Optional[str] = None,
) -> GenerationContext:
    payload = deepcopy(data)
    length_info = resolve_length_limits(theme, payload)
    payload["length_limits"] = {
        "min_chars": length_info.min_chars,
        "max_chars": length_info.max_chars,
    }
    payload["_length_limits_source"] = {
        "min": length_info.min_source,
        "max": length_info.max_source,
    }
    if length_info.profile_source:
        payload["_length_limits_profile_source"] = length_info.profile_source
    if length_info.warnings:
        payload["_length_limits_warnings"] = list(length_info.warnings)
    jsonld_requested = bool(payload.get("include_jsonld", False))
    payload.pop("include_jsonld", None)

    include_faq_flag = bool(payload.get("include_faq", True))
    faq_requested_raw = payload.get("faq_questions")
    resolved_faq = 0
    if include_faq_flag:
        try:
            resolved_faq = int(faq_requested_raw)
        except (TypeError, ValueError):
            resolved_faq = 0
        if resolved_faq <= 0:
            resolved_faq = 5
    payload["include_faq"] = include_faq_flag
    if resolved_faq > 0:
        payload["faq_questions"] = resolved_faq
    else:
        payload.pop("faq_questions", None)

    keywords_mode_raw = payload.get("keywords_mode")
    normalized_mode = None
    if isinstance(keywords_mode_raw, str):
        normalized_mode = keywords_mode_raw.strip().lower()
    if normalized_mode != "strict":
        payload["keywords_mode"] = "strict"

    requested_source = context_source if context_source is not None else payload.get("context_source")
    normalized_source = str(requested_source or "index.json").strip().lower() or "index.json"
    if normalized_source == "index":
        normalized_source = "index.json"
    payload["context_source"] = normalized_source

    raw_custom_context = custom_context_text
    if raw_custom_context is None and normalized_source == "custom":
        raw_custom_context = payload.get("context_text")

    filename = context_filename if context_filename is not None else payload.get("context_filename")
    if isinstance(filename, str):
        filename = filename.strip() or None
    else:
        filename = None

    payload.pop("context_text", None)
    if filename:
        payload["context_filename"] = filename
    else:
        payload.pop("context_filename", None)

    retrieval_k = k
    if normalized_source in {"off", "custom"}:
        retrieval_k = 0

    custom_context_normalized = ""
    custom_context_truncated = False
    custom_context_hash: Optional[str] = None
    custom_context_len = 0

    if normalized_source == "custom":
        custom_context_normalized, custom_context_truncated = _normalize_custom_context_text(
            raw_custom_context,
            max_chars=MAX_CUSTOM_CONTEXT_CHARS,
        )
        custom_context_len = len(custom_context_normalized)
        custom_context_hash = _hash_context_snippet(custom_context_normalized)
        tokens_est = len(custom_context_normalized.split())
        bundle = ContextBundle(
            items=[],
            total_tokens_est=tokens_est,
            index_missing=False,
            context_used=bool(custom_context_normalized),
            token_budget_limit=ContextBundle.token_budget_default(),
        )
    elif retrieval_k <= 0:
        bundle = ContextBundle(
            items=[],
            total_tokens_est=0,
            index_missing=False,
            context_used=False,
            token_budget_limit=ContextBundle.token_budget_default(),
        )
    else:
        bundle = retrieve_context(theme_slug=theme, query=payload.get("theme", ""), k=retrieval_k)

    manual_keywords = parse_manual_keywords(payload.get("keywords"))
    if manual_keywords:
        payload["keywords"] = manual_keywords
    else:
        payload.pop("keywords", None)

    messages = assemble_messages(
        data_path="",
        theme_slug=theme,
        k=retrieval_k,
        exemplars=bundle.items,
        data=payload,
        append_style_profile=append_style_profile,
        context_source=normalized_source,
        custom_context_text=custom_context_normalized,
    )
    clip_texts = [str(item.get("text", "")) for item in bundle.items if item.get("text")]
    style_profile_applied = False
    style_profile_source: Optional[str] = None
    style_profile_variant: Optional[str] = None
    for message in messages:
        if message.get("role") == "system" and message.get("style_profile_applied"):
            style_profile_applied = True
            style_profile_source = message.get("style_profile_source")
            style_profile_variant = message.get("style_profile_variant")
            break

    return GenerationContext(
        data=payload,
        context_bundle=bundle,
        messages=messages,
        clip_texts=clip_texts,
        style_profile_applied=style_profile_applied,
        style_profile_source=style_profile_source,
        style_profile_variant=style_profile_variant,
        keywords_manual=manual_keywords,
        context_source=normalized_source,
        custom_context_text=custom_context_normalized or None,
        custom_context_len=custom_context_len,
        custom_context_filename=filename,
        custom_context_hash=custom_context_hash,
        custom_context_truncated=custom_context_truncated,
        jsonld_requested=jsonld_requested,
        length_limits=length_info,
        faq_questions=resolved_faq if include_faq_flag else 0,
    )
def _make_output_path(theme: str, outfile: Optional[str]) -> Path:
    if outfile:
        return Path(outfile)
    timestamp = _local_now().strftime("%Y-%m-%d_%H%M")
    slug = _slugify(theme)
    filename = f"{timestamp}_{slug}_article.md"
    return _ensure_artifacts_dir() / filename


def _serialize_pipeline_logs(logs: Iterable[Any]) -> List[Dict[str, Any]]:
    serializable: List[Dict[str, Any]] = []
    for entry in logs:
        started_at = getattr(entry, "started_at", None)
        finished_at = getattr(entry, "finished_at", None)
        if isinstance(started_at, (int, float)):
            started_at = datetime.fromtimestamp(started_at, tz=ZoneInfo("UTC")).isoformat()
        if isinstance(finished_at, (int, float)):
            finished_at = datetime.fromtimestamp(finished_at, tz=ZoneInfo("UTC")).isoformat()
        payload = {
            "step": entry.step.value if hasattr(entry, "step") else str(entry),
            "status": getattr(entry, "status", "unknown"),
            "started_at": started_at,
            "finished_at": finished_at,
            "notes": getattr(entry, "notes", {}),
        }
        serializable.append(payload)
    return serializable


def _serialize_checkpoints(checkpoints: Dict[PipelineStep, str]) -> Dict[str, Dict[str, int]]:
    serialized: Dict[str, Dict[str, int]] = {}
    for step, text in checkpoints.items():
        serialized[step.value] = {
            "chars": len(text),
            "chars_no_spaces": len("".join(text.split())),
        }
    return serialized


def _build_metadata(
    *,
    theme: str,
    generation_context: GenerationContext,
    pipeline_state_text: str,
    validation: ValidationResult,
    pipeline_logs: Iterable[Any],
    checkpoints: Dict[PipelineStep, str],
    duration_seconds: float,
    model_used: Optional[str],
    fallback_used: Optional[str],
    fallback_reason: Optional[str],
    api_route: Optional[str],
    token_usage: Optional[float],
    degradation_flags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "schema_version": LATEST_SCHEMA_VERSION,
        "theme": theme,
        "generated_at": _local_now().isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "context_source": generation_context.context_source,
        "context_len": generation_context.custom_context_len,
        "context_filename": generation_context.custom_context_filename,
        "context_truncated": generation_context.custom_context_truncated,
        "style_profile_applied": generation_context.style_profile_applied,
        "style_profile_source": generation_context.style_profile_source,
        "style_profile_variant": generation_context.style_profile_variant,
        "keywords_manual": generation_context.keywords_manual,
        "length_limits": {
            "min": generation_context.length_limits.min_chars if generation_context.length_limits else TARGET_LENGTH_RANGE[0],
            "max": generation_context.length_limits.max_chars if generation_context.length_limits else TARGET_LENGTH_RANGE[1],
        },
        "faq_questions_requested": generation_context.faq_questions,
        "pipeline_logs": _serialize_pipeline_logs(pipeline_logs),
        "pipeline_checkpoints": _serialize_checkpoints(checkpoints),
        "validation": {
            "passed": validation.is_valid,
            "stats": validation.stats,
        },
        "length_no_spaces": length_no_spaces(pipeline_state_text),
    }
    if model_used:
        metadata["model_used"] = model_used
    if fallback_used:
        metadata["fallback_used"] = fallback_used
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason
    if api_route:
        metadata["api_route"] = api_route
    if isinstance(token_usage, (int, float)):
        metadata["token_usage"] = float(token_usage)
    if degradation_flags:
        normalized_flags = [
            str(flag).strip()
            for flag in degradation_flags
            if isinstance(flag, str) and str(flag).strip()
        ]
        if normalized_flags:
            metadata["degradation_flags"] = list(dict.fromkeys(normalized_flags))
    return metadata


def _write_outputs(markdown_path: Path, text: str, metadata: Dict[str, Any]) -> Dict[str, Path]:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    def _validate_markdown(tmp_path: Path) -> None:
        if not tmp_path.read_text(encoding="utf-8").strip():
            raise ValueError("Пустой файл статьи не может быть сохранён.")

    def _validate_metadata(tmp_path: Path) -> None:
        json.loads(tmp_path.read_text(encoding="utf-8"))

    store_atomic_write_text(markdown_path, text, validator=_validate_markdown)
    metadata_path = markdown_path.with_suffix(".json")
    store_atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2),
        validator=_validate_metadata,
    )
    register_artifact(markdown_path, metadata)
    return {"markdown": markdown_path, "metadata": metadata_path}


def _extract_keywords(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    raw_keywords = data.get("keywords") or []
    required: List[str] = []
    preferred: List[str] = []
    if isinstance(raw_keywords, dict):
        raw_required = raw_keywords.get("required", [])
        raw_preferred = raw_keywords.get("preferred", [])
        if isinstance(raw_required, (list, tuple, set)):
            required = [str(item).strip() for item in raw_required if str(item).strip()]
        elif isinstance(raw_required, str):
            required = [item.strip() for item in raw_required.split(",") if item.strip()]
        if isinstance(raw_preferred, (list, tuple, set)):
            preferred = [str(item).strip() for item in raw_preferred if str(item).strip()]
        elif isinstance(raw_preferred, str):
            preferred = [item.strip() for item in raw_preferred.split(",") if item.strip()]
    elif isinstance(raw_keywords, list):
        required = [str(item).strip() for item in raw_keywords if str(item).strip()]
    elif isinstance(raw_keywords, str):
        required = [item.strip() for item in raw_keywords.split(",") if item.strip()]

    normalized_required: List[str] = []
    seen = set()
    for term in required:
        if term not in seen:
            normalized_required.append(term)
            seen.add(term)

    normalized_preferred: List[str] = []
    for term in preferred:
        if term and term not in seen:
            normalized_preferred.append(term)
            seen.add(term)

    return normalized_required, normalized_preferred


def _prepare_outline(data: Dict[str, Any]) -> List[str]:
    raw_structure = data.get("structure") or []
    outline: List[str] = []
    if isinstance(raw_structure, list):
        for item in raw_structure:
            text = str(item).strip()
            if text:
                outline.append(text)
    return outline


def _generate_variant(
    *,
    theme: str,
    data: Dict[str, Any],
    data_path: str,
    k: int,
    model_name: str,
    max_tokens: int,
    timeout: int,
    mode: str,
    output_path: Path,
    variant_label: Optional[str] = None,
    backoff_schedule: Optional[List[float]] = None,
    append_style_profile: Optional[bool] = None,
    context_source: Optional[str] = None,
    context_text: Optional[str] = None,
    context_filename: Optional[str] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    payload = deepcopy(data)

    generation_context = make_generation_context(
        theme=theme,
        data=payload,
        k=k,
        append_style_profile=append_style_profile,
        context_source=context_source,
        custom_context_text=context_text,
        context_filename=context_filename,
    )

    prepared_data = generation_context.data
    length_limits = generation_context.length_limits or resolve_length_limits(theme, prepared_data)
    min_chars = length_limits.min_chars
    max_chars = length_limits.max_chars

    keywords_required, keywords_preferred = _extract_keywords(prepared_data)
    outline = _prepare_outline(prepared_data)
    topic = str(prepared_data.get("theme") or payload.get("theme") or theme).strip() or theme

    api_key = (os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    if not api_key:
        raise PipelineStepError(PipelineStep.SKELETON, "OPENAI_API_KEY не найден. Укажите действительный ключ.")

    pipeline = DeterministicPipeline(
        topic=topic,
        base_outline=outline,
        required_keywords=keywords_required,
        preferred_keywords=keywords_preferred,
        min_chars=min_chars,
        max_chars=max_chars,
        messages=generation_context.messages,
        model=model_name,
        max_tokens=max_tokens,
        timeout_s=timeout,
        backoff_schedule=backoff_schedule,
        provided_faq=prepared_data.get("faq_entries") if isinstance(prepared_data.get("faq_entries"), list) else None,
        jsonld_requested=generation_context.jsonld_requested,
        faq_questions=generation_context.faq_questions,
        progress_callback=progress_callback,
    )
    state = pipeline.run()
    if not state.validation or not state.validation.is_valid:
        raise RuntimeError("Pipeline validation failed; artifact not recorded.")

    final_text = state.text
    duration_seconds = time.time() - start_time
    metadata = _build_metadata(
        theme=theme,
        generation_context=generation_context,
        pipeline_state_text=final_text,
        validation=state.validation,
        pipeline_logs=state.logs,
        checkpoints=state.checkpoints,
        duration_seconds=duration_seconds,
        model_used=state.model_used,
        fallback_used=state.fallback_used,
        fallback_reason=state.fallback_reason,
        api_route=state.api_route,
        token_usage=state.token_usage,
        degradation_flags=pipeline.degradation_flags,
    )

    outputs = _write_outputs(output_path, final_text, metadata)
    return {
        "text": final_text,
        "metadata": metadata,
        "duration": duration_seconds,
        "artifact_files": outputs,
    }


def generate_article_from_payload(
    *,
    theme: str,
    data: Dict[str, Any],
    k: int,
    model: Optional[str] = None,
    max_tokens: int = 0,
    timeout: Optional[int] = None,
    mode: Optional[str] = None,
    backoff_schedule: Optional[List[float]] = None,
    outfile: Optional[str] = None,
    append_style_profile: Optional[bool] = None,
    context_source: Optional[str] = None,
    context_text: Optional[str] = None,
    context_filename: Optional[str] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    resolved_timeout = timeout if timeout is not None else 60
    resolved_model = DEFAULT_MODEL
    health_probe = _run_health_ping()
    if not health_probe.get("ok"):
        message = str(health_probe.get("message") or "health gate failed")
        raise RuntimeError(f"health_gate_failed: {message}")
    output_path = _make_output_path(theme, outfile)
    result = _generate_variant(
        theme=theme,
        data=data,
        data_path="<inline>",
        k=k,
        model_name=resolved_model,
        max_tokens=max_tokens,
        timeout=resolved_timeout,
        mode=mode or "final",
        output_path=output_path,
        backoff_schedule=backoff_schedule,
        append_style_profile=append_style_profile,
        context_source=context_source,
        context_text=context_text,
        context_filename=context_filename,
        progress_callback=progress_callback,
    )
    artifact_files = result.get("artifact_files")
    artifact_paths = None
    if artifact_files:
        artifact_paths = {
            "markdown": artifact_files["markdown"].as_posix(),
            "metadata": artifact_files["metadata"].as_posix(),
        }
    return {
        "text": result["text"],
        "metadata": result["metadata"],
        "artifact_paths": artifact_paths,
    }


def gather_health_status(theme: Optional[str]) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, object]] = {}

    artifacts_dir = Path("artifacts").resolve()
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        probe = artifacts_dir / ".write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["artifacts_writable"] = {
            "ok": True,
            "message": f"Артефакты доступны: {artifacts_dir}",
        }
    except Exception as exc:  # noqa: BLE001
        checks["artifacts_writable"] = {
            "ok": False,
            "message": f"Артефакты недоступны: {exc}",
        }

    api_key = (os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    masked_key = _mask_openai_key(api_key)
    if api_key:
        checks["openai_key"] = {
            "ok": True,
            "message": f"Ключ найден ({masked_key})",
        }
        checks["llm_ping"] = _run_health_ping()
    else:
        checks["openai_key"] = {
            "ok": False,
            "message": "OPENAI_API_KEY не найден",
        }
        checks["llm_ping"] = {
            "ok": False,
            "message": "Responses недоступен: ключ не задан",
            "route": LLM_ROUTE,
            "fallback_used": LLM_ALLOW_FALLBACK,
        }

    theme_slug = (theme or "").strip()
    if not theme_slug:
        checks["theme_index"] = {
            "ok": False,
            "message": "Тема не указана",
        }
    else:
        index_path = Path("profiles") / theme_slug / "index.json"
        if not index_path.exists():
            checks["theme_index"] = {
                "ok": False,
                "message": f"Индекс для темы '{theme_slug}' не найден",
            }
        else:
            try:
                json.loads(index_path.read_text(encoding="utf-8"))
                checks["theme_index"] = {
                    "ok": True,
                    "message": f"Профиль темы загружен ({index_path.as_posix()})",
                }
            except json.JSONDecodeError as exc:
                checks["theme_index"] = {
                    "ok": False,
                    "message": f"Индекс повреждён: {exc}",
                }

    ok = all(check.get("ok") is True for check in checks.values())
    return {"ok": ok, "checks": checks}


def _mask_openai_key(raw_key: str) -> str:
    key = (raw_key or "").strip()
    if not key:
        return "****"
    if key.startswith("sk-") and len(key) > 6:
        return f"sk-****{key[-4:]}"
    if len(key) <= 4:
        return "*" * len(key)
    return f"{key[:2]}***{key[-2:]}"


def _run_health_ping() -> Dict[str, object]:
    model = HEALTH_MODEL
    prompt = HEALTH_PROMPT
    max_tokens = HEALTH_INITIAL_MAX_TOKENS
    text_format = {"type": "text"}

    base_payload = build_responses_payload(
        model,
        None,
        prompt,
        max_tokens,
        text_format=text_format,
    )
    sanitized_payload, _ = sanitize_payload_for_responses(base_payload)
    sanitized_payload["text"] = {"format": deepcopy(text_format)}
    sanitized_payload["max_output_tokens"] = max_tokens

    api_key = (os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    if not api_key:
        return {
            "ok": False,
            "message": "Responses недоступен: ключ не задан",
            "route": LLM_ROUTE,
            "fallback_used": LLM_ALLOW_FALLBACK,
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    route = LLM_ROUTE
    fallback_used = LLM_ALLOW_FALLBACK
    model_url = f"https://api.openai.com/v1/models/{model}"

    start = time.perf_counter()
    attempts = 0
    max_attempts = 2
    min_bump_done = False
    current_max_tokens = max_tokens
    auto_bump_applied = False
    data: Optional[Dict[str, object]] = None
    response: Optional[httpx.Response] = None

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            model_probe = client.get(model_url, headers=headers)
            if model_probe.status_code != 200:
                detail = model_probe.text.strip()
                if len(detail) > 120:
                    detail = f"{detail[:117]}..."
                latency_ms = int((time.perf_counter() - start) * 1000)
                return {
                    "ok": False,
                    "message": (
                        f"Модель {model} недоступна: HTTP {model_probe.status_code}"
                        + (f" — {detail}" if detail else "")
                    ),
                    "route": "models",
                    "fallback_used": LLM_ALLOW_FALLBACK,
                    "latency_ms": latency_ms,
                }

            while attempts < max_attempts:
                attempts += 1
                payload_snapshot = dict(sanitized_payload)
                payload_snapshot["text"] = {"format": deepcopy(text_format)}
                payload_snapshot["max_output_tokens"] = current_max_tokens
                response = client.post(
                    RESPONSES_API_URL,
                    json=payload_snapshot,
                    headers=headers,
                )
                if (
                    response.status_code == 400
                    and not min_bump_done
                    and is_min_tokens_error(response)
                ):
                    current_max_tokens = max(current_max_tokens, HEALTH_MIN_BUMP_TOKENS)
                    sanitized_payload["max_output_tokens"] = current_max_tokens
                    min_bump_done = True
                    auto_bump_applied = True
                    continue
                break
    except httpx.TimeoutException:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "ok": False,
            "message": "Responses недоступен: таймаут",
            "route": route,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
        }
    except httpx.HTTPError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        reason = str(exc).strip() or exc.__class__.__name__
        return {
            "ok": False,
            "message": f"Responses недоступен: {reason}",
            "route": route,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
        }

    latency_ms = int((time.perf_counter() - start) * 1000)

    if response is None:
        return {
            "ok": False,
            "message": "Responses недоступен: нет ответа",
            "route": route,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
        }

    if response.status_code != 200:
        detail = response.text.strip()
        if len(detail) > 120:
            detail = f"{detail[:117]}..."
        return {
            "ok": False,
            "message": f"Responses недоступен: HTTP {response.status_code} — {detail or 'ошибка'}",
            "route": route,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
        }

    try:
        data = response.json()
    except ValueError:
        return {
            "ok": False,
            "message": "Responses недоступен: некорректный JSON",
            "route": route,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
        }

    status = str(data.get("status", "")).strip().lower()
    incomplete_reason = ""
    incomplete_details = data.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        reason_value = incomplete_details.get("reason")
        if isinstance(reason_value, str):
            incomplete_reason = reason_value.strip().lower()

    got_output = False
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        got_output = True
    elif isinstance(data.get("output"), list) or isinstance(data.get("outputs"), list):
        got_output = True

    extras: List[str] = []
    if auto_bump_applied:
        extras.append(f"auto-bump до {current_max_tokens}")

    if status == "completed":
        message = f"Responses OK (gpt-5, {current_max_tokens} токена"
        ok = True
    elif status == "incomplete" and incomplete_reason == "max_output_tokens":
        extras.insert(0, "incomplete по лимиту — норм для health")
        message = f"Responses OK (gpt-5, {current_max_tokens} токена"
        ok = True
    else:
        if not status:
            reason = "неизвестный статус"
        else:
            reason = status
            if incomplete_reason:
                reason = f"{reason} ({incomplete_reason})"
        return {
            "ok": False,
            "message": f"Responses недоступен: статус {reason}",
            "route": route,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
            "status": status or "",
        }

    if extras:
        message = f"{message}; {'; '.join(extras)})"
    else:
        message = f"{message})"

    result: Dict[str, object] = {
        "ok": ok,
        "message": message,
        "route": route,
        "fallback_used": fallback_used,
        "latency_ms": latency_ms,
        "tokens": current_max_tokens,
        "status": status,
        "incomplete_reason": incomplete_reason or None,
        "got_output": got_output,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic content pipeline")
    parser.add_argument("--theme", required=True, help="Theme slug (profiles/<theme>)")
    parser.add_argument("--data", required=True, help="Path to JSON brief")
    parser.add_argument("--outfile", help="Override output path")
    parser.add_argument("--k", type=int, default=0, help="Number of exemplar clips")
    parser.add_argument("--model", help="Optional model label for metadata")
    parser.add_argument("--max-tokens", type=int, default=0, dest="max_tokens")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--mode", default="final")
    parser.add_argument("--retry-backoff", help="Retry schedule (unused)")
    parser.add_argument("--append-style-profile", action="store_true")
    parser.add_argument("--context-source")
    parser.add_argument("--context-text")
    parser.add_argument("--context-filename")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _load_input(path: str) -> Dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Не найден файл входных данных: {payload_path}")
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {payload_path}: {exc}") from exc


def main() -> None:
    args = _parse_args()

    if args.check:
        status = gather_health_status(args.theme)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        sys.exit(0 if status.get("ok") else 1)

    data = _load_input(args.data)
    result = _generate_variant(
        theme=args.theme,
        data=data,
        data_path=args.data,
        k=args.k,
        model_name=(args.model or DEFAULT_MODEL).strip(),
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=_make_output_path(args.theme, args.outfile),
        append_style_profile=args.append_style_profile,
        context_source=args.context_source,
        context_text=args.context_text,
        context_filename=args.context_filename,
    )
    print(result["text"])


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
