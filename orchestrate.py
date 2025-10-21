# -*- coding: utf-8 -*-
"""End-to-end pipeline: assemble prompt → call LLM → store artefacts."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

import httpx

from assemble_messages import ContextBundle, assemble_messages, retrieve_context
from llm_client import DEFAULT_MODEL, GenerationResult, generate as llm_generate
from plagiarism_guard import is_too_similar
from artifacts_store import register_artifact
from config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    G5_MAX_OUTPUT_TOKENS_MAX,
    OPENAI_API_KEY,
)
from keywords import parse_manual_keywords
from post_analysis import (
    PostAnalysisRequirements,
    analyze as analyze_post,
    build_retry_instruction,
    should_retry as post_should_retry,
)


BELGRADE_TZ = ZoneInfo("Europe/Belgrade")
DEFAULT_CTA_TEXT = (
    "Семейная ипотека помогает молодым семьям купить жильё на понятных условиях. "
    "Сравните программы банков и сделайте первый шаг к дому своей мечты уже сегодня."
)
TARGET_LENGTH_RANGE: Tuple[int, int] = (DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH)
LENGTH_EXTEND_THRESHOLD = DEFAULT_MIN_LENGTH
LENGTH_SHRINK_THRESHOLD = DEFAULT_MAX_LENGTH
DISCLAIMER_TEMPLATE = (
    "⚠️ Дисклеймер: Материал носит информационный характер и не является финансовой рекомендацией. Прежде чем принимать решения, оцените риски и проконсультируйтесь со специалистом."
)


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


def _get_cta_text() -> str:
    cta = os.getenv("DEFAULT_CTA", DEFAULT_CTA_TEXT).strip()
    return cta or DEFAULT_CTA_TEXT


def _ensure_artifacts_dir() -> Path:
    base = Path("artifacts").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only)
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    if not sanitized:
        return "article"
    if len(sanitized) > 60:
        sanitized = sanitized[:60].rstrip("_") or sanitized[:60]
    return sanitized.lower()


def _resolve_cta_source(data: Dict[str, Any]) -> Tuple[str, bool]:
    custom_cta = str(data.get("cta", "")).strip()
    if custom_cta:
        return custom_cta, False
    return _get_cta_text(), True


def _is_truncated(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith("…") or stripped.endswith("...") or stripped.endswith(","):
        return True
    paragraphs = [para.strip() for para in stripped.splitlines() if para.strip()]
    if not paragraphs:
        return False
    last_paragraph = paragraphs[-1]
    return not last_paragraph.endswith((".", "!", "?"))


def _append_cta_if_needed(text: str, *, cta_text: str, default_cta: bool) -> Tuple[str, bool, bool]:
    if not _is_truncated(text):
        return text, False, False
    if text.strip():
        return text.rstrip() + "\n\n" + cta_text, True, default_cta
    return cta_text, True, default_cta


def _append_disclaimer_if_requested(text: str, data: Dict[str, Any]) -> Tuple[str, bool]:
    add_disclaimer = bool(data.get("add_disclaimer"))
    template = str(data.get("disclaimer_template") or DISCLAIMER_TEMPLATE).strip()
    if not add_disclaimer or not template:
        return text, False

    stripped = text.rstrip()
    if stripped.endswith(template):
        return text, False

    if stripped:
        return f"{stripped}\n\n{template}", True
    return template, True


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    if candidate <= 0:
        return default
    return candidate


def _safe_optional_positive_int(value: Any) -> Optional[int]:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0:
        return None
    return candidate


def _extract_source_values(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    result: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = str(item.get("value", "")).strip()
        else:
            value = str(item).strip()
        if value:
            result.append(value)
    return result


def _resolve_max_tokens_for_model(model_name: str, requested: int, max_chars: int) -> int:
    base = max(1, int(requested))
    lower_model = model_name.lower()
    if lower_model.startswith("gpt-5"):
        dynamic = max(1, int(max_chars / 3.5)) if max_chars > 0 else base
        base = dynamic
    return max(1, min(base, G5_MAX_OUTPUT_TOKENS_MAX))


def _should_expand_max_tokens(metadata: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(metadata, dict):
        return False
    reason = metadata.get("incomplete_reason")
    if isinstance(reason, str) and reason.strip().lower() == "max_output_tokens":
        return True
    return False


def _choose_section_for_extension(data: Dict[str, Any]) -> str:
    structure = data.get("structure")
    if isinstance(structure, Iterable):
        structure_list = [str(item).strip() for item in structure if str(item).strip()]
        if len(structure_list) >= 2:
            return structure_list[1]
        if structure_list:
            return structure_list[0]
    return "основную часть"


def _build_extend_prompt(section_name: str, *, min_target: int, max_target: int) -> str:
    return (
        f"Раскрой раздел «{section_name}», добавь факты и примеры, доведи объём до {min_target}\u2013{max_target} "
        "символов без пробелов, избегай повторов."
    )


def _build_shrink_prompt(*, min_target: int, max_target: int) -> str:
    return (
        f"Сократи повторы и второстепенные детали, приведи текст к {min_target}\u2013{max_target} символам без пробелов, "
        "сохрани исходную структуру."
    )


def _ensure_length(
    result: GenerationResult,
    messages: List[Dict[str, str]],
    *,
    data: Dict[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    min_target: Optional[int] = None,
    max_target: Optional[int] = None,
    backoff_schedule: Optional[List[float]] = None,
) -> Tuple[GenerationResult, Optional[str], List[Dict[str, str]]]:
    text = result.text
    length_no_spaces = len(re.sub(r"\s+", "", text))

    try:
        min_effective = int(min_target) if min_target is not None else LENGTH_EXTEND_THRESHOLD
    except (TypeError, ValueError):
        min_effective = LENGTH_EXTEND_THRESHOLD
    try:
        max_effective = int(max_target) if max_target is not None else LENGTH_SHRINK_THRESHOLD
    except (TypeError, ValueError):
        max_effective = LENGTH_SHRINK_THRESHOLD

    if max_effective < min_effective:
        max_effective = max(min_effective, LENGTH_SHRINK_THRESHOLD)

    if length_no_spaces < max(min_effective, 1):
        section = _choose_section_for_extension(data)
        prompt = _build_extend_prompt(section, min_target=min_effective, max_target=max_effective)
        adjusted_messages = list(messages)
        adjusted_messages.append({"role": "user", "content": prompt})
        new_result = llm_generate(
            adjusted_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        return new_result, "extend", adjusted_messages

    if length_no_spaces > max_effective:
        prompt = _build_shrink_prompt(min_target=min_effective, max_target=max_effective)
        adjusted_messages = list(messages)
        adjusted_messages.append({"role": "user", "content": prompt})
        new_result = llm_generate(
            adjusted_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        return new_result, "shrink", adjusted_messages

    return result, None, messages


def _local_now() -> datetime:
    return datetime.now(BELGRADE_TZ)


def make_generation_context(
    *,
    theme: str,
    data: Dict[str, Any],
    k: int,
    append_style_profile: Optional[bool] = None,
) -> GenerationContext:
    payload = deepcopy(data)
    if k <= 0:
        print("[orchestrate] CONTEXT: disabled (k=0)")
        bundle = ContextBundle(
            items=[],
            total_tokens_est=0,
            index_missing=False,
            context_used=False,
            token_budget_limit=ContextBundle.token_budget_default(),
        )
    else:
        bundle = retrieve_context(theme_slug=theme, query=payload.get("theme", ""), k=k)
        if bundle.index_missing:
            print("[orchestrate] CONTEXT: none (index missing)")

    manual_keywords = parse_manual_keywords(payload.get("keywords"))
    if manual_keywords:
        payload["keywords"] = manual_keywords
    else:
        payload.pop("keywords", None)

    messages = assemble_messages(
        data_path="",
        theme_slug=theme,
        k=k,
        exemplars=bundle.items,
        data=payload,
        append_style_profile=append_style_profile,
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
    )


def _default_timeout() -> int:
    env_timeout = os.getenv("LLM_TIMEOUT")
    try:
        return int(env_timeout) if env_timeout is not None else 60
    except ValueError:
        return 60


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an article using the configured LLM.")

    env_mode = os.getenv("GEN_MODE", "final").strip().lower() or "final"
    if env_mode not in {"draft", "final"}:
        env_mode = "final"
    default_timeout = _default_timeout()
    env_backoff = os.getenv("LLM_RETRY_BACKOFF")

    parser.add_argument("--theme", help="Theme slug (matches profiles/<theme>/...)")
    parser.add_argument("--data", help="Path to the JSON brief with generation parameters.")
    parser.add_argument(
        "--outfile",
        help="Optional path for the resulting markdown. Defaults to artifacts/<timestamp>__<theme>__article.md",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of exemplar clips to attach to CONTEXT (default: 0).",
    )
    parser.add_argument("--model", help="Override model name (otherwise uses LLM_MODEL env or default).")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3).")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1400,
        dest="max_tokens",
        help="Max tokens for generation (default: 1400).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=default_timeout,
        help="Timeout per request in seconds (default: 60 or LLM_TIMEOUT env).",
    )
    parser.add_argument(
        "--mode",
        choices=["draft", "final"],
        default=env_mode,
        help="Execution mode for metadata tags (defaults to GEN_MODE env or 'final').",
    )
    parser.add_argument("--ab", choices=["compare"], help="Run A/B comparison (compare: without vs with context).")
    parser.add_argument("--batch", help="Path to a JSON/YAML file describing batch generation payloads.")
    parser.add_argument("--check", action="store_true", help="Validate environment prerequisites and exit.")
    parser.add_argument(
        "--retry-backoff",
        default=env_backoff,
        help="Override retry backoff schedule in seconds, e.g. '0.5,1,2'.",
    )
    return parser.parse_args()


def _parse_backoff_schedule(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    schedule: List[float] = []
    for part in parts:
        try:
            schedule.append(float(part))
        except ValueError as exc:  # noqa: PERF203 - explicit feedback more helpful
            raise ValueError(f"Invalid retry backoff value: '{part}'") from exc
    return schedule


def _load_input(path: str) -> Dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Не найден файл входных данных: {payload_path}")
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {payload_path}: {exc}") from exc


def _resolve_model(cli_model: str | None) -> str:
    candidate = (cli_model or os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()
    return candidate or DEFAULT_MODEL


def _make_output_path(theme: str, outfile: str | None) -> Path:
    if outfile:
        return Path(outfile)
    timestamp = _local_now().strftime("%Y-%m-%d_%H%M")
    slug = _slugify(theme)
    filename = f"{timestamp}_{slug}_article.md"
    base_dir = _ensure_artifacts_dir()
    return base_dir / filename


def _write_outputs(markdown_path: Path, text: str, metadata: Dict[str, Any]) -> Dict[str, Path]:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text, encoding="utf-8")
    metadata_path = markdown_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        register_artifact(markdown_path, metadata)
    except Exception as exc:  # noqa: BLE001 - index update failures should not abort generation
        print(
            f"[orchestrate] warning: не удалось обновить индекс артефактов для {markdown_path}: {exc}",
            file=sys.stderr,
        )
    return {"markdown": markdown_path, "metadata": metadata_path}


def _generate_variant(
    *,
    theme: str,
    data: Dict[str, Any],
    data_path: str,
    k: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    mode: str,
    output_path: Path,
    variant_label: Optional[str] = None,
    backoff_schedule: Optional[List[float]] = None,
    append_style_profile: Optional[bool] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    payload = deepcopy(data)
    context_source = str(payload.get("context_source") or "index.json").strip().lower() or "index.json"
    effective_k = k
    if context_source == "off":
        effective_k = 0
    payload["context_source"] = context_source

    generation_context = make_generation_context(
        theme=theme,
        data=payload,
        k=effective_k,
        append_style_profile=append_style_profile,
    )
    prepared_data = generation_context.data
    active_messages = list(generation_context.messages)
    cta_text, cta_is_default = _resolve_cta_source(prepared_data)

    system_prompt = next((msg.get("content") for msg in active_messages if msg.get("role") == "system"), "")
    user_prompt = next((msg.get("content") for msg in reversed(active_messages) if msg.get("role") == "user"), "")

    length_limits_data = prepared_data.get("length_limits") or {}
    min_chars = _safe_positive_int(length_limits_data.get("min_chars"), DEFAULT_MIN_LENGTH)
    max_chars = _safe_positive_int(length_limits_data.get("max_chars"), DEFAULT_MAX_LENGTH)
    if max_chars < min_chars:
        max_chars = max(min_chars + 500, DEFAULT_MAX_LENGTH)

    keywords_required = [
        str(kw).strip()
        for kw in prepared_data.get("keywords", [])
        if isinstance(kw, str) and str(kw).strip()
    ]
    keyword_mode = str(prepared_data.get("keywords_mode") or "soft").strip().lower() or "soft"
    include_faq = bool(prepared_data.get("include_faq", True))
    faq_questions_raw = prepared_data.get("faq_questions") if include_faq else None
    faq_questions = _safe_optional_positive_int(faq_questions_raw)
    sources_values = _extract_source_values(prepared_data.get("sources"))
    requirements = PostAnalysisRequirements(
        min_chars=min_chars,
        max_chars=max_chars,
        keywords=list(keywords_required),
        keyword_mode=keyword_mode,
        faq_questions=faq_questions,
        sources=sources_values,
        style_profile=str(prepared_data.get("style_profile", "")),
    )

    max_tokens_requested = max_tokens
    max_tokens_current = _resolve_max_tokens_for_model(model_name, max_tokens_requested, max_chars)
    tokens_escalated = False

    llm_result = llm_generate(
        active_messages,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens_current,
        timeout_s=timeout,
        backoff_schedule=backoff_schedule,
    )

    article_text = llm_result.text
    effective_model = llm_result.model_used
    retry_used = llm_result.retry_used
    fallback_used = llm_result.fallback_used
    fallback_reason = llm_result.fallback_reason
    api_route = llm_result.api_route
    response_schema = llm_result.schema

    if model_name.lower().startswith("gpt-5"):
        escalation_attempts = 0
        while escalation_attempts < 2 and _should_expand_max_tokens(getattr(llm_result, "metadata", None)):
            candidate_limit = min(int(max_tokens_current * 1.2), G5_MAX_OUTPUT_TOKENS_MAX)
            if candidate_limit <= max_tokens_current:
                break
            max_tokens_current = candidate_limit
            tokens_escalated = True
            escalation_attempts += 1
            retry_used = True
            llm_result = llm_generate(
                active_messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens_current,
                timeout_s=timeout,
                backoff_schedule=backoff_schedule,
            )
            article_text = llm_result.text
            effective_model = llm_result.model_used
            fallback_used = llm_result.fallback_used
            fallback_reason = llm_result.fallback_reason
            api_route = llm_result.api_route
            response_schema = llm_result.schema
            retry_used = retry_used or llm_result.retry_used

    plagiarism_detected = False
    if generation_context.clip_texts and is_too_similar(article_text, generation_context.clip_texts):
        plagiarism_detected = True
        active_messages = list(active_messages)
        active_messages.append(
            {
                "role": "user",
                "content": "Перефразируй разделы, добавь списки и FAQ, избегай совпадений с примерами.",
            }
        )
        print("[orchestrate] Обнаружено совпадение с примерами, выполняю перегенерацию...", file=sys.stderr)
        llm_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens_current,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        article_text = llm_result.text
        effective_model = llm_result.model_used
        fallback_used = llm_result.fallback_used
        fallback_reason = llm_result.fallback_reason
        retry_used = True
        api_route = llm_result.api_route
        response_schema = llm_result.schema

    truncation_retry_used = False
    while True:
        llm_result, length_adjustment, active_messages = _ensure_length(
            llm_result,
            active_messages,
            data=prepared_data,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens_current,
            timeout=timeout,
            min_target=min_chars,
            max_target=max_chars,
            backoff_schedule=backoff_schedule,
        )
        article_text = llm_result.text
        effective_model = llm_result.model_used
        fallback_used = llm_result.fallback_used
        fallback_reason = llm_result.fallback_reason
        api_route = llm_result.api_route
        response_schema = llm_result.schema
        if not _is_truncated(article_text):
            break
        if truncation_retry_used:
            break
        truncation_retry_used = True
        print("[orchestrate] Детектор усечённого вывода — запускаю повторную генерацию", file=sys.stderr)
        llm_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens_current,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        article_text = llm_result.text
        effective_model = llm_result.model_used
        fallback_used = llm_result.fallback_used
        fallback_reason = llm_result.fallback_reason
        retry_used = True
        api_route = llm_result.api_route
        response_schema = llm_result.schema

    retry_used = retry_used or truncation_retry_used or llm_result.retry_used

    post_retry_attempts = 0
    post_analysis_report: Dict[str, object] = {}
    while True:
        article_text, postfix_appended, default_cta_used = _append_cta_if_needed(
            article_text,
            cta_text=cta_text,
            default_cta=cta_is_default,
        )
        article_text, disclaimer_appended = _append_disclaimer_if_requested(article_text, prepared_data)

        post_analysis_report = analyze_post(
            article_text,
            requirements=requirements,
            model=effective_model or model_name,
            retry_count=post_retry_attempts,
            fallback_used=bool(fallback_used),
        )
        if not post_should_retry(post_analysis_report) or post_retry_attempts >= 2:
            break
        refinement_instruction = build_retry_instruction(post_analysis_report, requirements)
        active_messages = list(active_messages)
        active_messages.append({"role": "user", "content": refinement_instruction})
        llm_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens_current,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        article_text = llm_result.text
        effective_model = llm_result.model_used
        fallback_used = llm_result.fallback_used
        fallback_reason = llm_result.fallback_reason
        api_route = llm_result.api_route
        response_schema = llm_result.schema
        retry_used = True
        post_retry_attempts += 1

    duration = time.time() - start_time
    context_bundle = generation_context.context_bundle
    context_used = bool(context_bundle.context_used and not context_bundle.index_missing and effective_k > 0)

    used_temperature = None
    if effective_model and not effective_model.lower().startswith("gpt-5"):
        used_temperature = temperature

    metadata: Dict[str, Any] = {
        "theme": theme,
        "data_path": data_path,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens_requested,
        "timeout_s": timeout,
        "retrieval_k": effective_k,
        "context_applied_k": len(context_bundle.items),
        "clips": [
            {
                "path": item.get("path"),
                "score": item.get("score"),
                "token_estimate": item.get("token_estimate"),
            }
            for item in context_bundle.items
        ],
        "plagiarism_detected": plagiarism_detected,
        "retry_used": retry_used,
        "generated_at": _local_now().isoformat(),
        "duration_seconds": round(duration, 3),
        "characters": len(article_text),
        "characters_no_spaces": len(re.sub(r"\s+", "", article_text)),
        "words": len(article_text.split()) if article_text.strip() else 0,
        "messages_count": len(active_messages),
        "context_used": context_used,
        "context_index_missing": context_bundle.index_missing,
        "context_budget_tokens_est": context_bundle.total_tokens_est,
        "context_budget_tokens_limit": context_bundle.token_budget_limit,
        "postfix_appended": postfix_appended,
        "length_adjustment": length_adjustment,
        "length_range_target": {"min": min_chars, "max": max_chars},
        "mode": mode,
        "model_used": effective_model,
        "temperature_used": used_temperature,
        "api_route": api_route,
        "response_schema": response_schema,
        "max_tokens_used": max_tokens_current,
        "max_tokens_escalated": tokens_escalated,
        "default_cta_used": default_cta_used,
        "truncation_retry_used": truncation_retry_used,
        "disclaimer_appended": disclaimer_appended,
        "facts_mode": prepared_data.get("facts_mode"),
        "input_data": prepared_data,
        "system_prompt_preview": system_prompt,
        "user_prompt_preview": user_prompt,
        "keywords_manual": generation_context.keywords_manual,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "length_limits": {"min_chars": min_chars, "max_chars": max_chars},
        "keywords_mode": keyword_mode,
        "sources_requested": prepared_data.get("sources"),
        "context_source": context_source,
        "include_faq": include_faq,
        "faq_questions": faq_questions,
        "include_jsonld": bool(prepared_data.get("include_jsonld", False)),
        "style_profile": prepared_data.get("style_profile"),
        "post_analysis": post_analysis_report,
        "post_analysis_retry_count": post_retry_attempts,
    }

    if generation_context.style_profile_applied:
        metadata["style_profile_applied"] = True
        if generation_context.style_profile_source:
            metadata["style_profile_source"] = generation_context.style_profile_source
        if generation_context.style_profile_variant:
            metadata["style_profile_variant"] = generation_context.style_profile_variant
    if variant_label:
        metadata["ab_variant"] = variant_label

    artifact_files: Optional[Dict[str, Path]] = None
    if article_text.strip():
        artifact_files = _write_outputs(output_path, article_text, metadata)
    else:
        print(
            f"[orchestrate] warning: пропускаю запись артефакта для {output_path.name} — пустой ответ",
            file=sys.stderr,
        )
    _summarise(theme, effective_k, effective_model or model_name, article_text, variant=variant_label)

    return {
        "text": article_text,
        "metadata": metadata,
        "output_path": output_path,
        "duration": duration,
        "variant": variant_label,
        "artifact_files": artifact_files,
    }


def generate_article_from_payload(
    *,
    theme: str,
    data: Dict[str, Any],
    k: int,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1400,
    timeout: Optional[int] = None,
    mode: Optional[str] = None,
    backoff_schedule: Optional[List[float]] = None,
    outfile: Optional[str] = None,
    append_style_profile: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for API usage.

    Returns
    -------
    Dict[str, Any]
        Dictionary with text, metadata and resulting artifact paths.
    """

    resolved_mode = (mode or os.getenv("GEN_MODE") or "final").strip().lower() or "final"
    if resolved_mode not in {"draft", "final"}:
        resolved_mode = "final"

    resolved_timeout = timeout if timeout is not None else _default_timeout()
    resolved_model = _resolve_model(model)
    if backoff_schedule is None:
        backoff_schedule = _parse_backoff_schedule(os.getenv("LLM_RETRY_BACKOFF"))

    output_path = _make_output_path(theme, outfile)
    result = _generate_variant(
        theme=theme,
        data=data,
        data_path="<inline>",
        k=k,
        model_name=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=resolved_timeout,
        mode=resolved_mode,
        output_path=output_path,
        backoff_schedule=backoff_schedule,
        append_style_profile=append_style_profile,
    )

    artifact_files = result.get("artifact_files")
    artifact_paths: Optional[Dict[str, str]] = None
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


def _summarise(theme: str, k: int, model: str, text: str, *, variant: str | None = None) -> None:
    chars = len(text)
    words = len(text.split()) if text.strip() else 0
    suffix = f" variant={variant}" if variant else ""
    print(f"[orchestrate] theme={theme}{suffix} k={k} model={model} length={chars} chars / {words} words")


def _suffix_output_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")


def _run_ab_compare(
    *,
    theme: str,
    data: Dict[str, Any],
    data_path: str,
    model_name: str,
    args: argparse.Namespace,
    base_output_path: Path,
    backoff_schedule: Optional[List[float]] = None,
) -> None:
    path_a = _suffix_output_path(base_output_path, "__A")
    path_b = _suffix_output_path(base_output_path, "__B")

    result_a = _generate_variant(
        theme=theme,
        data=data,
        data_path=data_path,
        k=0,
        model_name=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=path_a,
        variant_label="A",
        backoff_schedule=backoff_schedule,
    )

    result_b = _generate_variant(
        theme=theme,
        data=data,
        data_path=data_path,
        k=max(args.k, 0),
        model_name=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=path_b,
        variant_label="B",
        backoff_schedule=backoff_schedule,
    )

    len_a = len(result_a["text"])
    len_b = len(result_b["text"])
    duration_a = result_a["duration"]
    duration_b = result_b["duration"]
    print(
        "[orchestrate][A/B] len_A=%d len_B=%d Δlen=%+d duration_A=%.2fs duration_B=%.2fs Δt=%.2fs"
        % (len_a, len_b, len_b - len_a, duration_a, duration_b, duration_b - duration_a)
    )


def _load_batch_config(path: str) -> List[Dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл батча не найден: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Не удалось разобрать YAML. Установите PyYAML или используйте JSON."
            ) from exc
        data = yaml.safe_load(raw)
    if not isinstance(data, list):
        raise ValueError("Батч-файл должен содержать массив заданий.")
    return data


def _resolve_batch_entry(
    entry: Dict[str, Any],
    *,
    default_theme: Optional[str],
    default_mode: str,
    default_k: int,
    default_temperature: float,
    default_max_tokens: int,
    default_timeout: int,
    default_model: Optional[str],
) -> Tuple[str, Dict[str, Any], str, int, Optional[str], str, float, int, int, str]:
    theme = entry.get("theme") or default_theme
    if not theme:
        raise ValueError("Для задания в батче требуется указать theme либо задать его на уровне CLI.")

    data_field = entry.get("data")
    payload_field = entry.get("payload")
    if isinstance(data_field, dict):
        payload = data_field
        data_path = entry.get("data_path") or "<inline>"
    elif isinstance(payload_field, dict):
        payload = payload_field
        data_path = entry.get("data_path") or "<inline>"
    elif isinstance(data_field, str):
        payload = _load_input(data_field)
        data_path = str(Path(data_field).resolve())
    else:
        raise ValueError("Поле data должно быть путем к JSON или объектом с параметрами.")

    outfile = entry.get("outfile")
    mode = entry.get("mode", default_mode)
    k = int(entry.get("k", default_k))
    temperature = float(entry.get("temperature", default_temperature))
    max_tokens = int(entry.get("max_tokens", default_max_tokens))
    timeout = int(entry.get("timeout", default_timeout))
    model_name = _resolve_model(entry.get("model") or default_model)

    return theme, payload, data_path, k, outfile, mode, temperature, max_tokens, timeout, model_name


def _run_batch(args: argparse.Namespace) -> None:
    batch_items = _load_batch_config(args.batch)
    start = time.time()
    report_rows: List[Dict[str, Any]] = []
    successes = 0
    backoff_schedule = _parse_backoff_schedule(args.retry_backoff)

    for idx, entry in enumerate(batch_items, start=1):
        try:
            (
                theme,
                payload,
                data_path,
                k,
                outfile,
                mode,
                temperature,
                max_tokens,
                timeout,
                model_name,
            ) = _resolve_batch_entry(
                entry,
                default_theme=args.theme,
                default_mode=args.mode,
                default_k=args.k,
                default_temperature=args.temperature,
                default_max_tokens=args.max_tokens,
                default_timeout=args.timeout,
                default_model=args.model,
            )

            if outfile:
                output_path = Path(outfile)
            else:
                base_path = _make_output_path(theme, None)
                output_path = base_path.with_name(f"{base_path.stem}__{idx:02d}{base_path.suffix}")

            result = _generate_variant(
                theme=theme,
                data=payload,
                data_path=data_path,
                k=k,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                mode=mode,
                output_path=output_path,
                backoff_schedule=backoff_schedule,
            )

            report_rows.append(
                {
                    "index": idx,
                    "theme": theme,
                    "output_path": str(result["output_path"]),
                    "metadata_path": str(result["output_path"].with_suffix(".json")),
                    "characters": len(result["text"]),
                    "duration_seconds": round(result["duration"], 3),
                    "status": "ok",
                }
            )
            successes += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[batch] Ошибка в задании #{idx}: {exc}", file=sys.stderr)
            report_rows.append(
                {
                    "index": idx,
                    "theme": entry.get("theme"),
                    "status": "error",
                    "error": str(exc),
                }
            )

    total_duration = time.time() - start
    print(
        f"[batch] Completed {successes}/{len(batch_items)} items in {total_duration:.2f}s"
    )

    report = {
        "generated_at": _local_now().isoformat(),
        "total": len(batch_items),
        "success": successes,
        "duration_seconds": round(total_duration, 3),
        "results": report_rows,
    }
    report_path = Path("artifacts") / "batch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_api_key(api_key: str) -> str:
    cleaned = api_key.strip()
    if len(cleaned) <= 8:
        return "*" * len(cleaned)
    return f"{cleaned[:4]}{'*' * (len(cleaned) - 8)}{cleaned[-4:]}"


def gather_health_status(theme: Optional[str]) -> Dict[str, Any]:
    """Programmatic variant of ``--check`` used by the API server."""

    checks: Dict[str, Dict[str, object]] = {}
    ok = True

    api_key = (os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    openai_ok = False
    if not api_key:
        checks["openai_key"] = {"ok": False, "message": "OPENAI_API_KEY не найден"}
        ok = False
    else:
        masked = _mask_api_key(api_key)
        try:
            response = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                openai_ok = True
                checks["openai_key"] = {"ok": True, "message": f"Ключ активен ({masked})"}
            else:
                ok = False
                checks["openai_key"] = {
                    "ok": False,
                    "message": f"HTTP {response.status_code} при проверке ключа ({masked})",
                }
        except httpx.HTTPError as exc:
            ok = False
            checks["openai_key"] = {
                "ok": False,
                "message": f"Ошибка при обращении к OpenAI ({masked}): {exc}",
            }

    artifacts_dir = Path("artifacts")
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        probe = artifacts_dir / ".write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["artifacts_writable"] = {"ok": True, "message": "Запись в artifacts/ доступна"}
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks["artifacts_writable"] = {"ok": False, "message": f"Нет доступа к artifacts/: {exc}"}

    theme_slug = (theme or "").strip()
    if not theme_slug:
        checks["theme_index"] = {"ok": False, "message": "Тема не указана"}
        ok = False
    else:
        index_path = Path("profiles") / theme_slug / "index.json"
        if not index_path.exists():
            checks["theme_index"] = {
                "ok": False,
                "message": f"Индекс для темы '{theme_slug}' не найден",
            }
            ok = False
        else:
            try:
                json.loads(index_path.read_text(encoding="utf-8"))
                checks["theme_index"] = {"ok": True, "message": f"Индекс найден ({index_path})"}
            except json.JSONDecodeError as exc:
                ok = False
                checks["theme_index"] = {
                    "ok": False,
                    "message": f"Индекс повреждён: {exc}",
                }

    return {"ok": ok, "checks": checks, "openai_key": openai_ok}


def _run_checks(args: argparse.Namespace) -> None:
    ok = True

    python_version = sys.version.split()[0]
    print(f"✅ Python version: {python_version}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY не найден")
        ok = False
    else:
        masked = _mask_api_key(api_key)
        try:
            response = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                print(f"✅ OPENAI_API_KEY проверен ({masked})")
            else:
                print(f"❌ OPENAI_API_KEY отклонён ({masked}): HTTP {response.status_code}")
                ok = False
        except httpx.HTTPError as exc:
            print(f"❌ Не удалось проверить OPENAI_API_KEY ({masked}): {exc}")
            ok = False

    artifacts_dir = Path("artifacts")
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        probe = artifacts_dir / ".write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print("✅ Права на запись в artifacts/ подтверждены")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Нет доступа к artifacts/: {exc}")
        ok = False

    theme = (args.theme or "").strip()
    if not theme:
        print("❌ Тема не указана (--theme), невозможно проверить индекс.")
        ok = False
    else:
        index_path = Path("profiles") / theme / "index.json"
        if not index_path.exists():
            print(f"❌ Индекс для темы '{theme}' не найден: {index_path}")
            ok = False
        else:
            try:
                json.loads(index_path.read_text(encoding="utf-8"))
                print(f"✅ Индекс найден для темы '{theme}' ({index_path})")
            except json.JSONDecodeError as exc:
                print(f"❌ Индекс для темы '{theme}' повреждён: {exc}")
                ok = False

    sys.exit(0 if ok else 1)


def main() -> None:
    args = _parse_args()

    if args.check:
        _run_checks(args)
        return

    if args.batch:
        _run_batch(args)
        return

    if not args.theme or not args.data:
        raise ValueError("Параметры --theme и --data обязательны для одиночного запуска.")

    data = _load_input(args.data)
    data_path = str(Path(args.data).resolve())
    model_name = _resolve_model(args.model)
    base_output_path = _make_output_path(args.theme, args.outfile)

    backoff_schedule = _parse_backoff_schedule(args.retry_backoff)

    if args.ab == "compare":
        _run_ab_compare(
            theme=args.theme,
            data=data,
            data_path=data_path,
            model_name=model_name,
            args=args,
            base_output_path=base_output_path,
            backoff_schedule=backoff_schedule,
        )
        return

    result = _generate_variant(
        theme=args.theme,
        data=data,
        data_path=data_path,
        k=args.k,
        model_name=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=base_output_path,
        backoff_schedule=backoff_schedule,
    )
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
