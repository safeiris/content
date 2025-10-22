# -*- coding: utf-8 -*-
"""End-to-end pipeline: assemble prompt → call LLM → store artefacts."""
from __future__ import annotations

import argparse
import hashlib
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
    MAX_CUSTOM_CONTEXT_CHARS,
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
from retrieval import estimate_tokens


BELGRADE_TZ = ZoneInfo("Europe/Belgrade")
DEFAULT_CTA_TEXT = (
    "Семейная ипотека помогает молодым семьям купить жильё на понятных условиях. "
    "Сравните программы банков и сделайте первый шаг к дому своей мечты уже сегодня."
)
TARGET_LENGTH_RANGE: Tuple[int, int] = (DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH)
LENGTH_EXTEND_THRESHOLD = DEFAULT_MIN_LENGTH
QUALITY_EXTEND_MAX_TOKENS = 1500
QUALITY_EXTEND_MIN_TOKENS = 1200
LENGTH_SHRINK_THRESHOLD = DEFAULT_MAX_LENGTH
JSONLD_MAX_TOKENS = 800
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
    context_source: str = "index.json"
    custom_context_text: Optional[str] = None
    custom_context_len: int = 0
    custom_context_filename: Optional[str] = None
    custom_context_hash: Optional[str] = None
    custom_context_truncated: bool = False


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


def _strip_control_characters(text: str) -> str:
    allowed_whitespace = {"\n", "\t"}
    cleaned_chars: List[str] = []
    for char in text:
        if char == "\r":
            cleaned_chars.append("\n")
            continue
        if char in allowed_whitespace:
            cleaned_chars.append(" " if char == "\t" else char)
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def _collapse_blank_lines(text: str) -> str:
    lines = text.split("\n")
    collapsed: List[str] = []
    blank_pending = False
    for line in lines:
        stripped = line.strip()
        if stripped:
            collapsed.append(line.rstrip())
            blank_pending = False
            continue
        if collapsed and not blank_pending:
            collapsed.append("")
        blank_pending = True
    return "\n".join(collapsed).strip()


def _normalize_custom_context_text(raw_text: Any, *, max_chars: int = MAX_CUSTOM_CONTEXT_CHARS) -> Tuple[str, bool]:
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
    return hashlib.sha256(snippet).hexdigest()


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
        f"Раскрой и дополни раздел «{section_name}», добавь факты и примеры. "
        f"Приведи весь текст статьи к {min_target}\u2013{max_target} символам без пробелов (не меньше {min_target}). "
        "Убедись, что блок FAQ завершён и содержит 3\u20135 вопросов с развёрнутыми ответами, а все ключевые фразы "
        "использованы в точной форме. Верни полный обновлённый текст целиком, без пояснений и черновых пометок."
    )


def _build_shrink_prompt(*, min_target: int, max_target: int) -> str:
    return (
        f"Сократи повторы и второстепенные детали, приведи текст к {min_target}\u2013{max_target} символам без пробелов, "
        "сохрани исходную структуру."
    )


def _merge_extend_output(base_text: str, extension_text: str) -> Tuple[str, int]:
    base = base_text or ""
    extension = extension_text or ""
    cleaned_extension = extension.strip()
    if not cleaned_extension:
        return base, 0
    if not base:
        combined = cleaned_extension
    else:
        normalized_base = re.sub(r"\s+", " ", base).strip()
        normalized_extension = re.sub(r"\s+", " ", cleaned_extension).strip()
        base_len = len(base)
        extension_len = len(cleaned_extension)
        should_replace = False
        if normalized_extension:
            if extension_len >= max(base_len, QUALITY_EXTEND_MIN_TOKENS // 2):
                should_replace = True
            elif base_len > 0 and extension_len >= int(base_len * 0.6):
                should_replace = True
            elif normalized_base and normalized_base in normalized_extension and extension_len >= base_len:
                should_replace = True
        if should_replace:
            combined = cleaned_extension
        else:
            separator = ""
            if not base.endswith("\n") and not cleaned_extension.startswith("\n"):
                separator = "\n\n"
            combined = f"{base}{separator}{cleaned_extension}"
    delta = len(combined) - len(base)
    if delta < 0:
        delta = 0
    return combined, delta


def _resolve_extend_tokens(max_tokens: int) -> int:
    if max_tokens <= 0:
        return 1
    upper_bound = min(max_tokens, QUALITY_EXTEND_MAX_TOKENS)
    if max_tokens < QUALITY_EXTEND_MIN_TOKENS:
        return max(1, upper_bound)
    return max(QUALITY_EXTEND_MIN_TOKENS, upper_bound)


def _build_jsonld_prompt(article_text: str, requirements: PostAnalysisRequirements) -> str:
    faq_hint = requirements.faq_questions
    if isinstance(faq_hint, int) and faq_hint > 0:
        faq_line = f"Используй вопросы и ответы из блока FAQ (ровно {faq_hint} штук, без изменений)."
    else:
        faq_line = "Используй вопросы и ответы из блока FAQ (итоговый блок должен содержать 3\u20135 элементов)."
    return (
        "На основе финального текста статьи сформируй JSON-LD разметку FAQPage. "
        "Сохрани формулировки вопросов и ответов, не придумывай новые. "
        "Верни только валидный JSON без пояснений и префиксов.\n\n"
        f"{faq_line}\n\n"
        f"Текст статьи:\n{article_text.strip()}"
    )


def _build_jsonld_messages(
    article_text: str,
    requirements: PostAnalysisRequirements,
) -> List[Dict[str, str]]:
    system_message = (
        "Ты помощник SEO-редактора. Отвечай только валидным JSON-LD для FAQPage, без текста вне JSON."
    )
    user_message = _build_jsonld_prompt(article_text, requirements)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _should_force_quality_extend(
    report: Dict[str, object],
    requirements: PostAnalysisRequirements,
) -> bool:
    length_block = report.get("length") if isinstance(report, dict) else {}
    too_short = False
    if isinstance(length_block, dict):
        actual = length_block.get("chars_no_spaces")
        min_required = length_block.get("min", requirements.min_chars)
        try:
            too_short = int(actual) < int(min_required)
        except (TypeError, ValueError):
            too_short = False
    missing_keywords = report.get("missing_keywords") if isinstance(report, dict) else []
    has_missing_keywords = isinstance(missing_keywords, list) and bool(missing_keywords)
    faq_block = report.get("faq") if isinstance(report, dict) else {}
    faq_within_range = True
    if isinstance(faq_block, dict):
        faq_within_range = bool(faq_block.get("within_range", False))
    else:
        faq_count = report.get("faq_count") if isinstance(report, dict) else None
        if not isinstance(faq_count, int) or faq_count < 3 or faq_count > 5:
            faq_within_range = False
    return too_short or has_missing_keywords or not faq_within_range


def _build_quality_extend_prompt(
    report: Dict[str, object],
    requirements: PostAnalysisRequirements,
) -> str:
    min_required = requirements.min_chars
    max_required = requirements.max_chars
    missing_keywords = report.get("missing_keywords") if isinstance(report, dict) else []
    faq_block = report.get("faq") if isinstance(report, dict) else {}
    faq_count = None
    if isinstance(faq_block, dict):
        faq_count = faq_block.get("count")
    elif isinstance(report.get("faq_count"), int):
        faq_count = report.get("faq_count")

    parts: List[str] = [
        (
            f"Перепиши и расширь текст полностью, чтобы итоговый объём уверенно попал в диапазон {min_required}\u2013{max_required} символов без пробелов (не меньше {min_required})."
        )
    ]
    if isinstance(missing_keywords, list) and missing_keywords:
        highlighted = ", ".join(list(dict.fromkeys(missing_keywords)))
        parts.append(f"Добавь недостающие ключевые слова: {highlighted}.")
    else:
        parts.append("Убедись, что использованы все ключевые слова из списка.")

    faq_instruction = "Обязательно продолжить и завершить FAQ: сделай 3\u20135 вопросов с развёрнутыми ответами."
    if not isinstance(faq_count, int) or faq_count < 3:
        parts.append("Добавь недостающие вопросы в блок FAQ, чтобы было минимум три.")
    elif faq_count > 5:
        parts.append("Сократи блок FAQ до 3\u20135 вопросов.")
    parts.append(faq_instruction)
    parts.append(
        "Добавь недостающие ключевые фразы в точной форме, без изменения их написания или порядка слов."
    )
    parts.append("Верни полный обновлённый текст целиком, без пояснений и черновых пометок.")

    return " ".join(parts)


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
        adjusted_messages.append({"role": "assistant", "content": text})
        adjusted_messages.append({"role": "user", "content": prompt})
        extend_tokens = _resolve_extend_tokens(max_tokens)
        extend_result = llm_generate(
            adjusted_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=extend_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        combined_text, _ = _merge_extend_output(text, extend_result.text)
        new_result = GenerationResult(
            text=combined_text,
            model_used=extend_result.model_used,
            retry_used=True,
            fallback_used=extend_result.fallback_used,
            fallback_reason=extend_result.fallback_reason,
            api_route=extend_result.api_route,
            schema=extend_result.schema,
            metadata=extend_result.metadata,
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
    context_source: Optional[str] = None,
    custom_context_text: Optional[str] = None,
    context_filename: Optional[str] = None,
) -> GenerationContext:
    payload = deepcopy(data)
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
        tokens_est = estimate_tokens(custom_context_normalized) if custom_context_normalized else 0
        if custom_context_truncated:
            print(
                f"[orchestrate] CONTEXT: custom truncated to {MAX_CUSTOM_CONTEXT_CHARS} chars"
            )
        bundle = ContextBundle(
            items=[],
            total_tokens_est=tokens_est,
            index_missing=False,
            context_used=bool(custom_context_normalized),
            token_budget_limit=ContextBundle.token_budget_default(),
        )
    elif retrieval_k <= 0:
        reason = "source=off" if normalized_source == "off" else "k=0"
        print(f"[orchestrate] CONTEXT: disabled ({reason})")
        bundle = ContextBundle(
            items=[],
            total_tokens_est=0,
            index_missing=False,
            context_used=False,
            token_budget_limit=ContextBundle.token_budget_default(),
        )
    else:
        bundle = retrieve_context(theme_slug=theme, query=payload.get("theme", ""), k=retrieval_k)
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
        default=1500,
        dest="max_tokens",
        help="Max tokens for generation (default: 1500).",
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
    context_source: Optional[str] = None,
    context_text: Optional[str] = None,
    context_filename: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    payload = deepcopy(data)
    requested_source = context_source if context_source is not None else payload.get("context_source")
    normalized_source = str(requested_source or "index.json").strip().lower() or "index.json"
    if normalized_source == "index":
        normalized_source = "index.json"
    payload["context_source"] = normalized_source

    filename = context_filename if context_filename is not None else payload.get("context_filename")
    if isinstance(filename, str):
        filename = filename.strip() or None
    else:
        filename = None
    if filename:
        payload["context_filename"] = filename
    else:
        payload.pop("context_filename", None)

    raw_custom_context = context_text if context_text is not None else payload.get("context_text")
    if normalized_source != "custom":
        raw_custom_context = None

    effective_k = k
    if normalized_source in {"off", "custom"}:
        if normalized_source == "custom" and k > 0:
            print("[orchestrate] CONTEXT: parameter k ignored for custom source")
        effective_k = 0

    generation_context = make_generation_context(
        theme=theme,
        data=payload,
        k=effective_k,
        append_style_profile=append_style_profile,
        context_source=normalized_source,
        custom_context_text=raw_custom_context,
        context_filename=filename,
    )
    normalized_source = generation_context.context_source or normalized_source
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
    keyword_mode = str(prepared_data.get("keywords_mode") or "strict").strip().lower() or "strict"
    include_faq = bool(prepared_data.get("include_faq", True))
    faq_questions_raw = prepared_data.get("faq_questions") if include_faq else None
    faq_questions = _safe_optional_positive_int(faq_questions_raw)
    sources_values = _extract_source_values(prepared_data.get("sources"))
    include_jsonld_flag = bool(prepared_data.get("include_jsonld", False))
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
    quality_extend_used = False
    quality_extend_delta_chars = 0
    postfix_appended = False
    default_cta_used = False
    disclaimer_appended = False
    jsonld_generated = False
    jsonld_text: str = ""
    jsonld_model_used: Optional[str] = None
    jsonld_api_route: Optional[str] = None
    jsonld_metadata: Optional[Dict[str, Any]] = None
    jsonld_retry_used: Optional[bool] = None
    jsonld_fallback_used: Optional[bool] = None
    jsonld_fallback_reason: Optional[str] = None

    while True:
        post_analysis_report = analyze_post(
            article_text,
            requirements=requirements,
            model=effective_model or model_name,
            retry_count=post_retry_attempts,
            fallback_used=bool(fallback_used),
        )
        if not quality_extend_used and _should_force_quality_extend(post_analysis_report, requirements):
            extend_instruction = _build_quality_extend_prompt(post_analysis_report, requirements)
            previous_text = article_text
            active_messages = list(active_messages)
            active_messages.append({"role": "assistant", "content": previous_text})
            active_messages.append({"role": "user", "content": extend_instruction})
            extend_tokens = _resolve_extend_tokens(max_tokens_current)
            extend_result = llm_generate(
                active_messages,
                model=model_name,
                temperature=temperature,
                max_tokens=extend_tokens,
                timeout_s=timeout,
                backoff_schedule=backoff_schedule,
            )
            combined_text, delta = _merge_extend_output(previous_text, extend_result.text)
            article_text = combined_text
            effective_model = extend_result.model_used
            fallback_used = extend_result.fallback_used
            fallback_reason = extend_result.fallback_reason
            api_route = extend_result.api_route
            response_schema = extend_result.schema
            retry_used = True
            quality_extend_used = True
            quality_extend_delta_chars = delta
            llm_result = GenerationResult(
                text=article_text,
                model_used=effective_model,
                retry_used=True,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                api_route=api_route,
                schema=response_schema,
                metadata=extend_result.metadata,
            )
            continue
        if post_should_retry(post_analysis_report) and post_retry_attempts < 2:
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
            continue
        break

    quality_extend_total_chars = len(article_text)
    analysis_characters = len(article_text)
    analysis_characters_no_spaces = len(re.sub(r"\s+", "", article_text))
    if isinstance(post_analysis_report, dict):
        post_analysis_report["had_extend"] = quality_extend_used
        post_analysis_report["extend_delta_chars"] = quality_extend_delta_chars
        post_analysis_report["extend_total_chars"] = quality_extend_total_chars

    final_text = article_text
    final_text, postfix_appended, default_cta_used = _append_cta_if_needed(
        final_text,
        cta_text=cta_text,
        default_cta=cta_is_default,
    )
    final_text, disclaimer_appended = _append_disclaimer_if_requested(final_text, prepared_data)

    if include_jsonld_flag and post_analysis_report.get("meets_requirements") and article_text.strip():
        jsonld_messages = _build_jsonld_messages(article_text, requirements)
        jsonld_result = llm_generate(
            jsonld_messages,
            model=model_name,
            temperature=0.0,
            max_tokens=min(max_tokens_current, JSONLD_MAX_TOKENS),
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        jsonld_candidate = jsonld_result.text.strip()
        if jsonld_candidate:
            jsonld_generated = True
            jsonld_text = jsonld_candidate
            jsonld_model_used = jsonld_result.model_used
            jsonld_api_route = jsonld_result.api_route
            jsonld_metadata = jsonld_result.metadata
            jsonld_retry_used = jsonld_result.retry_used
            jsonld_fallback_used = jsonld_result.fallback_used
            jsonld_fallback_reason = jsonld_result.fallback_reason
            final_text = f"{final_text.rstrip()}\n\n{jsonld_text}\n"
            retry_used = retry_used or jsonld_result.retry_used

    article_text = final_text

    duration = time.time() - start_time
    context_bundle = generation_context.context_bundle
    if normalized_source == "custom":
        context_used = bool(generation_context.custom_context_text)
    else:
        context_used = bool(
            context_bundle.context_used and not context_bundle.index_missing and effective_k > 0
        )

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
        "analysis_characters": analysis_characters,
        "analysis_characters_no_spaces": analysis_characters_no_spaces,
        "words": len(article_text.split()) if article_text.strip() else 0,
        "messages_count": len(active_messages),
        "context_used": context_used,
        "context_index_missing": context_bundle.index_missing,
        "context_budget_tokens_est": context_bundle.total_tokens_est,
        "context_budget_tokens_limit": context_bundle.token_budget_limit,
        "postfix_appended": postfix_appended,
        "length_adjustment": length_adjustment,
        "quality_extend_triggered": quality_extend_used,
        "quality_extend_delta_chars": quality_extend_delta_chars,
        "quality_extend_total_chars": quality_extend_total_chars,
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
        "context_source": normalized_source,
        "include_faq": include_faq,
        "faq_questions": faq_questions,
        "include_jsonld": include_jsonld_flag,
        "jsonld_generated": jsonld_generated,
        "jsonld_text": jsonld_text,
        "jsonld_model_used": jsonld_model_used,
        "jsonld_api_route": jsonld_api_route,
        "jsonld_metadata": jsonld_metadata,
        "jsonld_retry_used": jsonld_retry_used,
        "jsonld_fallback_used": jsonld_fallback_used,
        "jsonld_fallback_reason": jsonld_fallback_reason,
        "style_profile": prepared_data.get("style_profile"),
        "post_analysis": post_analysis_report,
        "post_analysis_retry_count": post_retry_attempts,
    }

    if normalized_source == "custom":
        metadata["context_len"] = generation_context.custom_context_len
        if generation_context.custom_context_filename:
            metadata["context_filename"] = generation_context.custom_context_filename
        if generation_context.custom_context_hash:
            metadata["context_hash"] = generation_context.custom_context_hash
        metadata["context_note"] = "k_ignored"
        metadata["context_truncated"] = bool(generation_context.custom_context_truncated)
        if generation_context.custom_context_text:
            metadata["custom_context_text"] = generation_context.custom_context_text

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
    context_source: Optional[str] = None,
    context_text: Optional[str] = None,
    context_filename: Optional[str] = None,
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
        context_source=context_source,
        context_text=context_text,
        context_filename=context_filename,
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
