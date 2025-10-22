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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
from length_limits import ResolvedLengthLimits, resolve_length_limits
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
QUALITY_EXTEND_MAX_TOKENS = 2800
QUALITY_EXTEND_MIN_TOKENS = 2200
KEYWORDS_ONLY_MIN_TOKENS = 600
KEYWORDS_ONLY_MAX_TOKENS = 900
FAQ_PASS_MIN_TOKENS = 900
FAQ_PASS_MAX_TOKENS = 1200
FAQ_PASS_MAX_ITERATIONS = 2
TRIM_PASS_MAX_TOKENS = 900
LENGTH_SHRINK_THRESHOLD = DEFAULT_MAX_LENGTH
JSONLD_MAX_TOKENS = 800
FULL_TEXT_MIN_CHARS = 1200
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
    jsonld_requested: bool = False
    length_limits: Optional[ResolvedLengthLimits] = None


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


def _clean_trailing_noise(text: str) -> str:
    if not text:
        return ""
    cleaned = text.rstrip()
    if not cleaned:
        return cleaned
    default_cta = _get_cta_text().rstrip()
    if default_cta and cleaned.endswith(default_cta):
        cleaned = cleaned[: -len(default_cta)].rstrip()
    return cleaned


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
        base = QUALITY_EXTEND_MAX_TOKENS
    else:
        base = min(max_tokens, QUALITY_EXTEND_MAX_TOKENS)
    candidate = max(QUALITY_EXTEND_MIN_TOKENS, base)
    return min(max(candidate, QUALITY_EXTEND_MIN_TOKENS), QUALITY_EXTEND_MAX_TOKENS)


def _normalize_keyword_for_tracking(term: str) -> str:
    normalized = re.sub(r"\s+", " ", str(term or "").lower()).strip()
    return normalized


@dataclass
class ArticleSnapshot:
    text: str
    chars: int
    chars_no_spaces: int
    keywords_usage_percent: float
    keywords_found: Set[str]
    faq_count: int
    meets_requirements: bool


def _build_snapshot(text: str, report: Dict[str, Any]) -> ArticleSnapshot:
    chars = len(text or "")
    chars_no_spaces = len(re.sub(r"\s+", "", text or ""))
    usage = 0.0
    found_keywords: Set[str] = set()
    if isinstance(report, dict):
        usage = float(report.get("keywords_usage_percent") or 0.0)
        coverage = report.get("keywords_coverage")
        if isinstance(coverage, list):
            for item in coverage:
                if not isinstance(item, dict):
                    continue
                if not item.get("found"):
                    continue
                term = str(item.get("term") or "").strip()
                if not term:
                    continue
                found_keywords.add(_normalize_keyword_for_tracking(term))
    faq_count = 0
    meets_requirements = False
    if isinstance(report, dict):
        faq_count = int(report.get("faq_count") or 0)
        meets_requirements = bool(report.get("meets_requirements"))
    return ArticleSnapshot(
        text=text or "",
        chars=chars,
        chars_no_spaces=chars_no_spaces,
        keywords_usage_percent=usage,
        keywords_found=found_keywords,
        faq_count=faq_count,
        meets_requirements=meets_requirements,
    )


def _validate_snapshot(
    before: ArticleSnapshot,
    after: ArticleSnapshot,
    *,
    pass_name: str,
    enforce_keyword_superset: bool = False,
) -> Tuple[bool, Optional[str], bool]:
    regression = False
    reason: Optional[str] = None
    keywords_regressed = False
    if before.chars_no_spaces > 0:
        threshold = before.chars_no_spaces * 0.9
        if after.chars_no_spaces < threshold:
            regression = True
            reason = "length_regression"
    if after.keywords_usage_percent + 0.05 < before.keywords_usage_percent:
        regression = True
        reason = "keywords_usage_regression"
    if after.faq_count < before.faq_count:
        regression = True
        reason = "faq_regression"
    if enforce_keyword_superset and not after.keywords_found.issuperset(before.keywords_found):
        regression = True
        keywords_regressed = True
        reason = reason or "keywords_removed"
    return (not regression, reason, keywords_regressed)


def _is_full_article(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < FULL_TEXT_MIN_CHARS:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    heading = lines[0]
    if not re.search(r"[A-Za-zА-Яа-яЁё]", heading):
        return False
    paragraph_count = sum(1 for line in lines if len(line.split()) >= 3)
    return paragraph_count >= 4


def _detect_repair_fragment(text: str, *, min_chars: int = 1500) -> Tuple[bool, str]:
    stripped = (text or "").strip()
    if not stripped:
        return True, "empty"
    if len(stripped) < min_chars:
        return True, "short_length"
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return True, "no_visible_lines"
    heading = lines[0]
    if not re.search(r"[A-Za-zА-Яа-яЁё]", heading):
        return True, "missing_heading"
    section_like = sum(1 for line in lines if len(line.split()) >= 3)
    if section_like < 3:
        return True, "insufficient_sections"
    return False, ""


def _normalize_faq_question(question: str) -> str:
    normalized = re.sub(r"[^A-Za-zА-Яа-яЁё0-9\s]", "", question or "", flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _render_faq_block(pairs: List[Tuple[str, str]]) -> str:
    lines: List[str] = ["## FAQ"]
    for index, (question, answer) in enumerate(pairs, start=1):
        question_clean = question.strip()
        answer_clean = answer.strip()
        lines.append(f"**Вопрос {index}.** {question_clean}")
        lines.append(f"**Ответ.** {answer_clean}")
        lines.append("")
    return "\n".join(line for line in lines if line is not None).strip()


def _merge_faq_patch(
    base_text: str,
    patch_text: str,
    *,
    target_pairs: int,
) -> Tuple[str, int, bool]:
    base_prefix, base_block, base_suffix = _extract_faq_block(base_text)
    _, candidate_block, _ = _extract_faq_block(patch_text)
    candidate_source = candidate_block or patch_text.strip()
    if not candidate_source:
        return base_text, 0, False
    candidate_pairs = _parse_faq_pairs(candidate_source)
    if not candidate_pairs:
        return base_text, 0, False
    existing_pairs = _parse_faq_pairs(base_block)
    order: List[str] = []
    merged: Dict[str, Tuple[str, str]] = {}
    for question, answer in existing_pairs:
        key = _normalize_faq_question(question)
        if not key or key in merged:
            continue
        merged[key] = (question, answer)
        order.append(key)
    for question, answer in candidate_pairs:
        key = _normalize_faq_question(question)
        if not key:
            continue
        if key in merged:
            merged[key] = (question, answer)
        else:
            merged[key] = (question, answer)
            order.append(key)
    selected_keys = order[:target_pairs]
    selected_pairs = [merged[key] for key in selected_keys if key in merged]
    if len(selected_pairs) < min(3, target_pairs):
        return base_text, 0, False
    rendered_block = _render_faq_block(selected_pairs)
    prefix = base_prefix.rstrip()
    suffix = base_suffix.lstrip()
    if prefix:
        combined = f"{prefix}\n\n{rendered_block}"
    else:
        combined = rendered_block
    if suffix:
        combined = f"{combined}\n\n{suffix}"
    before_set = {_normalize_faq_question(q) for q, _ in existing_pairs if _normalize_faq_question(q)}
    after_set = {_normalize_faq_question(q) for q, _ in selected_pairs if _normalize_faq_question(q)}
    added_pairs = max(0, len(after_set - before_set))
    return combined.strip(), added_pairs, True


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
    keyword_list: List[str] = []
    if isinstance(missing_keywords, list):
        keyword_list = [
            str(term).strip()
            for term in missing_keywords
            if isinstance(term, str) and str(term).strip()
        ]
    keyword_list = list(dict.fromkeys(keyword_list))
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
    parts.append("Добавь 5 вопросов FAQ, если их нет.")
    if keyword_list:
        bullet_list = "\n".join(f"- {term}" for term in keyword_list)
        parts.append(
            "Используй недостающие ключевые фразы строго в указанном виде:" "\n" + bullet_list
        )
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


def _extract_faq_block(text: str) -> Tuple[str, str, str]:
    if not text:
        return "", "", ""
    pattern = re.compile(r"(?im)(^|\n)(##?\s*)?faq\s*\n")
    match = pattern.search(text)
    if not match:
        return text, "", ""
    start = match.start()
    block_start = match.end()
    remainder = text[block_start:]
    end_match = re.search(r"\n#(?!#?\s*faq)", remainder, flags=re.IGNORECASE)
    block_end = block_start + end_match.start() if end_match else len(text)
    prefix = text[:start].rstrip()
    block = text[start:block_end].strip()
    suffix = text[block_end:].lstrip("\n")
    return prefix, block, suffix


def _parse_faq_pairs(block: str) -> List[Tuple[str, str]]:
    if not block:
        return []
    lines = block.splitlines()
    content_lines: List[str] = []
    heading_skipped = False
    for line in lines:
        if not heading_skipped and "faq" in line.lower():
            heading_skipped = True
            continue
        content_lines.append(line)
    content = "\n".join(content_lines).strip()
    if not content:
        return []
    pattern = re.compile(
        r"\*\*Вопрос[^*]*\*\*\s*(.+?)\n+\*\*Ответ\.\*\*\s*(.*?)(?=\n\*\*Вопрос|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    pairs: List[Tuple[str, str]] = []
    for match in pattern.finditer(content):
        question = re.sub(r"\s+", " ", match.group(1)).strip()
        answer = match.group(2).strip()
        pairs.append((question, answer))
    return pairs


def _faq_block_format_valid(text: str, expected_pairs: int) -> Tuple[bool, Dict[str, Any]]:
    _, block, _ = _extract_faq_block(text)
    result: Dict[str, Any] = {
        "has_block": bool(block),
        "expected_pairs": expected_pairs,
    }
    if not block:
        result["pairs_found"] = 0
        result["unique_questions"] = 0
        result["invalid_answers"] = expected_pairs
        return False, result
    pairs = _parse_faq_pairs(block)
    result["pairs_found"] = len(pairs)
    if len(pairs) != expected_pairs:
        result["unique_questions"] = len({q.lower(): q for q, _ in pairs})
        result["invalid_answers"] = expected_pairs
        return False, result
    unique_questions: Dict[str, str] = {}
    invalid_answers = 0
    for question, answer in pairs:
        normalized_question = re.sub(r"\s+", " ", question.lower()).strip()
        unique_questions[normalized_question] = question
        sentences = re.findall(r"[^.!?]+[.!?]", answer)
        sentence_count = len(sentences)
        if sentence_count < 2 or sentence_count > 5:
            invalid_answers += 1
            continue
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", answer.lower())
        frequency: Dict[str, int] = {}
        spam_detected = False
        for word in words:
            if len(word) <= 3:
                continue
            frequency[word] = frequency.get(word, 0) + 1
            if frequency[word] >= 4:
                spam_detected = True
                break
        if spam_detected:
            invalid_answers += 1
    result["unique_questions"] = len(unique_questions)
    result["invalid_answers"] = invalid_answers
    if len(unique_questions) != expected_pairs:
        return False, result
    if invalid_answers > 0:
        return False, result
    heading_present = False
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading_present = "faq" in stripped.lower()
        break
    result["has_heading"] = heading_present
    if not heading_present:
        return False, result
    return True, result


def _build_faq_only_prompt(
    article_text: str,
    report: Dict[str, object],
    requirements: PostAnalysisRequirements,
    *,
    target_pairs: int,
) -> str:
    length_block = report.get("length") if isinstance(report, dict) else {}
    min_required = length_block.get("min", requirements.min_chars)
    max_required = length_block.get("max", requirements.max_chars)
    return (
        "Ты опытный редактор. Перепиши материал, изменяя только блок FAQ. Остальные разделы оставь слово в слово.\n"
        f"Сформируй блок FAQ на ровно {target_pairs} уникальных пар «Вопрос/Ответ». Добавь недостающие Q&A до пяти и верни ПОЛНЫЙ обновлённый текст статьи.\n"
        "Требования к FAQ: заголовок «FAQ», далее по порядку пары с префиксами «**Вопрос N.**» и «**Ответ.**».\n"
        "Ответ должен содержать 2–5 информативных предложений без повторов и keyword-спама.\n"
        "Не повторяй вопросы, не изменяй структуру остальных разделов. Верни полный текст без пояснений.\n"
        f"Соблюдай итоговый диапазон {min_required}\u2013{max_required} символов без пробелов.\n\n"
        f"Текущая версия:\n{article_text.strip()}"
    )


def _build_trim_prompt(
    article_text: str,
    requirements: PostAnalysisRequirements,
    *,
    target_max: int,
) -> str:
    _, faq_block, _ = _extract_faq_block(article_text)
    faq_section = faq_block.strip()
    faq_instruction = (
        "Сохрани блок FAQ без единого изменения: вопросы, ответы, форматирование."
        if faq_section
        else ""
    )
    return (
        "Сократи материал до верхнего предела по длине, убрав воду и повторы в основных разделах."
        f" Итог должен быть ≤ {target_max} символов без пробелов.\n"
        f"{faq_instruction}\n"
        "Смысловая структура разделов должна сохраниться. Верни полный текст без пояснений.\n\n"
        f"Текущая версия:\n{article_text.strip()}"
    )


def _build_repair_prompt(
    article_text: str,
    report: Dict[str, Any],
    requirements: PostAnalysisRequirements,
) -> str:
    instructions: List[str] = [
        "Доведи материал до соответствия брифу. Работай точечно, не переписывай готовые разделы целиком.",
        "Верни полный итоговый текст статьи без пояснений и метаданных.",
    ]
    length_block = report.get("length") if isinstance(report, dict) else {}
    if isinstance(length_block, dict) and not length_block.get("within_limits", True):
        min_required = length_block.get("min", requirements.min_chars)
        max_required = length_block.get("max", requirements.max_chars)
        instructions.append(
            f"Соблюдай диапазон {min_required}\u2013{max_required} символов без пробелов, добавь недостающие факты или сократи повторы."
        )
    missing_keywords = report.get("missing_keywords") if isinstance(report, dict) else []
    if isinstance(missing_keywords, list) and missing_keywords:
        highlighted = ", ".join(list(dict.fromkeys(str(term).strip() for term in missing_keywords if str(term).strip())))
        if highlighted:
            instructions.append("Добавь недостающие ключевые слова в точной форме: " + highlighted + ".")
    faq_block = report.get("faq") if isinstance(report, dict) else {}
    if isinstance(faq_block, dict) and not faq_block.get("within_range", True):
        instructions.append("Допиши или выровняй блок FAQ до 5 уникальных вопросов с развёрнутыми ответами.")
    instructions.append("Сохрани структуру заголовков и формат FAQ с префиксами **Вопрос N.** / **Ответ.**")
    return (
        "Ты финальный редактор. Исправь только недостающие элементы без переписывания готовых частей.\n"
        + " ".join(instructions)
        + "\n\nТекущая версия:\n"
        + article_text.strip()
    )


def _build_keywords_only_prompt(missing_keywords: List[str]) -> str:
    keyword_list = [
        str(term).strip()
        for term in missing_keywords
        if isinstance(term, str) and str(term).strip()
    ]
    keyword_list = list(dict.fromkeys(keyword_list))
    if not keyword_list:
        return (
            "Проверь текст и убедись, что все ключевые слова из брифа сохранены в точной форме."
            " Верни полный текст статьи без пояснений."
        )
    bullet_list = "\n".join(f"- {term}" for term in keyword_list)
    return (
        "Аккуратно добавь недостающие ключевые фразы в точной форме, сохрани структуру и объём текста. "
        "Не сокращай и не расширяй материал, просто интегрируй ключи в подходящие абзацы. "
        "Список обязательных фраз:\n"
        f"{bullet_list}\n"
        "Верни полный обновлённый текст без пояснений и служебных пометок."
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
    if "include_jsonld" in payload:
        payload.pop("include_jsonld", None)

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
        jsonld_requested=jsonld_requested,
        length_limits=length_info,
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

    length_info: ResolvedLengthLimits
    if generation_context.length_limits is not None:
        length_info = generation_context.length_limits
    else:
        length_info = resolve_length_limits(theme, prepared_data)
        existing_limits = prepared_data.get("length_limits")
        if not isinstance(existing_limits, dict):
            existing_limits = {}
        existing_limits.update(
            {"min_chars": length_info.min_chars, "max_chars": length_info.max_chars}
        )
        prepared_data["length_limits"] = existing_limits

    min_chars = length_info.min_chars
    max_chars = length_info.max_chars
    length_sources = {"min": length_info.min_source, "max": length_info.max_source}
    input_length_limits = prepared_data.get("length_limits")
    if isinstance(input_length_limits, dict):
        min_candidate = _safe_optional_positive_int(
            input_length_limits.get("min_chars") or input_length_limits.get("min")
        )
        max_candidate = _safe_optional_positive_int(
            input_length_limits.get("max_chars") or input_length_limits.get("max")
        )
        if min_candidate is not None:
            min_chars = min_candidate
        if max_candidate is not None:
            max_chars = max_candidate
    length_sources_override = prepared_data.get("_length_limits_source")
    if isinstance(length_sources_override, dict):
        length_sources.update({
            "min": length_sources_override.get("min", length_sources.get("min")),
            "max": length_sources_override.get("max", length_sources.get("max")),
        })
    length_warnings = list(length_info.warnings)
    if length_info.swapped and not length_warnings:
        length_warnings.append(
            "Минимальный объём в брифе был больше максимального; значения переставлены местами."
        )
    source_label = f"min={length_sources['min']}, max={length_sources['max']}"
    if length_info.profile_source and "profile" in length_sources.values():
        source_label += f" (profile={length_info.profile_source})"
    print(f"[orchestrate] LENGTH LIMITS: {min_chars}\u2013{max_chars} ({source_label})")
    for note in length_warnings:
        print(f"[orchestrate] LENGTH LIMITS WARNING: {note}")

    keywords_required = [
        str(kw).strip()
        for kw in prepared_data.get("keywords", [])
        if isinstance(kw, str) and str(kw).strip()
    ]
    keyword_mode = str(prepared_data.get("keywords_mode") or "strict").strip().lower() or "strict"
    if keyword_mode != "strict":
        keyword_mode = "strict"
    include_faq = bool(prepared_data.get("include_faq", True))
    faq_questions_raw = prepared_data.get("faq_questions") if include_faq else None
    faq_questions = _safe_optional_positive_int(faq_questions_raw)
    sources_values = _extract_source_values(prepared_data.get("sources"))
    include_jsonld_flag = bool(getattr(generation_context, "jsonld_requested", False))
    requirements = PostAnalysisRequirements(
        min_chars=min_chars,
        max_chars=max_chars,
        keywords=list(keywords_required),
        keyword_mode=keyword_mode,
        faq_questions=faq_questions,
        sources=sources_values,
        style_profile=str(prepared_data.get("style_profile", "")),
        length_sources=dict(length_sources),
        jsonld_enabled=include_jsonld_flag,
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
        article_text = _clean_trailing_noise(article_text)
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
        article_text = _clean_trailing_noise(article_text)

    retry_used = retry_used or truncation_retry_used or llm_result.retry_used

    post_retry_attempts = 0
    post_analysis_report: Dict[str, object] = {}
    quality_extend_used = False
    quality_extend_delta_chars = 0
    quality_extend_passes = 0
    quality_extend_iterations: List[Dict[str, Any]] = []
    quality_extend_max_iterations = 3
    extend_incomplete = False
    keywords_only_extend_used = False
    last_missing_keywords: List[str] = []
    faq_only_passes = 0
    faq_only_iterations: List[Dict[str, Any]] = []
    trim_pass_used = False
    trim_pass_delta_chars = 0
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
    jsonld_deferred = False
    rollback_info: Dict[str, Any] = {"used": False, "reason": None, "pass": None}
    faq_added_pairs_total = 0
    keywords_regress_prevented = False
    repair_pass_fallback_used = False
    repair_pass_rollback_used = False
    repair_pass_reason: Optional[str] = None

    while True:
        article_text = _clean_trailing_noise(article_text)
        post_analysis_report = analyze_post(
            article_text,
            requirements=requirements,
            model=effective_model or model_name,
            retry_count=post_retry_attempts,
            fallback_used=bool(fallback_used),
        )
        missing_keywords_raw = (
            post_analysis_report.get("missing_keywords")
            if isinstance(post_analysis_report, dict)
            else []
        )
        missing_keywords_list = [
            str(term).strip()
            for term in missing_keywords_raw
            if isinstance(term, str) and str(term).strip()
        ]
        last_missing_keywords = list(missing_keywords_list)
        length_block = post_analysis_report.get("length") if isinstance(post_analysis_report, dict) else {}
        chars_no_spaces = None
        if isinstance(length_block, dict):
            chars_no_spaces = length_block.get("chars_no_spaces")
        try:
            length_issue = int(chars_no_spaces) < int(requirements.min_chars)
        except (TypeError, ValueError):
            length_issue = False
        faq_block = post_analysis_report.get("faq") if isinstance(post_analysis_report, dict) else {}
        faq_issue = False
        if isinstance(faq_block, dict):
            faq_issue = not bool(faq_block.get("within_range", False))
        needs_quality_extend = bool(length_issue or missing_keywords_list or faq_issue)
        if needs_quality_extend:
            if quality_extend_passes >= quality_extend_max_iterations:
                if length_issue:
                    extend_incomplete = True
                break
            extend_instruction = _build_quality_extend_prompt(post_analysis_report, requirements)
            previous_text = article_text
            previous_report = post_analysis_report
            before_snapshot = _build_snapshot(previous_text, previous_report)
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
            before_chars = len(previous_text)
            before_chars_no_spaces = len(re.sub(r"\s+", "", previous_text))
            combined_text, delta = _merge_extend_output(previous_text, extend_result.text)
            growth_detected = delta > 0 or quality_extend_passes == 0
            candidate_text = combined_text if growth_detected else previous_text
            candidate_text = _clean_trailing_noise(candidate_text)
            effective_model = extend_result.model_used
            fallback_used = extend_result.fallback_used
            fallback_reason = extend_result.fallback_reason
            api_route = extend_result.api_route
            response_schema = extend_result.schema
            retry_used = True
            quality_extend_used = True
            after_report = analyze_post(
                candidate_text,
                requirements=requirements,
                model=effective_model or model_name,
                retry_count=post_retry_attempts,
                fallback_used=bool(fallback_used),
            )
            after_snapshot = _build_snapshot(candidate_text, after_report)
            valid, rollback_reason, _ = _validate_snapshot(
                before_snapshot,
                after_snapshot,
                pass_name="quality_extend",
            )
            if not _is_full_article(candidate_text):
                valid = False
                rollback_reason = rollback_reason or "full_text_guard"
            applied = bool(valid and growth_detected)
            iteration_number = quality_extend_passes + 1
            if applied:
                article_text = candidate_text
                post_analysis_report = after_report
                after_chars = len(article_text)
                after_chars_no_spaces = len(re.sub(r"\s+", "", article_text))
                delta_chars = max(0, after_chars - before_chars)
                quality_extend_delta_chars += delta_chars
                quality_extend_passes = iteration_number
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
            else:
                article_text = previous_text
                post_analysis_report = previous_report
                after_chars = len(candidate_text)
                after_chars_no_spaces = len(re.sub(r"\s+", "", candidate_text))
                if not rollback_info["used"]:
                    rollback_info = {"used": True, "reason": rollback_reason or "no_growth", "pass": "quality_extend"}
                if not growth_detected and quality_extend_passes > 0:
                    print("[orchestrate] QUALITY EXTEND: no growth detected, stopping extend loop")
                    extend_incomplete = True
                    quality_extend_iterations.append(
                        {
                            "iteration": iteration_number,
                            "mode": "quality",
                            "max_iterations": quality_extend_max_iterations,
                            "before_chars": before_chars,
                            "before_chars_no_spaces": before_chars_no_spaces,
                            "after_chars": after_chars,
                            "after_chars_no_spaces": after_chars_no_spaces,
                            "length_issue": bool(length_issue),
                            "faq_issue": bool(faq_issue),
                            "missing_keywords": list(missing_keywords_list),
                            "applied": False,
                            "rollback_reason": rollback_reason or "no_growth",
                        }
                    )
                    break
            if applied:
                quality_extend_iterations.append(
                    {
                        "iteration": iteration_number,
                        "mode": "quality",
                        "max_iterations": quality_extend_max_iterations,
                        "before_chars": before_chars,
                        "before_chars_no_spaces": before_chars_no_spaces,
                        "after_chars": after_chars,
                        "after_chars_no_spaces": after_chars_no_spaces,
                        "length_issue": bool(length_issue),
                        "faq_issue": bool(faq_issue),
                        "missing_keywords": list(missing_keywords_list),
                        "applied": True,
                    }
                )
                continue
            extend_incomplete = True
            quality_extend_iterations.append(
                {
                    "iteration": iteration_number,
                    "mode": "quality",
                    "max_iterations": quality_extend_max_iterations,
                    "before_chars": before_chars,
                    "before_chars_no_spaces": before_chars_no_spaces,
                    "after_chars": after_chars,
                    "after_chars_no_spaces": after_chars_no_spaces,
                    "length_issue": bool(length_issue),
                    "faq_issue": bool(faq_issue),
                    "missing_keywords": list(missing_keywords_list),
                    "applied": False,
                    "rollback_reason": rollback_reason or "validation_failed",
                }
            )
            break
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

    if last_missing_keywords and not keywords_only_extend_used:
        keyword_prompt = _build_keywords_only_prompt(last_missing_keywords)
        previous_text = article_text
        previous_report = post_analysis_report
        before_snapshot = _build_snapshot(previous_text, previous_report)
        active_messages = list(active_messages)
        active_messages.append({"role": "assistant", "content": previous_text})
        active_messages.append({"role": "user", "content": keyword_prompt})
        keyword_tokens = KEYWORDS_ONLY_MAX_TOKENS if KEYWORDS_ONLY_MAX_TOKENS > 0 else max_tokens_current
        if max_tokens_current > 0:
            keyword_tokens = min(max_tokens_current, KEYWORDS_ONLY_MAX_TOKENS)
        keyword_tokens = max(KEYWORDS_ONLY_MIN_TOKENS, keyword_tokens)
        extend_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=keyword_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        before_chars = len(previous_text)
        before_chars_no_spaces = len(re.sub(r"\s+", "", previous_text))
        combined_text, _ = _merge_extend_output(previous_text, extend_result.text)
        candidate_text = _clean_trailing_noise(combined_text)
        effective_model = extend_result.model_used
        fallback_used = extend_result.fallback_used
        fallback_reason = extend_result.fallback_reason
        api_route = extend_result.api_route
        response_schema = extend_result.schema
        retry_used = True
        quality_extend_used = True
        keywords_only_extend_used = True
        after_report = analyze_post(
            candidate_text,
            requirements=requirements,
            model=effective_model or model_name,
            retry_count=post_retry_attempts,
            fallback_used=bool(fallback_used),
        )
        after_snapshot = _build_snapshot(candidate_text, after_report)
        valid, rollback_reason, keywords_regressed = _validate_snapshot(
            before_snapshot,
            after_snapshot,
            pass_name="keywords_only",
            enforce_keyword_superset=True,
        )
        if not _is_full_article(candidate_text):
            valid = False
            rollback_reason = rollback_reason or "full_text_guard"
        applied = bool(valid)
        iteration_number = quality_extend_passes + 1
        if applied:
            article_text = candidate_text
            post_analysis_report = after_report
            last_missing_keywords = [
                str(term).strip()
                for term in (post_analysis_report.get("missing_keywords") or [])
                if isinstance(term, str) and str(term).strip()
            ]
            after_chars = len(article_text)
            after_chars_no_spaces = len(re.sub(r"\s+", "", article_text))
            delta_chars = max(0, after_chars - before_chars)
            quality_extend_delta_chars += delta_chars
            quality_extend_passes = iteration_number
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
        else:
            article_text = previous_text
            post_analysis_report = previous_report
            if keywords_regressed:
                keywords_regress_prevented = True
            if not rollback_info["used"]:
                rollback_info = {"used": True, "reason": rollback_reason or "keywords_regression", "pass": "keywords_only"}
            after_chars = len(candidate_text)
            after_chars_no_spaces = len(re.sub(r"\s+", "", candidate_text))
        quality_extend_iterations.append(
            {
                "iteration": iteration_number,
                "mode": "keywords",
                "max_iterations": quality_extend_max_iterations,
                "before_chars": before_chars,
                "before_chars_no_spaces": before_chars_no_spaces,
                "after_chars": after_chars,
                "after_chars_no_spaces": after_chars_no_spaces,
                "length_issue": False,
                "faq_issue": False,
                "missing_keywords": list(last_missing_keywords),
                "applied": applied,
                "rollback_reason": None if applied else rollback_reason or "validation_failed",
            }
        )

    faq_target_pairs = requirements.faq_questions if isinstance(requirements.faq_questions, int) and requirements.faq_questions > 0 else 5
    faq_only_attempts = 0
    while faq_only_attempts < FAQ_PASS_MAX_ITERATIONS:
        faq_block = post_analysis_report.get("faq") if isinstance(post_analysis_report, dict) else {}
        required_pairs = None
        if isinstance(faq_block, dict):
            required_pairs = faq_block.get("required")
        if not isinstance(required_pairs, int) or required_pairs <= 0:
            required_pairs = faq_target_pairs
        target_pairs = max(5, required_pairs)
        format_ok, format_meta = _faq_block_format_valid(article_text, target_pairs)
        faq_count = None
        if isinstance(faq_block, dict):
            faq_count = faq_block.get("count")
        needs_pass = False
        if not isinstance(faq_count, int) or faq_count < target_pairs:
            needs_pass = True
        if not format_ok:
            needs_pass = True
        if not needs_pass:
            break
        faq_only_attempts += 1
        previous_text = article_text
        previous_report = post_analysis_report
        before_snapshot = _build_snapshot(previous_text, previous_report)
        faq_prompt = _build_faq_only_prompt(
            article_text,
            post_analysis_report,
            requirements,
            target_pairs=target_pairs,
        )
        active_messages = list(active_messages)
        active_messages.append({"role": "assistant", "content": previous_text})
        active_messages.append({"role": "user", "content": faq_prompt})
        faq_tokens = FAQ_PASS_MAX_TOKENS if FAQ_PASS_MAX_TOKENS > 0 else max_tokens_current
        if max_tokens_current > 0:
            faq_tokens = min(max_tokens_current, FAQ_PASS_MAX_TOKENS)
        faq_tokens = max(FAQ_PASS_MIN_TOKENS, faq_tokens)
        faq_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=faq_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        before_chars = len(previous_text)
        before_chars_no_spaces = len(re.sub(r"\s+", "", previous_text))
        article_candidate = faq_result.text
        if not article_candidate.strip():
            if not rollback_info["used"]:
                rollback_info = {"used": True, "reason": "empty_faq_response", "pass": "faq_only"}
            break
        candidate_text = _clean_trailing_noise(article_candidate)
        patch_applied = False
        added_pairs = 0
        if not _is_full_article(candidate_text):
            merged_text, added_pairs_candidate, patched = _merge_faq_patch(
                previous_text,
                candidate_text,
                target_pairs=target_pairs,
            )
            if patched:
                candidate_text = _clean_trailing_noise(merged_text)
                patch_applied = True
                added_pairs = added_pairs_candidate
            else:
                candidate_text = previous_text
        effective_model = faq_result.model_used
        fallback_used = faq_result.fallback_used
        fallback_reason = faq_result.fallback_reason
        api_route = faq_result.api_route
        response_schema = faq_result.schema
        retry_used = True
        after_report = analyze_post(
            candidate_text,
            requirements=requirements,
            model=effective_model or model_name,
            retry_count=post_retry_attempts,
            fallback_used=bool(fallback_used),
        )
        after_snapshot = _build_snapshot(candidate_text, after_report)
        valid, rollback_reason, _ = _validate_snapshot(
            before_snapshot,
            after_snapshot,
            pass_name="faq_only",
        )
        if candidate_text.strip() == previous_text.strip():
            valid = False
            rollback_reason = rollback_reason or "no_change"
        format_ok_after, format_meta_after = _faq_block_format_valid(candidate_text, target_pairs)
        if not format_ok_after:
            valid = False
            rollback_reason = rollback_reason or "faq_format_invalid"
        if not _is_full_article(candidate_text):
            valid = False
            rollback_reason = rollback_reason or "full_text_guard"
        after_chars = len(candidate_text)
        after_chars_no_spaces = len(re.sub(r"\s+", "", candidate_text))
        iteration_payload: Dict[str, Any] = {
            "iteration": faq_only_attempts,
            "target_pairs": target_pairs,
            "before_chars": before_chars,
            "before_chars_no_spaces": before_chars_no_spaces,
            "after_chars": after_chars,
            "after_chars_no_spaces": after_chars_no_spaces,
            "count_before": faq_count,
            "count_after": after_snapshot.faq_count,
            "format_ok_before": format_ok,
            "format_ok_after": format_ok_after,
            "format_meta_before": format_meta,
            "format_meta_after": format_meta_after,
            "patch_applied": patch_applied,
        }
        if valid:
            article_text = candidate_text
            post_analysis_report = after_report
            faq_only_passes += 1
            llm_result = GenerationResult(
                text=article_text,
                model_used=effective_model,
                retry_used=True,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                api_route=api_route,
                schema=response_schema,
                metadata=faq_result.metadata,
            )
            faq_increment = max(0, after_snapshot.faq_count - before_snapshot.faq_count)
            if added_pairs > 0:
                faq_increment = max(faq_increment, added_pairs)
            if faq_increment > 0:
                faq_added_pairs_total += faq_increment
            iteration_payload["applied"] = True
            iteration_payload["rollback_reason"] = None
            last_missing_keywords = [
                str(term).strip()
                for term in (post_analysis_report.get("missing_keywords") or [])
                if isinstance(term, str) and str(term).strip()
            ]
        else:
            article_text = previous_text
            post_analysis_report = previous_report
            iteration_payload["applied"] = False
            iteration_payload["rollback_reason"] = rollback_reason or "validation_failed"
            if not rollback_info["used"]:
                rollback_info = {
                    "used": True,
                    "reason": iteration_payload["rollback_reason"],
                    "pass": "faq_only",
                }
        faq_only_iterations.append(iteration_payload)
        if not valid:
            if faq_only_attempts >= FAQ_PASS_MAX_ITERATIONS:
                break
            continue

    length_block = post_analysis_report.get("length") if isinstance(post_analysis_report, dict) else {}
    chars_no_spaces_final = None
    if isinstance(length_block, dict):
        chars_no_spaces_final = length_block.get("chars_no_spaces")
    trim_needed = False
    try:
        if int(chars_no_spaces_final) > int(requirements.max_chars):
            trim_needed = True
    except (TypeError, ValueError):
        trim_needed = False
    if trim_needed:
        previous_text = article_text
        trim_prompt = _build_trim_prompt(
            article_text,
            requirements,
            target_max=requirements.max_chars,
        )
        active_messages = list(active_messages)
        active_messages.append({"role": "assistant", "content": previous_text})
        active_messages.append({"role": "user", "content": trim_prompt})
        trim_tokens = TRIM_PASS_MAX_TOKENS if TRIM_PASS_MAX_TOKENS > 0 else max_tokens_current
        trim_tokens = min(max_tokens_current, trim_tokens)
        trim_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=trim_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        article_candidate = trim_result.text
        if article_candidate.strip():
            before_chars = len(previous_text)
            before_chars_no_spaces = len(re.sub(r"\s+", "", previous_text))
            article_text = _clean_trailing_noise(article_candidate)
            effective_model = trim_result.model_used
            fallback_used = trim_result.fallback_used
            fallback_reason = trim_result.fallback_reason
            api_route = trim_result.api_route
            response_schema = trim_result.schema
            retry_used = True
            trim_pass_used = True
            after_chars = len(article_text)
            after_chars_no_spaces = len(re.sub(r"\s+", "", article_text))
            trim_pass_delta_chars = max(0, before_chars - after_chars)
            llm_result = GenerationResult(
                text=article_text,
                model_used=effective_model,
                retry_used=True,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                api_route=api_route,
                schema=response_schema,
                metadata=trim_result.metadata,
            )
        post_analysis_report = analyze_post(
            article_text,
            requirements=requirements,
            model=effective_model or model_name,
            retry_count=post_retry_attempts,
            fallback_used=bool(fallback_used),
        )
        last_missing_keywords = [
            str(term).strip()
            for term in (post_analysis_report.get("missing_keywords") or [])
            if isinstance(term, str) and str(term).strip()
        ]

    if isinstance(post_analysis_report, dict) and not post_analysis_report.get("meets_requirements"):
        repair_prompt = _build_repair_prompt(article_text, post_analysis_report, requirements)
        previous_text = article_text
        previous_report = post_analysis_report
        before_snapshot = _build_snapshot(previous_text, previous_report)
        active_messages = list(active_messages)
        active_messages.append({"role": "assistant", "content": previous_text})
        active_messages.append({"role": "user", "content": repair_prompt})
        repair_tokens = _resolve_extend_tokens(max_tokens_current)
        repair_result = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=repair_tokens,
            timeout_s=timeout,
            backoff_schedule=backoff_schedule,
        )
        candidate_text = _clean_trailing_noise(repair_result.text)
        effective_model = repair_result.model_used
        fallback_used = repair_result.fallback_used
        fallback_reason = repair_result.fallback_reason
        api_route = repair_result.api_route
        response_schema = repair_result.schema
        retry_used = True
        repair_pass_fallback_used = True
        fragment_triggered, fragment_reason = _detect_repair_fragment(candidate_text)
        patch_applied = False
        added_pairs_candidate = 0
        if fragment_triggered:
            repair_pass_rollback_used = True
            repair_pass_reason = "short_output_guard_triggered"
            merged_text, added_pairs_candidate, patched = _merge_faq_patch(
                previous_text,
                candidate_text,
                target_pairs=faq_target_pairs,
            )
            if patched:
                candidate_text = _clean_trailing_noise(merged_text)
                patch_applied = True
                if added_pairs_candidate > 0:
                    faq_added_pairs_total += added_pairs_candidate
            else:
                candidate_text = previous_text

        rollback_reason = None
        if candidate_text is previous_text and fragment_triggered and not patch_applied:
            after_report = previous_report
            after_snapshot = before_snapshot
            valid = False
            rollback_reason = "short_output_guard_triggered"
        else:
            after_report = analyze_post(
                candidate_text,
                requirements=requirements,
                model=effective_model or model_name,
                retry_count=post_retry_attempts,
                fallback_used=bool(fallback_used),
            )
            after_snapshot = _build_snapshot(candidate_text, after_report)
            valid, rollback_reason, _ = _validate_snapshot(
                before_snapshot,
                after_snapshot,
                pass_name="repair",
            )
            if not _is_full_article(candidate_text):
                valid = False
                rollback_reason = rollback_reason or "full_text_guard"
        if valid:
            article_text = candidate_text
            post_analysis_report = after_report
            llm_result = GenerationResult(
                text=article_text,
                model_used=effective_model,
                retry_used=True,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                api_route=api_route,
                schema=response_schema,
                metadata=repair_result.metadata,
            )
        else:
            if not rollback_info["used"]:
                rollback_info = {
                    "used": True,
                    "reason": rollback_reason or "repair_failed",
                    "pass": "repair",
                }
            article_text = previous_text
            post_analysis_report = previous_report
            repair_pass_rollback_used = True
            if repair_pass_reason is None:
                repair_pass_reason = rollback_reason or "repair_failed"

    quality_extend_total_chars = len(article_text)
    analysis_characters = len(article_text)
    analysis_characters_no_spaces = len(re.sub(r"\s+", "", article_text))
    if isinstance(post_analysis_report, dict):
        post_analysis_report["had_extend"] = quality_extend_used
        post_analysis_report["extend_delta_chars"] = quality_extend_delta_chars
        post_analysis_report["extend_total_chars"] = quality_extend_total_chars
        post_analysis_report["extend_passes"] = quality_extend_passes
        post_analysis_report["extend_iterations"] = quality_extend_iterations
        post_analysis_report["extend_incomplete"] = extend_incomplete
        post_analysis_report["faq_only_passes"] = faq_only_passes
        post_analysis_report["faq_only_iterations"] = faq_only_iterations
        post_analysis_report["faq_only_max_iterations"] = FAQ_PASS_MAX_ITERATIONS
        post_analysis_report["trim_pass_used"] = trim_pass_used
        post_analysis_report["trim_pass_delta_chars"] = trim_pass_delta_chars
        post_analysis_report["rollback"] = rollback_info
        post_analysis_report["faq_added_pairs"] = faq_added_pairs_total
        post_analysis_report["keywords_regress_prevented"] = keywords_regress_prevented
        post_analysis_report["jsonld_deferred"] = jsonld_deferred
        post_analysis_report["repair_pass_fallback"] = repair_pass_fallback_used
        post_analysis_report["repair_pass_rollback"] = repair_pass_rollback_used
        post_analysis_report["repair_pass_reason"] = repair_pass_reason

    final_text = article_text
    final_text, postfix_appended, default_cta_used = _append_cta_if_needed(
        final_text,
        cta_text=cta_text,
        default_cta=cta_is_default,
    )
    final_text, disclaimer_appended = _append_disclaimer_if_requested(final_text, prepared_data)

    if include_jsonld_flag and not post_analysis_report.get("meets_requirements"):
        jsonld_deferred = True
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
        "quality_extend_passes": quality_extend_passes,
        "quality_extend_iterations": quality_extend_iterations,
        "quality_extend_max_iterations": quality_extend_max_iterations,
        "quality_extend_keywords_used": keywords_only_extend_used,
        "extend_incomplete": extend_incomplete,
        "faq_only_passes": faq_only_passes,
        "faq_only_iterations": faq_only_iterations,
        "faq_only_max_iterations": FAQ_PASS_MAX_ITERATIONS,
        "trim_pass_used": trim_pass_used,
        "trim_pass_delta_chars": trim_pass_delta_chars,
        "rollback": rollback_info,
        "faq_added_pairs": faq_added_pairs_total,
        "keywords_regress_prevented": keywords_regress_prevented,
        "jsonld_deferred": jsonld_deferred,
        "repair_pass_fallback": repair_pass_fallback_used,
        "length_range_target": {"min": min_chars, "max": max_chars},
        "length_limits_applied": {"min": min_chars, "max": max_chars},
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

    if length_info.profile_source and "profile" in length_sources.values():
        metadata["length_limits_profile_source"] = length_info.profile_source
    if length_warnings:
        metadata["length_limits_warnings"] = length_warnings
        metadata["length_limits_warning"] = length_warnings[0]
    metadata["length_limits_source"] = length_sources

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
