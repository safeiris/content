from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from config import DEFAULT_MAX_LENGTH, DEFAULT_MIN_LENGTH
from length_limits import compute_soft_length_bounds
from skeleton_utils import normalize_skeleton_payload
from keyword_injector import LOCK_END, LOCK_START_TEMPLATE

_FAQ_START = "<!--FAQ_START-->"
_FAQ_END = "<!--FAQ_END-->"
_JSONLD_PATTERN = re.compile(r"<script\s+type=\"application/ld\+json\">(.*?)</script>", re.DOTALL)
_FAQ_ENTRY_PATTERN = re.compile(
    r"\*\*Вопрос\s+(?P<index>\d+)\.\*\*\s*(?P<question>.+?)\s*\n\*\*Ответ\.\*\*\s*(?P<answer>.*?)(?=\n\*\*Вопрос\s+\d+\.\*\*|\Z)",
    re.DOTALL,
)


class ValidationError(RuntimeError):
    """Raised when one of the blocking validation groups fails."""

    def __init__(self, group: str, message: str, *, details: Optional[Dict[str, object]] = None) -> None:
        super().__init__(message)
        self.group = group
        self.details = details or {}


@dataclass
class ValidationResult:
    skeleton_ok: bool
    keywords_ok: bool
    faq_ok: bool
    length_ok: bool
    jsonld_ok: bool
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.skeleton_ok and self.keywords_ok and self.faq_ok and self.length_ok


def strip_jsonld(text: str) -> str:
    return _JSONLD_PATTERN.sub("", text, count=1)


def _length_no_spaces(text: str) -> int:
    return len(re.sub(r"\s+", "", strip_jsonld(text)))


def length_no_spaces(text: str) -> int:
    return _length_no_spaces(text)


def _faq_pairs(text: str) -> List[str]:
    if _FAQ_START not in text or _FAQ_END not in text:
        return []
    block = text.split(_FAQ_START, 1)[1].split(_FAQ_END, 1)[0]
    return re.findall(r"\*\*Вопрос\s+\d+\.\*\*", block)


def _parse_markdown_faq(text: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if _FAQ_START not in text or _FAQ_END not in text:
        return [], "Блок FAQ в markdown отсутствует."

    block = text.split(_FAQ_START, 1)[1].split(_FAQ_END, 1)[0]
    entries: List[Dict[str, str]] = []
    for match in _FAQ_ENTRY_PATTERN.finditer(block.strip()):
        question = match.group("question").strip()
        answer = match.group("answer").strip()
        index = int(match.group("index"))
        if not question or not answer:
            return entries, "FAQ содержит пустой вопрос или ответ."
        paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", answer) if segment.strip()]
        if not 1 <= len(paragraphs) <= 3:
            return entries, f"Ответ на вопрос '{question}' должен состоять из 1–3 абзацев."
        entries.append({"index": index, "question": question, "answer": answer})

    if len(entries) != 5:
        return entries, "FAQ должен содержать ровно 5 вопросов и ответов."

    indices = [entry["index"] for entry in entries]
    if indices != list(range(1, len(entries) + 1)):
        return entries, "Нумерация вопросов в FAQ должна идти последовательно от 1 до 5."

    return entries, None


def _parse_jsonld_entries(text: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    match = _JSONLD_PATTERN.search(text)
    if not match:
        return [], "JSON-LD FAQ недействителен или отсутствует."
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return [], "JSON-LD FAQ недействителен или отсутствует."
    if not isinstance(payload, dict) or payload.get("@type") != "FAQPage":
        return [], "JSON-LD FAQ недействителен или отсутствует."
    entities = payload.get("mainEntity")
    if not isinstance(entities, list) or len(entities) != 5:
        return [], "JSON-LD FAQ должен содержать ровно 5 вопросов и ответов."
    parsed: List[Dict[str, str]] = []
    for idx, entry in enumerate(entities, start=1):
        if not isinstance(entry, dict) or entry.get("@type") != "Question":
            return [], f"JSON-LD вопрос №{idx} имеет неверный формат."
        answer = entry.get("acceptedAnswer")
        if not isinstance(answer, dict) or answer.get("@type") != "Answer":
            return [], f"JSON-LD ответ для вопроса №{idx} имеет неверный формат."
        question = str(entry.get("name", "")).strip()
        answer_text = str(answer.get("text", "")).strip()
        if not question or not answer_text:
            return [], f"JSON-LD вопрос №{idx} содержит пустые данные."
        parsed.append({"index": idx, "question": question, "answer": answer_text})
    return parsed, None


def _skeleton_status(
    skeleton_payload: Optional[Dict[str, object]],
    text: str,
) -> Tuple[bool, Optional[str]]:
    if skeleton_payload is None:
        if "## FAQ" in text and _FAQ_START in text and _FAQ_END in text:
            return True, None
        return False, "В markdown нет заголовка FAQ и маркеров <!--FAQ_START/END-->."
    if not isinstance(skeleton_payload, dict):
        return False, "Данные скелета не получены или имеют неверный формат."

    intro = str(skeleton_payload.get("intro") or "").strip()
    main = skeleton_payload.get("main")
    faq = skeleton_payload.get("faq")
    conclusion = str(skeleton_payload.get("conclusion") or "").strip()

    missing: List[str] = []
    if not intro:
        missing.append("intro")
    if not isinstance(main, list) or not main:
        missing.append("main[]")
    if not isinstance(faq, list) or not faq:
        missing.append("faq[]")
    if not conclusion:
        missing.append("conclusion")
    if missing:
        return (
            False,
            "Отсутствуют обязательные поля после нормализации: " + ", ".join(missing),
        )

    for idx, item in enumerate(main):
        if not isinstance(item, str) or not item.strip():
            return False, f"Блок основной части №{idx + 1} пуст."

    for idx, entry in enumerate(faq, start=1):
        if not isinstance(entry, dict):
            return False, f"FAQ элемент №{idx} имеет неверный формат."
        question = str(entry.get("q") or entry.get("question") or "").strip()
        answer = str(entry.get("a") or entry.get("answer") or "").strip()
        if not question or not answer:
            return False, f"FAQ элемент №{idx} пуст."

    outline = skeleton_payload.get("outline")
    if outline and isinstance(outline, list):
        normalized_outline = [str(entry).strip() for entry in outline if str(entry).strip()]
    else:
        normalized_outline = []

    expected_main = max(1, len(normalized_outline) - 2) if normalized_outline else len(main)
    if len(main) != expected_main:
        return False, "Количество блоков основной части не совпадает с ожидаемым."
    if "## FAQ" not in text or _FAQ_START not in text or _FAQ_END not in text:
        return False, "В markdown нет заголовка FAQ и маркеров <!--FAQ_START/END-->."
    return True, None


def validate_article(
    text: str,
    *,
    keywords: Iterable[str],
    min_chars: int,
    max_chars: int,
    skeleton_payload: Optional[Dict[str, object]] = None,
    keyword_coverage_percent: Optional[float] = None,
) -> ValidationResult:
    length = _length_no_spaces(text)
    default_soft_min, default_soft_max, default_tol_below, default_tol_above = compute_soft_length_bounds(
        DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH
    )
    requested_soft_min, requested_soft_max, req_tol_below, req_tol_above = compute_soft_length_bounds(
        min_chars, max_chars
    )
    normalized_skeleton = (
        normalize_skeleton_payload(skeleton_payload)
        if skeleton_payload is not None
        else None
    )
    skeleton_ok, skeleton_message = _skeleton_status(normalized_skeleton, text)

    normalized_keywords = [str(term).strip() for term in keywords if str(term).strip()]
    missing: List[str] = []
    article = strip_jsonld(text)
    for term in normalized_keywords:
        pattern = _keyword_regex(term)
        if not pattern.search(article):
            missing.append(term)
            continue
        lock_token = LOCK_START_TEMPLATE.format(term=term)
        lock_pattern = re.compile(rf"{re.escape(lock_token)}.*?{re.escape(LOCK_END)}", re.DOTALL)
        if not lock_pattern.search(text):
            missing.append(term)
    keywords_ok = len(missing) == 0

    markdown_faq, markdown_error = _parse_markdown_faq(text)
    faq_count = len(markdown_faq)
    jsonld_entries, jsonld_error = _parse_jsonld_entries(text)
    jsonld_ok = jsonld_error is None

    faq_ok = False
    faq_error: Optional[str] = None
    mismatched_questions: List[str] = []
    if markdown_error:
        faq_error = markdown_error
    elif jsonld_error:
        faq_error = jsonld_error
    else:
        faq_ok = True
        for idx, entry in enumerate(markdown_faq):
            jsonld_entry = jsonld_entries[idx]
            if entry["question"] != jsonld_entry["question"] or entry["answer"] != jsonld_entry["answer"]:
                faq_ok = False
                mismatched_questions.append(entry["question"])
        if mismatched_questions:
            faq_error = (
                "FAQ в markdown не совпадает с JSON-LD (например, вопрос '"
                + mismatched_questions[0]
                + "')."
            )
    if not faq_ok and faq_error is None:
        faq_error = "FAQ должен содержать ровно 5 вопросов и ответов."

    coverage_percent = 100.0 if not normalized_keywords else round(
        (len(normalized_keywords) - len(missing)) / len(normalized_keywords) * 100,
        2,
    )

    length_ok = default_soft_min <= length <= default_soft_max
    requested_range_ok = requested_soft_min <= length <= requested_soft_max

    stats: Dict[str, object] = {
        "length_no_spaces": length,
        "keywords_total": len(normalized_keywords),
        "keywords_missing": missing,
        "keywords_found": len(normalized_keywords) - len(missing),
        "keywords_coverage": f"{len(normalized_keywords) - len(missing)}/{len(normalized_keywords) if normalized_keywords else 0}",
        "keywords_coverage_percent": coverage_percent,
        "keyword_coverage_expected_percent": keyword_coverage_percent,
        "faq_count": faq_count,
        "faq_jsonld_count": len(jsonld_entries),
        "faq_mismatched_questions": mismatched_questions,
        "jsonld_ok": jsonld_ok,
        "length_requested_range_ok": requested_range_ok,
        "length_required_min": DEFAULT_MIN_LENGTH,
        "length_required_max": DEFAULT_MAX_LENGTH,
        "length_soft_min": default_soft_min,
        "length_soft_max": default_soft_max,
        "length_tolerance_default_below": default_tol_below,
        "length_tolerance_default_above": default_tol_above,
        "length_requested_soft_min": requested_soft_min,
        "length_requested_soft_max": requested_soft_max,
        "length_requested_tolerance_below": req_tol_below,
        "length_requested_tolerance_above": req_tol_above,
    }

    if keyword_coverage_percent is not None and keyword_coverage_percent < 100.0:
        raise ValidationError(
            "keywords",
            (
                "Этап подстановки ключевых слов завершился с покрытием "
                f"{keyword_coverage_percent:.0f}%, требуется 100%."
            ),
            details=stats,
        )

    result = ValidationResult(
        skeleton_ok=skeleton_ok,
        keywords_ok=keywords_ok,
        faq_ok=faq_ok,
        length_ok=length_ok,
        jsonld_ok=jsonld_ok,
        stats=stats,
    )

    if not skeleton_ok:
        raise ValidationError("skeleton", skeleton_message or "Ошибка структуры статьи.", details=stats)
    if not keywords_ok:
        raise ValidationError(
            "keywords",
            "Ключевые слова покрыты не полностью.",
            details={"missing": missing, **stats},
        )
    if not faq_ok:
        message = faq_error or "FAQ должен содержать 5 вопросов и корректный JSON-LD."
        raise ValidationError("faq", message, details=stats)
    if not length_ok:
        raise ValidationError(
            "length",
            (
                f"Объём статьи {length} зн. без пробелов, требуется "
                f"{DEFAULT_MIN_LENGTH}-{DEFAULT_MAX_LENGTH} (допуск {default_soft_min}-{default_soft_max})."
            ),
            details=stats,
        )
    return result
def _keyword_regex(term: str) -> re.Pattern:
    pattern = rf"(?i)(?<!\\w){re.escape(term)}(?!\\w)"
    return re.compile(pattern)

