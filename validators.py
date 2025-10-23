from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from keyword_injector import LOCK_START_TEMPLATE

_FAQ_START = "<!--FAQ_START-->"
_FAQ_END = "<!--FAQ_END-->"
_JSONLD_PATTERN = re.compile(r"<script\s+type=\"application/ld\+json\">(.*?)</script>", re.DOTALL)


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


def _jsonld_valid(text: str) -> bool:
    match = _JSONLD_PATTERN.search(text)
    if not match:
        return False
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("@type") != "FAQPage":
        return False
    entities = payload.get("mainEntity")
    if not isinstance(entities, list) or len(entities) != 5:
        return False
    for entry in entities:
        if not isinstance(entry, dict):
            return False
        if entry.get("@type") != "Question":
            return False
        answer = entry.get("acceptedAnswer")
        if not isinstance(answer, dict) or answer.get("@type") != "Answer":
            return False
        if not str(entry.get("name", "")).strip():
            return False
        if not str(answer.get("text", "")).strip():
            return False
    return True


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
    outro = str(skeleton_payload.get("outro") or "").strip()
    main = skeleton_payload.get("main")
    if not intro or not outro or not isinstance(main, list) or not main:
        return False, "Скелет не содержит обязательных полей intro/main/outro."
    for idx, item in enumerate(main):
        if not isinstance(item, str) or not item.strip():
            return False, f"Блок основной части №{idx + 1} пуст."

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
) -> ValidationResult:
    length = _length_no_spaces(text)
    skeleton_ok, skeleton_message = _skeleton_status(skeleton_payload, text)

    normalized_keywords = [str(term).strip() for term in keywords if str(term).strip()]
    missing: List[str] = []
    for term in normalized_keywords:
        lock_token = LOCK_START_TEMPLATE.format(term=term)
        if lock_token not in text:
            missing.append(term)
    keywords_ok = len(missing) == 0

    faq_pairs = _faq_pairs(text)
    faq_count = len(faq_pairs)
    jsonld_ok = _jsonld_valid(text)
    faq_ok = faq_count == 5 and jsonld_ok

    length_ok = min_chars <= length <= max_chars

    stats: Dict[str, object] = {
        "length_no_spaces": length,
        "keywords_total": len(normalized_keywords),
        "keywords_missing": missing,
        "keywords_found": len(normalized_keywords) - len(missing),
        "faq_count": faq_count,
        "jsonld_ok": jsonld_ok,
    }

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
        message = "FAQ должен содержать 5 вопросов и корректный JSON-LD."
        if not jsonld_ok:
            message = "JSON-LD FAQ недействителен или отсутствует."
        raise ValidationError("faq", message, details=stats)
    if not length_ok:
        raise ValidationError(
            "length",
            f"Объём статьи {length} зн. без пробелов, требуется {min_chars}-{max_chars}.",
            details=stats,
        )
    return result
