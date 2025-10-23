from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from keyword_injector import LOCK_START_TEMPLATE

_FAQ_START = "<!--FAQ_START-->"
_FAQ_END = "<!--FAQ_END-->"
_JSONLD_PATTERN = re.compile(r"<script\s+type=\"application/ld\+json\">(.*?)</script>", re.DOTALL)


@dataclass
class ValidationResult:
    length_ok: bool
    keywords_ok: bool
    faq_ok: bool
    jsonld_ok: bool
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.length_ok and self.keywords_ok and self.faq_ok and self.jsonld_ok


def strip_jsonld(text: str) -> str:
    return _JSONLD_PATTERN.sub("", text, count=1)


def _length_no_spaces(text: str) -> int:
    return len(re.sub(r"\s+", "", strip_jsonld(text)))


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


def validate_article(text: str, *, keywords: Iterable[str], min_chars: int, max_chars: int) -> ValidationResult:
    length = _length_no_spaces(text)
    length_ok = min_chars <= length <= max_chars

    normalized_keywords = [str(term).strip() for term in keywords if str(term).strip()]
    keywords_ok = True
    missing: List[str] = []
    for term in normalized_keywords:
        lock_token = LOCK_START_TEMPLATE.format(term=term)
        if lock_token not in text:
            keywords_ok = False
            missing.append(term)
    faq_pairs = _faq_pairs(text)
    faq_count = len(faq_pairs)
    faq_ok = faq_count == 5
    jsonld_ok = _jsonld_valid(text)

    stats: Dict[str, object] = {
        "length_no_spaces": length,
        "keywords_total": len(normalized_keywords),
        "keywords_missing": missing,
        "keywords_found": len(normalized_keywords) - len(missing),
        "faq_count": faq_count,
    }
    return ValidationResult(
        length_ok=length_ok,
        keywords_ok=keywords_ok,
        faq_ok=faq_ok,
        jsonld_ok=jsonld_ok,
        stats=stats,
    )
