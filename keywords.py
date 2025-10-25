"""Utilities for keyword normalization and lightweight auto-suggestion."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from config import KEYWORDS_ALLOW_AUTO

KEYWORD_LIMIT = 8
KEYWORD_MAX_LENGTH = 64

_PUNCT_STRIP = "\u00ab\u00bb\u201e\u201c\u201d\"'()[]{}<>.,:;!?-–—"
_STOPWORDS = {
    "как",
    "что",
    "это",
    "для",
    "или",
    "про",
    "чтобы",
    "где",
    "когда",
    "от",
    "до",
    "при",
    "под",
    "над",
    "без",
    "через",
    "между",
    "если",
    "ли",
    "же",
    "бы",
    "еще",
    "у",
    "на",
    "по",
    "во",
    "со",
    "из",
    "об",
    "и",
    "в",
    "с",
    "но",
    "ни",
    "да",
    "нет",
}

_PHRASE_PATTERN = re.compile(r"[а-яё]{3,}(?:\s+[а-яё]{3,}){0,2}")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cleanup_keyword(raw: str, *, allow_digits: bool) -> str:
    cleaned = _normalize_space(raw.lower().replace("ё", "е"))
    cleaned = cleaned.strip(_PUNCT_STRIP)
    if not cleaned:
        return ""
    if not allow_digits and any(char.isdigit() for char in cleaned):
        return ""
    if cleaned in _STOPWORDS:
        return ""
    if len(cleaned) > KEYWORD_MAX_LENGTH:
        cleaned = cleaned[:KEYWORD_MAX_LENGTH].rstrip()
    return cleaned


def _cleanup_manual_keyword(raw: str) -> str:
    cleaned = _normalize_space(str(raw).replace("\r", " ").replace("\n", " "))
    cleaned = cleaned.strip(_PUNCT_STRIP)
    if not cleaned:
        return ""
    if len(cleaned) > KEYWORD_MAX_LENGTH:
        cleaned = cleaned[:KEYWORD_MAX_LENGTH].rstrip()
    return cleaned


def parse_manual_keywords(raw: object) -> List[str]:
    """Return a normalized list of user-provided keywords."""

    items: List[str] = []
    if isinstance(raw, str):
        parts = [part for part in re.split(r",|\n|;", raw) if part]
        items.extend(parts)
    elif isinstance(raw, (list, tuple, set)):
        items.extend(str(item) for item in raw)

    normalized: List[str] = []
    seen = set()
    for item in items:
        display = _cleanup_manual_keyword(item)
        if not display:
            continue
        normalized_key = _cleanup_keyword(item, allow_digits=True)
        if not normalized_key or normalized_key in seen:
            continue
        normalized.append(display)
        seen.add(normalized_key)
    return normalized


def _extract_phrases(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.lower().replace("ё", "е")
    return _PHRASE_PATTERN.findall(normalized)


def suggest_keywords(
    *,
    title: str,
    structure: Sequence[str],
    tone: str = "",
    style_text: str = "",
    exemplars: Sequence[dict] = (),
    limit: int = KEYWORD_LIMIT,
) -> List[str]:
    """Extract lightweight keyword suggestions from available textual cues."""

    counter: Counter[str] = Counter()

    def _consume(text: str, weight: int = 1) -> None:
        for phrase in _extract_phrases(text):
            cleaned = _cleanup_keyword(phrase, allow_digits=False)
            if not cleaned:
                continue
            counter[cleaned] += weight

    if title:
        _consume(title, weight=3)
        cleaned_title = _cleanup_keyword(title, allow_digits=False)
        if cleaned_title:
            counter[cleaned_title] += 4

    for entry in structure:
        _consume(str(entry), weight=2)

    if tone:
        _consume(tone, weight=1)

    if style_text:
        _consume(style_text, weight=1)

    for item in exemplars:
        text = str(item.get("text", ""))
        if not text:
            continue
        snippet = text[:800]
        _consume(snippet, weight=1)

    if not counter:
        return []

    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    suggestions: List[str] = []
    for phrase, _ in ranked:
        if phrase in suggestions:
            continue
        suggestions.append(phrase)
        if len(suggestions) >= limit:
            break
    return suggestions


def merge_keywords(
    manual: Iterable[str],
    auto: Iterable[str],
    *,
    limit: int = KEYWORD_LIMIT,
) -> Tuple[List[str], List[str], List[str]]:
    """Merge manual and auto-generated keywords with deduplication and limits."""

    manual_candidates = [kw for kw in manual if kw]
    auto_candidates: List[str]
    if KEYWORDS_ALLOW_AUTO:
        auto_candidates = [kw for kw in auto if kw]
    else:
        auto_candidates = []

    manual_used: List[str] = []
    auto_used: List[str] = []
    final: List[str] = []
    seen = set()

    for kw in manual_candidates:
        display = _cleanup_manual_keyword(kw)
        if not display:
            continue
        normalized = _cleanup_keyword(kw, allow_digits=True)
        if not normalized or normalized in seen:
            continue
        manual_used.append(display)
        final.append(display)
        seen.add(normalized)

    auto_budget = limit if limit > 0 else 0
    if auto_budget and auto_candidates:
        for kw in auto_candidates:
            normalized = _cleanup_keyword(kw, allow_digits=False)
            if not normalized or normalized in seen:
                continue
            auto_used.append(normalized)
            final.append(normalized)
            seen.add(normalized)
            if len(auto_used) >= auto_budget:
                break

    return manual_used, auto_used, final


def format_keywords_block(keywords: Sequence[str]) -> str:
    """Render keywords for prompt inclusion as a multi-line block."""

    items = [kw for kw in keywords if kw]
    if not items:
        return ""
    bullet_list = "\n".join(f"- {kw}" for kw in items)
    instructions = (
        "Ключевые слова (используй каждое хотя бы один раз в точной форме, без изменений):\n"
        + bullet_list
        + "\nРаспредели их по разделам, не собирай все в одном абзаце и избегай переспама.\n\n"
    )
    return instructions

