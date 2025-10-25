# -*- coding: utf-8 -*-
"""Helpers for normalizing skeleton payloads returned by LLM."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List

from domain.faq_seeds import get_faq_seeds

LOGGER = logging.getLogger(__name__)

_CANONICAL_CONCLUSION_KEYS = ("conclusion", "outro", "ending", "final", "summary")
_DEFAULT_MAIN_PLACEHOLDER = (
    "Этот раздел будет расширен детальными рекомендациями в финальной версии статьи."
)
_FAQ_TARGET_COUNT = 5
_WORD_RE = re.compile(r"[\w-]+", flags=re.UNICODE)


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _normalize_faq_item(item: Any) -> Dict[str, str] | None:
    if isinstance(item, dict):
        question = (
            item.get("q")
            or item.get("question")
            or item.get("name")
            or item.get("title")
            or ""
        )
        answer_block = item.get("a") or item.get("answer") or item.get("text")
        if isinstance(answer_block, dict):
            answer = answer_block.get("text") or answer_block.get("content") or ""
        else:
            answer = answer_block or ""
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        question = item[0]
        answer = item[1]
    else:
        question = ""
        answer = ""
    question_text = str(question or "").strip()
    answer_text = str(answer or "").strip()
    if not question_text or not answer_text:
        return None
    return {"q": question_text, "a": answer_text}


def _deduplicate_entries(entries: Iterable[Dict[str, str]], *, seen: Iterable[str] | None = None) -> List[Dict[str, str]]:
    seen_questions = {str(question).strip().lower() for question in (seen or []) if str(question).strip()}
    unique_entries: List[Dict[str, str]] = []
    for entry in entries:
        question = str(entry.get("q", "")).strip()
        answer = str(entry.get("a", "")).strip()
        if not question or not answer:
            continue
        key = question.lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        unique_entries.append({"q": question, "a": answer})
    return unique_entries


def _extract_keywords(payload: Dict[str, Any]) -> List[str]:
    keyword_fields = (
        "normalized_keywords",
        "keywords",
        "required_keywords",
        "normalized_required_keywords",
        "preferred_keywords",
        "normalized_preferred_keywords",
    )
    collected: List[str] = []
    seen: set[str] = set()
    for field in keyword_fields:
        values = payload.get(field)
        if isinstance(values, (list, tuple, set)):
            for value in values:
                token = str(value or "").strip().lower()
                if not token or token in seen:
                    continue
                seen.add(token)
                collected.append(token)
    return collected


def _score_entry(entry: Dict[str, str], keywords: Iterable[str]) -> tuple[float, float, int]:
    text = f"{entry.get('q', '')} {entry.get('a', '')}".lower()
    coverage = 0
    for keyword in keywords:
        if keyword and keyword in text:
            coverage += 1
    words = _WORD_RE.findall(text)
    diversity = 0.0
    if words:
        diversity = len(set(words)) / max(1, len(words))
    length = len(entry.get("a", ""))
    return float(coverage), diversity, length


def _select_top_entries(entries: List[Dict[str, str]], keywords: List[str], limit: int) -> List[Dict[str, str]]:
    scored: List[tuple[float, float, int, int, Dict[str, str]]] = []
    for index, entry in enumerate(entries):
        coverage, diversity, length = _score_entry(entry, keywords)
        scored.append((coverage, diversity, length, index, entry))
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return [item[4] for item in scored[:limit]]


def _describe_keys(payload: Dict[str, Any]) -> str:
    descriptors = []
    if "intro" in payload:
        descriptors.append("intro")
    if "main" in payload:
        descriptors.append("main[]")
    if "faq" in payload:
        descriptors.append("faq[]")
    if "conclusion" in payload:
        descriptors.append("conclusion")
    return ",".join(descriptors)


def normalize_skeleton_payload(payload: Any) -> Any:
    """Return a normalized skeleton payload with canonical keys."""

    if not isinstance(payload, dict):
        return payload

    normalized: Dict[str, Any] = dict(payload)

    conclusion_value = None
    for key in _CANONICAL_CONCLUSION_KEYS:
        if key in normalized:
            value = normalized.get(key)
            if value is not None and str(value).strip():
                conclusion_value = value
                break
    if conclusion_value is not None:
        normalized["conclusion"] = conclusion_value
    for legacy_key in ("outro", "ending", "final", "summary"):
        normalized.pop(legacy_key, None)

    normalized_main = [
        str(item or "").strip() for item in _as_list(normalized.get("main")) if str(item or "").strip()
    ]
    if len(normalized_main) > 4:
        LOGGER.info("LOG:SKELETON_MAIN_TRIM normalize from=%d to=4", len(normalized_main))
        normalized_main = normalized_main[:4]
    while len(normalized_main) < 3:
        normalized_main.append(_DEFAULT_MAIN_PLACEHOLDER)
    normalized["main"] = normalized_main
    raw_faq = _as_list(normalized.get("faq"))
    faq_entries = _deduplicate_entries(filter(None, (_normalize_faq_item(item) for item in raw_faq)))

    theme = str(normalized.get("theme") or normalized.get("topic") or "").strip().lower() or "finance"
    keywords = _extract_keywords(normalized)
    seeds = _deduplicate_entries(get_faq_seeds(theme), seen=(entry["q"] for entry in faq_entries))

    combined = list(faq_entries)
    needs_fill = len(faq_entries) < _FAQ_TARGET_COUNT
    if needs_fill and seeds:
        combined.extend(seeds)

    if needs_fill and seeds:
        missing = _FAQ_TARGET_COUNT - len(faq_entries)
        LOGGER.info("LOG:SKELETON_FAQ_FILL missing=%d seed_pool=%d", missing, len(seeds))

    if len(faq_entries) > _FAQ_TARGET_COUNT:
        LOGGER.info("LOG:SKELETON_FAQ_TRIM from=%d to=%d", len(faq_entries), _FAQ_TARGET_COUNT)

    if not combined:
        combined = seeds[:]

    selected = _select_top_entries(combined, keywords, _FAQ_TARGET_COUNT)
    existing_lower = {entry.get("q", "").lower() for entry in selected if entry.get("q")}

    if len(selected) < _FAQ_TARGET_COUNT:
        supplemental = get_faq_seeds(theme)
        for entry in supplemental:
            if len(selected) >= _FAQ_TARGET_COUNT:
                break
            question = entry.get("q", "").strip()
            answer = entry.get("a", "").strip()
            if not question or not answer:
                continue
            q_lower = question.lower()
            if q_lower in existing_lower:
                continue
            existing_lower.add(q_lower)
            selected.append({"q": question, "a": answer})
        if len(selected) < _FAQ_TARGET_COUNT and supplemental:
            for entry in supplemental:
                if len(selected) >= _FAQ_TARGET_COUNT:
                    break
                question = str(entry.get("q", "")).strip()
                answer = str(entry.get("a", "")).strip()
                if not question or not answer:
                    continue
                q_lower = question.lower()
                if q_lower in existing_lower:
                    continue
                existing_lower.add(q_lower)
                selected.append({"q": question, "a": answer})

    normalized["faq"] = selected[:_FAQ_TARGET_COUNT]

    keys_descriptor = _describe_keys(normalized)
    LOGGER.info("LOG:SKELETON_NORMALIZED keys=%s", keys_descriptor)
    return normalized


__all__ = ["normalize_skeleton_payload"]
