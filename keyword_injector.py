from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOCK_START_TEMPLATE = "<!--LOCK_START term=\"{term}\"-->"
LOCK_END = "<!--LOCK_END-->"
_TERMS_SECTION_HEADING = "### Разбираемся в терминах"


def build_term_pattern(term: str) -> re.Pattern[str]:
    """Return a compiled regex that matches the exact term with word boundaries."""

    return re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")


@dataclass
class KeywordInjectionResult:
    """Result of the keyword injection step."""

    text: str
    coverage: Dict[str, bool]
    locked_terms: List[str] = field(default_factory=list)
    inserted_section: bool = False
    total_terms: int = 0
    found_terms: int = 0
    missing_terms: List[str] = field(default_factory=list)
    coverage_report: str = "0/0"


def _normalize_keywords(keywords: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw in keywords:
        term = str(raw).strip()
        if not term:
            continue
        if term in seen:
            continue
        seen.add(term)
        normalized.append(term)
    return normalized


def _contains_term(text: str, term: str) -> bool:
    pattern = build_term_pattern(term)
    return bool(pattern.search(text))


def _existing_lock_spans(text: str, term: str) -> List[Tuple[int, int]]:
    lock_start = re.escape(LOCK_START_TEMPLATE.format(term=term))
    block_pattern = re.compile(rf"{lock_start}(.*?){re.escape(LOCK_END)}", re.DOTALL)
    return [(match.start(), match.end()) for match in block_pattern.finditer(text)]


def _ensure_lock(text: str, term: str) -> str:
    pattern = build_term_pattern(term)
    lock_start = LOCK_START_TEMPLATE.format(term=term)
    locked_spans = _existing_lock_spans(text, term)
    if not pattern.search(text):
        return text

    result: List[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if any(span_start <= start < span_end for span_start, span_end in locked_spans):
            continue
        result.append(text[cursor:start])
        result.append(f"{lock_start}{match.group(0)}{LOCK_END}")
        cursor = end
    result.append(text[cursor:])
    updated = "".join(result)
    return updated


def _build_terms_inset(terms: Sequence[str]) -> str:
    items = ", ".join(terms)
    lines = [_TERMS_SECTION_HEADING, ""]
    lines.append(
        "Разбираемся в терминах: фиксируем ключевые формулировки, которые должны остаться неизменными."
    )
    lines.append(
        f"В материале используем {items} в исходном написании, чтобы автоматические проверки проходили без расхождений."
    )
    lines.append("Просим не редактировать эти формулировки при дальнейшей работе с текстом.")
    lines.append("")
    return "\n".join(lines)


def _insert_terms_inset(text: str, terms: Sequence[str]) -> Tuple[str, bool]:
    if not terms:
        return text, False
    inset = _build_terms_inset(terms)
    bounds = _find_main_section_bounds(text)
    if bounds:
        start, end = bounds
        section = text[start:end].rstrip()
        updated_section = f"{section}\n\n{inset}" if section else inset
        return f"{text[:start]}{updated_section}\n{text[end:]}" if updated_section else text, True
    faq_anchor = "\n## FAQ"
    anchor_idx = text.find(faq_anchor)
    if anchor_idx == -1:
        return f"{text.rstrip()}\n\n{inset}\n", True
    return f"{text[:anchor_idx].rstrip()}\n\n{inset}\n\n{text[anchor_idx:]}", True


def _find_main_section_bounds(text: str) -> Optional[Tuple[int, int]]:
    heading = "## Основная часть"
    start_idx = text.find(heading)
    if start_idx == -1:
        return None
    section_start = text.find("\n", start_idx)
    if section_start == -1:
        return None
    section_start += 1
    match = re.search(r"\n## ", text[section_start:])
    section_end = section_start + match.start() if match else len(text)
    return section_start, section_end


def _insert_term_into_main_section(text: str, term: str) -> Tuple[str, bool]:
    bounds = _find_main_section_bounds(text)
    if not bounds:
        return text, False

    start, end = bounds
    section = text[start:end]
    paragraphs = section.split("\n\n")
    for idx, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            appended = (
                paragraph.rstrip()
                + f" Дополнительно рассматривается {term} через прикладные сценарии."
            )
            paragraphs[idx] = appended
            new_section = "\n\n".join(paragraphs)
            return f"{text[:start]}{new_section}{text[end:]}", True

    return text, False


def inject_keywords(text: str, keywords: Iterable[str]) -> KeywordInjectionResult:
    """Insert missing keywords and protect them with lock markers."""

    normalized = _normalize_keywords(keywords)
    if not normalized:
        return KeywordInjectionResult(text=text, coverage={}, locked_terms=[])

    coverage: Dict[str, bool] = {}
    working = text
    missing_terms: List[str] = []

    for term in normalized:
        if not term:
            continue

        if _contains_term(working, term):
            working = _ensure_lock(working, term)
            continue

        working, inserted = _insert_term_into_main_section(working, term)
        if inserted and _contains_term(working, term):
            working = _ensure_lock(working, term)
            continue

        missing_terms.append(term)

    inserted_section = False
    if missing_terms:
        working, inserted_section = _insert_terms_inset(working, missing_terms)

    for term in missing_terms:
        working = _ensure_lock(working, term)

    locked_terms: List[str] = []
    missing_report: List[str] = []
    for term in normalized:
        lock_token = LOCK_START_TEMPLATE.format(term=term)
        lock_present = bool(re.search(rf"{re.escape(lock_token)}.*?{re.escape(LOCK_END)}", working, re.DOTALL))
        present = bool(build_term_pattern(term).search(working))
        ok = present and lock_present
        coverage[term] = ok
        if ok:
            locked_terms.append(term)
        else:
            missing_report.append(term)

    total_terms = len(normalized)
    found_terms = total_terms - len(missing_report)
    coverage_report = f"{found_terms}/{total_terms}" if total_terms else "0/0"
    return KeywordInjectionResult(
        text=working,
        coverage=coverage,
        locked_terms=locked_terms,
        inserted_section=inserted_section,
        total_terms=total_terms,
        found_terms=found_terms,
        missing_terms=missing_report,
        coverage_report=coverage_report,
    )
