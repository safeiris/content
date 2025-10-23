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


def _ensure_lock(text: str, term: str) -> str:
    lock_start = LOCK_START_TEMPLATE.format(term=term)
    if lock_start in text:
        return text

    pattern = build_term_pattern(term)

    def _replacement(match: re.Match[str]) -> str:
        return f"{lock_start}{match.group(0)}{LOCK_END}"

    updated, count = pattern.subn(_replacement, text, count=1)
    if count:
        return updated
    return text


def _build_terms_section(terms: Sequence[str]) -> str:
    lines = [_TERMS_SECTION_HEADING, ""]
    for term in terms:
        lines.append(
            f"{term} — ключевой термин, который раскрывается в материале на практических примерах."
        )
    lines.append("")
    return "\n".join(lines)


def _insert_terms_section(text: str, terms: Sequence[str]) -> str:
    section = _build_terms_section(terms)
    if _TERMS_SECTION_HEADING in text:
        return text

    faq_anchor = "\n## FAQ"
    anchor_idx = text.find(faq_anchor)
    if anchor_idx == -1:
        return f"{text.rstrip()}\n\n{section}\n"
    return f"{text[:anchor_idx].rstrip()}\n\n{section}\n\n{text[anchor_idx:]}"


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
    missing_for_section: List[str] = []

    for term in normalized:
        if not term:
            continue

        if _contains_term(working, term):
            working = _ensure_lock(working, term)
            coverage[term] = True
            continue

        inserted = False
        working, inserted = _insert_term_into_main_section(working, term)
        if inserted and _contains_term(working, term):
            working = _ensure_lock(working, term)
            coverage[term] = True
        else:
            coverage[term] = False

        missing_for_section.append(term)

    inserted_section = False
    if missing_for_section:
        updated = _insert_terms_section(working, missing_for_section)
        inserted_section = updated != working
        working = updated

    for term in missing_for_section:
        working = _ensure_lock(working, term)
        coverage[term] = LOCK_START_TEMPLATE.format(term=term) in working

    for term in normalized:
        coverage.setdefault(term, LOCK_START_TEMPLATE.format(term=term) in working)

    locked_terms = [term for term in normalized if LOCK_START_TEMPLATE.format(term=term) in working]
    found_terms = sum(1 for term in normalized if coverage.get(term))
    total_terms = len(normalized)
    return KeywordInjectionResult(
        text=working,
        coverage=coverage,
        locked_terms=locked_terms,
        inserted_section=inserted_section,
        total_terms=total_terms,
        found_terms=found_terms,
    )
