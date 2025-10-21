"""Lightweight quality checks for generated materials."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


_SPACE_RE = re.compile(r"\s+", re.MULTILINE)
@dataclass(frozen=True)
class PostAnalysisRequirements:
    min_chars: int
    max_chars: int
    keywords: List[str]
    keyword_mode: str
    faq_questions: Optional[int]
    sources: List[str]
    style_profile: str


def analyze(
    text: str,
    *,
    requirements: PostAnalysisRequirements,
    model: str,
    retry_count: int,
    fallback_used: bool,
) -> Dict[str, object]:
    """Compute quality diagnostics for the generated article."""

    normalized = text or ""
    chars_no_spaces = len(_SPACE_RE.sub("", normalized))
    within_limits = requirements.min_chars <= chars_no_spaces <= requirements.max_chars

    keywords_coverage: List[Dict[str, object]] = []
    sources_used: List[str] = []

    lowered = normalized.lower()
    for keyword in requirements.keywords:
        term = keyword.strip()
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        count = len(pattern.findall(normalized))
        keywords_coverage.append({"term": term, "found": count > 0, "count": count})

    for source in requirements.sources:
        candidate = source.strip()
        if not candidate:
            continue
        if candidate.lower() in lowered or candidate.lower() in lowered.replace("https://", "").replace("http://", ""):
            sources_used.append(candidate)
        else:
            domain = _extract_domain(candidate)
            if domain and domain in lowered:
                sources_used.append(candidate)

    faq_count = _estimate_faq_questions(normalized)

    report: Dict[str, object] = {
        "length": {
            "chars_no_spaces": chars_no_spaces,
            "within_limits": within_limits,
            "min": requirements.min_chars,
            "max": requirements.max_chars,
        },
        "keywords_coverage": keywords_coverage,
        "missing_keywords": [item["term"] for item in keywords_coverage if not item["found"]],
        "faq_count": faq_count,
        "sources_used": sources_used,
        "style_profile": requirements.style_profile,
        "model": model,
        "retry_count": retry_count,
        "fallback": bool(fallback_used),
    }
    return report


def should_retry(report: Dict[str, object]) -> bool:
    """Return True when the analysis indicates that a soft retry is needed."""

    length_block = report.get("length") if isinstance(report, dict) else {}
    if isinstance(length_block, dict) and not length_block.get("within_limits", True):
        return True
    missing = report.get("missing_keywords")
    if isinstance(missing, list) and len(missing) > 2:
        return True
    return False


def build_retry_instruction(
    report: Dict[str, object],
    requirements: PostAnalysisRequirements,
) -> str:
    """Generate a user-level instruction for the follow-up retry."""

    instructions: List[str] = []
    length_block = report.get("length") if isinstance(report, dict) else {}
    if isinstance(length_block, dict) and not length_block.get("within_limits", True):
        target_min = length_block.get("min", requirements.min_chars)
        target_max = length_block.get("max", requirements.max_chars)
        actual = length_block.get("chars_no_spaces")
        if isinstance(actual, int) and actual < target_min:
            instructions.append(
                f"Расширь текст до {target_min}\u2013{target_max} символов без пробелов, добавь новые факты и примеры."
            )
        elif isinstance(actual, int) and actual > target_max:
            instructions.append(
                f"Сократи текст до {target_min}\u2013{target_max} символов без пробелов, убери повторы и оставь главное."
            )
        else:
            instructions.append(
                f"Соблюдай диапазон {target_min}\u2013{target_max} символов без пробелов."
            )

    missing = report.get("missing_keywords")
    if isinstance(missing, list) and missing:
        highlighted = ", ".join(list(dict.fromkeys(missing)))
        instructions.append(
            "Добавь недостающие ключевые слова в естественном виде: " + highlighted + "."
        )

    if not instructions:
        return "Уточни ответ с учётом исходных требований."
    return " ".join(instructions)


def _extract_domain(value: str) -> str:
    cleaned = value.lower().strip()
    cleaned = cleaned.replace("https://", "").replace("http://", "")
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    return cleaned


def _estimate_faq_questions(text: str) -> int:
    lines = text.splitlines()
    question_count = 0
    in_faq = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("faq") or lowered.startswith("# faq") or lowered.startswith("## faq"):
            in_faq = True
            continue
        if in_faq and line.startswith("#") and "faq" not in lowered:
            in_faq = False
        if not in_faq:
            continue
        if line.startswith(('-', '*', '—')) or line[:2].isdigit() or line.lower().startswith("вопрос"):
            if "?" in line:
                question_count += 1
        elif line.endswith("?"):
            question_count += 1
    if question_count == 0:
        matches = re.findall(r"\n[^\n?]{0,120}\?", text)
        question_count = len(matches)
    return question_count


__all__ = [
    "PostAnalysisRequirements",
    "analyze",
    "should_retry",
    "build_retry_instruction",
]

