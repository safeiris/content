from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from keyword_injector import LOCK_START_TEMPLATE, LOCK_END

_FAQ_START = "<!--FAQ_START-->"
_FAQ_END = "<!--FAQ_END-->"


@dataclass
class TrimResult:
    text: str
    removed_paragraphs: List[str]


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [part for part in (part.strip("\n") for part in parts)]


def _is_protected(paragraph: str) -> bool:
    if not paragraph.strip():
        return True
    if paragraph.strip().startswith("##"):
        return True
    if LOCK_START_TEMPLATE.split("{term}")[0] in paragraph:
        return True
    if LOCK_END in paragraph:
        return True
    if _FAQ_START in paragraph or _FAQ_END in paragraph:
        return True
    if paragraph.lower().startswith(("**вопрос", "**ответ")):
        return True
    return False


def _score_paragraph(paragraph: str) -> float:
    if not paragraph.strip():
        return 1e9
    lowered = paragraph.lower()
    penalties = 0.0
    if any(token in lowered for token in ["во-первых", "во-вторых", "таким образом", "в целом"]):
        penalties += 2.5
    if len(paragraph) < 220:
        penalties += 1.5
    if paragraph.endswith(":"):
        penalties += 1.0
    return penalties + len(paragraph) / 400.0


def _rebuild_text(paragraphs: Sequence[str]) -> str:
    return "\n\n".join(paragraphs).strip() + "\n"


def trim_text(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    protected_blocks: Iterable[str] | None = None,
) -> TrimResult:
    working = text
    removed: List[str] = []
    protect_patterns = list(protected_blocks or [])

    def _length(current: str) -> int:
        return len(re.sub(r"\s+", "", current))

    while _length(working) > max_chars:
        paragraphs = _split_paragraphs(working)
        candidates: List[tuple[float, int]] = []
        faq_zone = False
        for idx, paragraph in enumerate(paragraphs):
            if _FAQ_START in paragraph:
                faq_zone = True
            if _FAQ_END in paragraph:
                faq_zone = False
            if faq_zone or _is_protected(paragraph):
                continue
            if any(pattern in paragraph for pattern in protect_patterns):
                continue
            score = _score_paragraph(paragraph)
            candidates.append((score, idx))
        if not candidates:
            break
        candidates.sort()
        _, drop_idx = candidates[0]
        removed_para = paragraphs.pop(drop_idx)
        removed.append(removed_para)
        working = _rebuild_text(paragraphs)
        if _length(working) < min_chars:
            paragraphs.insert(drop_idx, removed.pop())
            working = _rebuild_text(paragraphs)
            protect_patterns.append(removed_para[:40])
            continue
    return TrimResult(text=working, removed_paragraphs=removed)
