from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

from keyword_injector import LOCK_START_TEMPLATE, LOCK_END
from validators import strip_jsonld

_FAQ_START = "<!--FAQ_START-->"
_FAQ_END = "<!--FAQ_END-->"
_JSONLD_PATTERN = re.compile(r"<script\s+type=\"application/ld\+json\">.*?</script>", re.DOTALL)


class TrimValidationError(RuntimeError):
    """Raised when trimming corrupts protected content."""

    pass


@dataclass
class TrimResult:
    text: str
    removed_paragraphs: List[str]


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [part.strip("\n") for part in parts]


def _is_protected(paragraph: str, lock_tokens: Set[str]) -> bool:
    if not paragraph.strip():
        return True
    if paragraph.strip().startswith("##"):
        return True
    if "<!--LOCK_START" in paragraph:
        return True
    if LOCK_END in paragraph:
        return True
    if _FAQ_START in paragraph or _FAQ_END in paragraph:
        return True
    if paragraph.lower().startswith(("**вопрос", "**ответ")):
        return True
    if any(token in paragraph for token in lock_tokens):
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


def _extract_jsonld(text: str) -> Tuple[str, str]:
    match = _JSONLD_PATTERN.search(text)
    if not match:
        return text, ""
    jsonld_block = match.group(0)
    before = text[: match.start()].rstrip()
    after = text[match.end() :].lstrip()
    article = before
    if after:
        article = f"{article}\n\n{after}" if article else after
    return article, jsonld_block.strip()


def _paragraph_signature(paragraph: str) -> str:
    normalized = re.sub(r"\s+", " ", paragraph.strip().lower())
    return normalized


def _validate_locked_terms(text: str, terms: Sequence[str]) -> None:
    if not terms:
        return
    missing: List[str] = []
    for term in terms:
        if not term:
            continue
        lock_token = LOCK_START_TEMPLATE.format(term=term)
        pattern = re.compile(rf"{re.escape(lock_token)}.*?{re.escape(LOCK_END)}", re.DOTALL)
        if not pattern.search(text):
            missing.append(term)
    if missing:
        raise TrimValidationError(
            "После тримминга потеряны ключевые фразы: " + ", ".join(sorted(missing))
        )


def _validate_faq(text: str) -> None:
    if _FAQ_START not in text or _FAQ_END not in text:
        raise TrimValidationError("После тримминга нарушена структура блока FAQ.")
    block = text.split(_FAQ_START, 1)[1].split(_FAQ_END, 1)[0]
    pairs = re.findall(r"\*\*Вопрос\s+\d+\.\*\*", block)
    if len(pairs) != 5:
        raise TrimValidationError("FAQ должен содержать ровно 5 вопросов и ответов.")


def trim_text(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    protected_blocks: Iterable[str] | None = None,
) -> TrimResult:
    article, jsonld_block = _extract_jsonld(text)
    working = article
    removed: List[str] = []
    protected_terms = [str(term).strip() for term in (protected_blocks or []) if str(term).strip()]
    lock_tokens: Set[str] = {LOCK_START_TEMPLATE.format(term=term) for term in protected_terms}
    skip_signatures: Set[str] = set()

    def _length(current: str) -> int:
        return len(re.sub(r"\s+", "", strip_jsonld(current)))

    while _length(working) > max_chars:
        paragraphs = _split_paragraphs(working)
        candidates: List[tuple[float, int]] = []
        faq_zone = False
        for idx, paragraph in enumerate(paragraphs):
            if _FAQ_START in paragraph:
                faq_zone = True
            if _FAQ_END in paragraph:
                faq_zone = False
            signature = _paragraph_signature(paragraph)
            if faq_zone or _is_protected(paragraph, lock_tokens) or signature in skip_signatures:
                continue
            score = _score_paragraph(paragraph)
            candidates.append((score, idx))
        if not candidates:
            break
        candidates.sort()
        _, drop_idx = candidates[0]
        removed_para = paragraphs.pop(drop_idx)
        removed.append(removed_para)
        updated = _rebuild_text(paragraphs)
        if _length(updated) < min_chars:
            paragraphs.insert(drop_idx, removed.pop())
            skip_signatures.add(_paragraph_signature(paragraphs[drop_idx]))
            working = _rebuild_text(paragraphs)
            continue
        skip_signatures.add(_paragraph_signature(removed_para))
        working = updated

    working = working.rstrip() + "\n"
    _validate_locked_terms(working, protected_terms)
    _validate_faq(working)

    final_text = working
    if jsonld_block:
        final_text = f"{final_text.rstrip()}\n\n{jsonld_block}\n"
    return TrimResult(text=final_text, removed_paragraphs=removed)
