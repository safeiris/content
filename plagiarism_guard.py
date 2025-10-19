# -*- coding: utf-8 -*-
"""Basic anti-plagiarism utilities."""
from __future__ import annotations

from typing import Iterable, List, Set


def _ngram_set(text: str, n: int = 7) -> Set[str]:
    normalized = (text or "").lower()
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def is_too_similar(text: str, clips: List[str], threshold: float = 0.35) -> bool:
    """Return ``True`` if ``text`` is too close to any exemplar snippets."""

    if not clips:
        return False

    reference = _ngram_set(text)
    if not reference:
        return False

    for clip in clips:
        candidate = _ngram_set(clip)
        score = _jaccard(reference, candidate)
        if score >= threshold:
            return True
    return False


__all__ = ["is_too_similar"]
