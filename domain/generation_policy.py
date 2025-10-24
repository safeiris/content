"""Helpers describing generation policies and token budgeting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

AVERAGE_CHARS_PER_TOKEN = 4
JSONLD_RESERVE_TOKENS = 320
FAQ_RESERVE_TOKENS = 180


@dataclass
class TokenBudget:
    total_limit: int
    estimated_prompt_tokens: int
    available_for_body: int
    needs_segmentation: bool
    segments: List[str]


def estimate_tokens(system: str, messages: Sequence[dict]) -> int:
    """Rough token estimation without external dependencies."""

    total_chars = len(system or "")
    for message in messages:
        content = str(message.get("content", ""))
        total_chars += len(content)
    tokens = max(1, int(total_chars / AVERAGE_CHARS_PER_TOKEN))
    return tokens


def _segment_structure(structure: Sequence[str]) -> List[str]:
    segments: List[str] = []
    for item in structure:
        normalized = (item or "").strip()
        if not normalized:
            continue
        if normalized.lower().startswith("основ"):
            segments.append(f"{normalized} — часть 1")
            segments.append(f"{normalized} — часть 2")
        else:
            segments.append(normalized)
    return segments or ["Введение", "Основная часть", "FAQ", "Вывод"]


def build_token_budget(structure: Sequence[str], *, max_tokens: int, system: str, messages: Sequence[dict]) -> TokenBudget:
    estimated_prompt = estimate_tokens(system, messages)
    reserve = JSONLD_RESERVE_TOKENS + FAQ_RESERVE_TOKENS
    available = max(0, max_tokens - reserve)
    needs_segmentation = estimated_prompt > available and available > 0
    segments = list(structure)
    if needs_segmentation:
        segments = _segment_structure(structure)
    return TokenBudget(
        total_limit=max_tokens,
        estimated_prompt_tokens=estimated_prompt,
        available_for_body=available,
        needs_segmentation=needs_segmentation,
        segments=segments,
    )
