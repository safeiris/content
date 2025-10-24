"""Pure prompt construction utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_DEFAULT_TONE = "экспертный, дружелюбный"
_DEFAULT_GOAL = "Сформируй структурированную SEO-статью."
_FINANCE_PROFILE_PATH = Path("profiles/finance/style_profile.md")


@lru_cache(maxsize=1)
def _load_finance_profile() -> str:
    try:
        return _FINANCE_PROFILE_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _format_keywords(keywords: Iterable[str]) -> str:
    cleaned = [kw.strip() for kw in keywords if kw and kw.strip()]
    if not cleaned:
        return ""
    return "Ключевые слова: " + ", ".join(cleaned)


def build_prompt(
    *,
    theme: str,
    tone: str,
    audience: str,
    keywords: Sequence[str],
    structure: Sequence[str],
    goal: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Build system and user messages for the generation request."""

    profile_excerpt = _load_finance_profile()
    system_parts = [
        "Ты финансовый обозреватель портала «Трубы» и пишешь понятные тексты про деньги.",
        f"Пиши в тоне: {tone or _DEFAULT_TONE}.",
        "Всегда придерживайся указанной структуры и не добавляй лишних разделов.",
    ]
    if profile_excerpt:
        system_parts.append("Следуй стилю из профиля:\n" + profile_excerpt)
    if audience:
        system_parts.append(f"Целевая аудитория: {audience}.")
    if goal:
        system_parts.append(goal)
    else:
        system_parts.append(_DEFAULT_GOAL)

    structure_lines = "\n".join(f"- {item}" for item in structure if item)
    keyword_line = _format_keywords(keywords)

    user_payload = [
        f"Тема: {theme.strip()}.",
        "Используй структуру:\n" + structure_lines,
    ]
    if keyword_line:
        user_payload.append(keyword_line)

    messages = [
        {
            "role": "user",
            "content": "\n\n".join(segment for segment in user_payload if segment),
        }
    ]
    return " \n".join(system_parts), messages
