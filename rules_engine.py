# -*- coding: utf-8 -*-
"""Prompt assembly helpers for the content factory."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    DEFAULT_SEO_DENSITY,
    DEFAULT_STRUCTURE,
    DEFAULT_TONE,
)
from helpers import list_to_block
from keywords import format_keywords_block


# Шаблон промпта рядом с файлом
PROMPT_PATH = Path(__file__).resolve().parent / "base_prompt.txt"


STYLE_PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "sravni.ru": "экспертный, структурный, без воды, структура: введение – основная часть – FAQ – вывод",
    "tinkoff.ru": "дружелюбный и прагматичный тон, делись конкретными примерами и инструкциями",
    "banki.ru": "аналитичный и деловой тон с акцентом на выгоды и риски для читателя",
    "off": "универсальный деловой стиль без привязки к бренду",
}


@dataclass
class InputSpec:
    theme: str
    goal: str = "SEO-статья"
    keywords: List[str] = field(default_factory=list)
    tone: str = DEFAULT_TONE
    structure: List[str] = field(default_factory=lambda: list(DEFAULT_STRUCTURE))
    min_len: int = DEFAULT_MIN_LENGTH
    max_len: int = DEFAULT_MAX_LENGTH
    seo_density: int = DEFAULT_SEO_DENSITY
    audience: str = ""
    title: str = ""
    style_profile: str = "sravni.ru"
    keywords_mode: str = "soft"
    include_faq: bool = True
    faq_questions: Optional[int] = None
    include_jsonld: bool = False
    sources: List[Dict[str, str]] = field(default_factory=list)


def build_prompt(data: Dict[str, Any]) -> str:
    """Собирает системный промпт из шаблона и входных параметров."""

    length_limits = data.get("length_limits") or {}
    min_len = _safe_int(length_limits.get("min_chars"), DEFAULT_MIN_LENGTH)
    max_len = _safe_int(length_limits.get("max_chars"), DEFAULT_MAX_LENGTH)
    keywords_mode = str(data.get("keywords_mode") or "soft").strip().lower() or "soft"

    spec = InputSpec(
        theme=str(data.get("theme", "общая тема")).strip() or "общая тема",
        goal=str(data.get("goal", "SEO-статья")).strip() or "SEO-статья",
        keywords=list(data.get("keywords", [])),
        tone=str(data.get("tone", DEFAULT_TONE)).strip() or DEFAULT_TONE,
        structure=list(data.get("structure", DEFAULT_STRUCTURE)),
        min_len=min_len,
        max_len=max_len,
        seo_density=int(data.get("seo_density", DEFAULT_SEO_DENSITY)),
        audience=str(data.get("target_audience", "")).strip(),
        title=str(data.get("title", "")).strip(),
        style_profile=str(data.get("style_profile", "sravni.ru")).strip().lower() or "sravni.ru",
        keywords_mode=keywords_mode,
        include_faq=bool(data.get("include_faq", True)),
        faq_questions=_safe_int(data.get("faq_questions")),
        include_jsonld=bool(data.get("include_jsonld", False)),
        sources=_normalize_sources(data.get("sources")),
    )

    tmpl = PROMPT_PATH.read_text(encoding="utf-8")
    structure_block = list_to_block(spec.structure)
    keywords_block = format_keywords_block(spec.keywords)

    style_line = _render_style_line(spec.style_profile)
    audience_line = _render_optional_line("Целевая аудитория", spec.audience)
    title_line = _render_optional_line("Название", spec.title)
    length_line = _render_length_line(spec.min_len, spec.max_len)
    keywords_mode_line = _render_keywords_mode_line(spec.keywords_mode)
    sources_block = _render_sources_block(spec.sources)
    faq_line = _render_faq_line(spec.include_faq, spec.faq_questions)
    jsonld_line = _render_jsonld_line(spec.include_jsonld)

    prompt = tmpl.format(
        theme=spec.theme,
        goal=spec.goal,
        tone=spec.tone,
        length_line=length_line,
        style_line=style_line,
        audience_line=audience_line,
        title_line=title_line,
        structure_block=structure_block,
        keywords_block=keywords_block,
        keywords_mode_line=keywords_mode_line,
        sources_block=sources_block,
        faq_line=faq_line,
        jsonld_line=jsonld_line,
    )
    return prompt


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _render_style_line(profile: str) -> str:
    key = profile or "sravni.ru"
    description = STYLE_PROFILE_DESCRIPTIONS.get(key, STYLE_PROFILE_DESCRIPTIONS["sravni.ru"])
    if key == "off":
        return "Пиши в универсальном деловом стиле без упоминания конкретного портала."
    return f"Используй стиль портала {key}: {description}."


def _render_optional_line(label: str, value: str) -> str:
    if not value:
        return ""
    return f"{label}: {value}\n"


def _render_length_line(min_len: int, max_len: int) -> str:
    return f"{min_len}\u2013{max_len} символов без пробелов."


def _render_keywords_mode_line(mode: str) -> str:
    normalized = (mode or "soft").lower()
    labels = {
        "soft": "мягкий — допускаются синонимы и естественные формы",
        "strict": "строгий — используй точные вхождения",
        "anti_spam": "запрет переспама — не чаще двух повторов на 1000 символов",
    }
    label = labels.get(normalized, labels["soft"])
    return f"Режим ключевых слов: {label}.\n\n"


def _render_sources_block(sources: List[Dict[str, str]]) -> str:
    if not sources:
        return ""
    lines: List[str] = []
    usage_labels = {
        "quote": "цитата с атрибуцией",
        "summary": "пересказ со ссылкой",
        "inspiration": "вдохновение без ссылки",
    }
    for source in sources:
        value = str(source.get("value", "")).strip()
        if not value:
            continue
        usage = str(source.get("usage", "")).strip().lower()
        usage_label = usage_labels.get(usage, usage)
        if usage_label:
            lines.append(f"- {value} — {usage_label}")
        else:
            lines.append(f"- {value}")
    if not lines:
        return ""
    return "Если указаны источники — используй только их:\n" + "\n".join(lines) + "\n\n"


def _render_faq_line(include_faq: bool, faq_questions: Optional[int]) -> str:
    if not include_faq:
        return ""
    if faq_questions and faq_questions > 0:
        return (
            f"В конце добавь блок FAQ (часто задаваемые вопросы по теме) на {faq_questions} вопросов с ответами.\n"
        )
    return "В конце добавь блок FAQ (часто задаваемые вопросы по теме) на 3\u20135 вопросов с ответами.\n"


def _render_jsonld_line(include_jsonld: bool) -> str:
    if not include_jsonld:
        return ""
    return "Если включена опция JSON-LD — сгенерируй корректную SEO-разметку FAQPage.\n\n"


def _normalize_sources(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        usage = str(item.get("usage", "")).strip().lower()
        if not value:
            continue
        normalized.append({"value": value, "usage": usage})
    return normalized


__all__ = ["build_prompt"]

