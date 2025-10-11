# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

from config import (
    DEFAULT_TONE,
    DEFAULT_STRUCTURE,
    DEFAULT_SEO_DENSITY,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MAX_LENGTH,
)
from helpers import list_to_block, normalize_keywords

# Шаблон промпта рядом с файлом
PROMPT_PATH = Path(__file__).resolve().parent / "base_prompt.txt"

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

def build_prompt(data: Dict[str, Any]) -> str:
    """
    Собирает системный промпт из шаблона и входных параметров.
    """
    spec = InputSpec(
        theme = data.get("theme", "общая тема"),
        goal = data.get("goal", "SEO-статья"),
        keywords = data.get("keywords", []),
        tone = data.get("tone", DEFAULT_TONE),
        structure = data.get("structure", DEFAULT_STRUCTURE),
        min_len = int(data.get("min_len", DEFAULT_MIN_LENGTH)),
        max_len = int(data.get("max_len", DEFAULT_MAX_LENGTH)),
        seo_density = int(data.get("seo_density", DEFAULT_SEO_DENSITY)),
    )

    tmpl = PROMPT_PATH.read_text(encoding="utf-8")
    structure_block = list_to_block(spec.structure)
    keywords_text = normalize_keywords(spec.keywords)

    prompt = tmpl.format(
        theme=spec.theme,
        goal=spec.goal,
        tone=spec.tone,
        min_len=spec.min_len,
        max_len=spec.max_len,
        structure_block=structure_block,
        keywords=keywords_text,
    )
    return prompt

