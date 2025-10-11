# -*- coding: utf-8 -*-
from typing import List

def list_to_block(items: List[str]) -> str:
    """
    Превращает список секций в маркированный блок для промпта.
    """
    if not items:
        items = ["Введение", "Основная часть", "Вывод"]
    return "\n".join(f"- {x}" for x in items)

def normalize_keywords(keywords: List[str]) -> str:
    """
    Нормализует список ключевых слов для вставки в промпт.
    """
    if not keywords:
        return "—"
    parts = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    return ", ".join(parts) if parts else "—"

