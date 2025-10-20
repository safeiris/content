# -*- coding: utf-8 -*-
from typing import List


def list_to_block(items: List[str]) -> str:
    """
    Превращает список секций в маркированный блок для промпта.
    """
    if not items:
        items = ["Введение", "Основная часть", "Вывод"]
    return "\n".join(f"- {x}" for x in items)

