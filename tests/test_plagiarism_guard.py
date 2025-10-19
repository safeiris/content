# -*- coding: utf-8 -*-
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plagiarism_guard import is_too_similar


def test_detects_similar_text():
    original = "Эта статья рассказывает о тонкостях инвестиционных стратегий на развивающихся рынках."
    clip = "Текст рассказывает о тонкостях инвестиционных стратегий на развивающихся рынках с примерами."
    assert is_too_similar(original, [clip], threshold=0.3)


def test_detects_safe_text():
    original = "Сегодня мы обсуждаем рецепты домашнего хлеба и лучшие техники выпечки."
    clip = "План маркетинговой кампании включает анализ аудитории и выбор каналов продвижения."
    assert not is_too_similar(original, [clip])
