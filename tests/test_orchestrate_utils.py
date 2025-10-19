# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from orchestrate import (  # noqa: E402
    _append_cta_if_needed,
    _choose_section_for_extension,
    _is_truncated,
)


def test_is_truncated_detects_comma():
    assert _is_truncated("Незавершённое предложение,")


def test_is_truncated_accepts_finished_sentence():
    assert not _is_truncated("Предложение завершено.")


def test_append_cta_appends_when_needed():
    env_var = "DEFAULT_CTA"
    previous = os.environ.get(env_var)
    try:
        os.environ[env_var] = "Тестовый CTA."
        appended_text, appended = _append_cta_if_needed("Описание продукта,")
        assert appended
        assert appended_text.endswith("Тестовый CTA.")
        assert "\n\n" in appended_text
    finally:
        if previous is not None:
            os.environ[env_var] = previous
        elif env_var in os.environ:
            del os.environ[env_var]


def test_choose_section_prefers_second_item():
    data = {"structure": ["Введение", "Основная часть", "Заключение"]}
    assert _choose_section_for_extension(data) == "Основная часть"


def test_append_cta_respects_complete_text():
    text = "Готовый вывод."
    appended_text, appended = _append_cta_if_needed(text)
    assert not appended
    assert appended_text == text
