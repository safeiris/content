# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[1]))

from orchestrate import (  # noqa: E402
    LENGTH_EXTEND_THRESHOLD,
    LENGTH_SHRINK_THRESHOLD,
    _append_cta_if_needed,
    _choose_section_for_extension,
    _ensure_length,
    _is_truncated,
    _make_output_path,
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
        appended_text, appended, default_used = _append_cta_if_needed(
            "Описание продукта,",
            cta_text="Тестовый CTA.",
            default_cta=True,
        )
        assert appended
        assert appended_text.endswith("Тестовый CTA.")
        assert "\n\n" in appended_text
        assert default_used
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
    appended_text, appended, default_used = _append_cta_if_needed(
        text,
        cta_text="CTA",
        default_cta=True,
    )
    assert not appended
    assert appended_text == text
    assert not default_used


def test_is_truncated_detects_ellipsis():
    assert _is_truncated("Оборванный текст...")


def test_ensure_length_triggers_extend(monkeypatch):
    captured = {}

    def fake_llm(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return "extended"

    monkeypatch.setattr("orchestrate.llm_generate", fake_llm)
    short_text = "s"
    assert len(short_text) < LENGTH_EXTEND_THRESHOLD
    base_messages = [{"role": "system", "content": "base"}]
    data = {"structure": ["Введение", "Основная часть"]}

    new_text, adjustment, new_messages = _ensure_length(
        short_text,
        base_messages,
        data=data,
        model_name="model",
        temperature=0.3,
        max_tokens=100,
        timeout=5,
        backoff_schedule=[0.5],
    )

    assert new_text == "extended"
    assert adjustment == "extend"
    assert len(new_messages) == len(base_messages) + 1
    assert "Основная часть" in captured["prompt"]


def test_ensure_length_triggers_shrink(monkeypatch):
    captured = {}

    def fake_llm(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return "shrunk"

    monkeypatch.setattr("orchestrate.llm_generate", fake_llm)
    long_text = "x" * (LENGTH_SHRINK_THRESHOLD + 10)
    base_messages = [{"role": "system", "content": "base"}]

    new_text, adjustment, new_messages = _ensure_length(
        long_text,
        base_messages,
        data={},
        model_name="model",
        temperature=0.3,
        max_tokens=100,
        timeout=5,
        backoff_schedule=[0.5],
    )

    assert new_text == "shrunk"
    assert adjustment == "shrink"
    assert len(new_messages) == len(base_messages) + 1
    assert "Сократи повторы" in captured["prompt"]


def test_make_output_path_uses_belgrade_timezone(monkeypatch):
    fixed_now = datetime(2024, 1, 2, 9, 30, tzinfo=ZoneInfo("Europe/Belgrade"))
    monkeypatch.setattr("orchestrate._local_now", lambda: fixed_now)
    output_path = _make_output_path("finance", None)
    assert output_path.name == "2024-01-02_0930__finance__article.md"
