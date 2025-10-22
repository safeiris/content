# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_client import GenerationResult  # noqa: E402
from orchestrate import (  # noqa: E402
    LENGTH_EXTEND_THRESHOLD,
    LENGTH_SHRINK_THRESHOLD,
    _append_cta_if_needed,
    _choose_section_for_extension,
    _build_quality_extend_prompt,
    _ensure_length,
    _is_truncated,
    _make_output_path,
    _normalize_custom_context_text,
    _should_force_quality_extend,
    generate_article_from_payload,
    make_generation_context,
)
from post_analysis import PostAnalysisRequirements  # noqa: E402
from config import MAX_CUSTOM_CONTEXT_CHARS  # noqa: E402


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


def _make_requirements() -> PostAnalysisRequirements:
    return PostAnalysisRequirements(
        min_chars=3500,
        max_chars=6000,
        keywords=["ключевое слово"],
        keyword_mode="soft",
        faq_questions=None,
        sources=[],
        style_profile="",
    )


def test_quality_extend_triggers_on_missing_faq():
    report = {
        "length": {"chars_no_spaces": 3600, "min": 3500, "max": 6000},
        "missing_keywords": [],
        "faq": {"within_range": False, "count": 1},
    }
    assert _should_force_quality_extend(report, _make_requirements())


def test_quality_extend_prompt_mentions_keywords_and_faq():
    report = {
        "length": {"chars_no_spaces": 2800, "min": 3500, "max": 6000},
        "missing_keywords": ["ключевое слово"],
        "faq": {"within_range": False, "count": 1},
    }
    prompt = _build_quality_extend_prompt(report, _make_requirements())
    assert "продолжить и завершить FAQ" in prompt
    assert "ключевое слово" in prompt
    assert "3500" in prompt and "6000" in prompt
    assert "Добавь недостающие ключевые фразы" in prompt


def test_ensure_length_triggers_extend(monkeypatch):
    captured = {}

    def fake_llm(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return GenerationResult(text="extended", model_used="model", retry_used=False, fallback_used=None)

    monkeypatch.setattr("orchestrate.llm_generate", fake_llm)
    short_text = "s"
    assert len(short_text) < LENGTH_EXTEND_THRESHOLD
    base_messages = [{"role": "system", "content": "base"}]
    data = {"structure": ["Введение", "Основная часть"]}

    base_result = GenerationResult(text=short_text, model_used="model", retry_used=False, fallback_used=None)

    new_result, adjustment, new_messages = _ensure_length(
        base_result,
        base_messages,
        data=data,
        model_name="model",
        temperature=0.3,
        max_tokens=100,
        timeout=5,
        backoff_schedule=[0.5],
    )

    assert new_result.text.endswith("extended")
    assert new_result.text.startswith(short_text)
    assert adjustment == "extend"
    assert len(new_messages) == len(base_messages) + 2
    assert new_messages[-2]["role"] == "assistant"
    assert new_messages[-2]["content"] == short_text
    assert "Основная часть" in captured["prompt"]


def test_ensure_length_triggers_shrink(monkeypatch):
    captured = {}

    def fake_llm(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return GenerationResult(text="shrunk", model_used="model", retry_used=False, fallback_used=None)

    monkeypatch.setattr("orchestrate.llm_generate", fake_llm)
    long_text = "x" * (LENGTH_SHRINK_THRESHOLD + 10)
    base_messages = [{"role": "system", "content": "base"}]

    base_result = GenerationResult(text=long_text, model_used="model", retry_used=False, fallback_used=None)

    new_result, adjustment, new_messages = _ensure_length(
        base_result,
        base_messages,
        data={},
        model_name="model",
        temperature=0.3,
        max_tokens=100,
        timeout=5,
        backoff_schedule=[0.5],
    )

    assert new_result.text == "shrunk"
    assert adjustment == "shrink"
    assert len(new_messages) == len(base_messages) + 1
    assert "Сократи повторы" in captured["prompt"]


def test_make_output_path_uses_belgrade_timezone(monkeypatch):
    fixed_now = datetime(2024, 1, 2, 9, 30, tzinfo=ZoneInfo("Europe/Belgrade"))
    monkeypatch.setattr("orchestrate._local_now", lambda: fixed_now)
    output_path = _make_output_path("finance", None)
    assert output_path.name == "2024-01-02_0930_finance_article.md"


def test_make_generation_context_custom_includes_message():
    raw_text = "Первый абзац\n\n\nВторой абзац"
    context = make_generation_context(
        theme="finance",
        data={"theme": "Тест"},
        k=3,
        context_source="custom",
        custom_context_text=raw_text,
        context_filename="brief.json",
    )

    custom_messages = [
        msg
        for msg in context.messages
        if msg.get("role") == "system" and str(msg.get("content", "")).startswith("CONTEXT (CUSTOM):")
    ]
    assert custom_messages, "Ожидался системный блок CONTEXT (CUSTOM)"
    assert "Первый абзац" in custom_messages[0]["content"]
    assert "Второй абзац" in custom_messages[0]["content"]
    assert context.custom_context_len == len("Первый абзац\n\nВторой абзац")
    assert context.custom_context_filename == "brief.json"
    assert context.context_source == "custom"


def test_make_generation_context_truncates_custom_text():
    long_text = "x" * (MAX_CUSTOM_CONTEXT_CHARS + 100)
    context = make_generation_context(
        theme="finance",
        data={"theme": "Тест"},
        k=1,
        context_source="custom",
        custom_context_text=long_text,
    )
    assert context.custom_context_len == MAX_CUSTOM_CONTEXT_CHARS
    assert context.custom_context_truncated


def test_normalize_custom_context_text_strips_noise():
    raw_text = "\x00Первый\r\nабзац\r\n\r\n\t\tВторой\tабзац\u0007\n\n\n"
    normalized, truncated = _normalize_custom_context_text(raw_text)

    assert "\x00" not in normalized
    assert "\u0007" not in normalized
    assert "\t" not in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Первый")
    assert normalized.rstrip().endswith("Второй абзац")
    assert not truncated


def test_generate_article_with_custom_context_metadata(monkeypatch):
    def fake_llm(messages, **kwargs):
        return GenerationResult(text="OK", model_used="model", retry_used=False, fallback_used=None)

    monkeypatch.setattr("orchestrate.llm_generate", fake_llm)
    monkeypatch.setattr("orchestrate._write_outputs", lambda path, text, metadata: {})

    result = generate_article_from_payload(
        theme="finance",
        data={"theme": "Тест"},
        k=2,
        context_source="custom",
        context_text="Параграф один\n\nПараграф два",
        context_filename="notes.txt",
    )

    metadata = result["metadata"]
    assert metadata["context_source"] == "custom"
    assert metadata["context_len"] == len("Параграф один\n\nПараграф два")
    assert metadata["context_filename"] == "notes.txt"
    assert metadata["context_note"] == "k_ignored"
    assert metadata["custom_context_text"].startswith("Параграф один")
