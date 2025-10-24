import json
import uuid
from pathlib import Path

import pytest

from deterministic_pipeline import DeterministicPipeline, PipelineStep
from faq_builder import build_faq_block
from keyword_injector import LOCK_START_TEMPLATE, inject_keywords
from length_limits import compute_soft_length_bounds
from length_trimmer import trim_text
from llm_client import GenerationResult
from orchestrate import generate_article_from_payload, gather_health_status
from validators import ValidationError, strip_jsonld, validate_article

MIN_REQUIRED = 3500
MAX_REQUIRED = 6000

def test_keyword_injection_protects_terms_and_locks_all_occurrences():
    base_text = "## Основная часть\n\nОписание практик.\n\n## FAQ\n\n<!--FAQ_START-->\n<!--FAQ_END-->\n"
    result = inject_keywords(base_text, ["ключевая фраза", "дополнительный термин"])
    assert result.coverage_report == "2/2"
    assert result.missing_terms == []
    main_section = result.text.split("## FAQ", 1)[0]
    first_phrase = (
        "Дополнительно рассматривается "
        + f"{LOCK_START_TEMPLATE.format(term='ключевая фраза')}ключевая фраза<!--LOCK_END-->"
        + " через прикладные сценарии."
    )
    second_phrase = (
        "Дополнительно рассматривается "
        + f"{LOCK_START_TEMPLATE.format(term='дополнительный термин')}дополнительный термин<!--LOCK_END-->"
        + " через прикладные сценарии."
    )
    assert first_phrase in main_section
    assert second_phrase in main_section
    assert "### Разбираемся в терминах" not in result.text
    assert not result.inserted_section


def test_keyword_injection_adds_terms_inset_when_needed():
    base_text = "# Заголовок\n\nВступление.\n\n## FAQ\n\n<!--FAQ_START-->\n<!--FAQ_END-->\n"
    result = inject_keywords(base_text, ["редкий термин"])
    assert result.inserted_section is True
    assert "### Разбираемся в терминах" in result.text
    lock_token = LOCK_START_TEMPLATE.format(term="редкий термин")
    assert lock_token in result.text
    assert result.coverage_report == "1/1"


def test_faq_builder_produces_jsonld_block():
    base_text = "## FAQ\n\n<!--FAQ_START-->\n<!--FAQ_END-->\n"
    faq_result = build_faq_block(base_text=base_text, topic="Долговая нагрузка", keywords=["платёж"])
    assert faq_result.text.count("**Вопрос") == 5
    assert faq_result.jsonld.strip().startswith('<script type="application/ld+json">')
    payload = json.loads(faq_result.jsonld.split("\n", 1)[1].rsplit("\n", 1)[0])
    assert payload["@type"] == "FAQPage"
    assert len(payload["mainEntity"]) == 5


def test_trim_preserves_locked_and_faq():
    intro = " ".join(["Параграф с вводной информацией, который можно сократить." for _ in range(4)])
    removable = "Дополнительный абзац с примерами, который допустимо удалить."
    faq_lines = []
    for idx in range(1, 6):
        faq_lines.append(f"**Вопрос {idx}.** Что важно?")
        faq_lines.append("**Ответ.** Ответ с деталями.")
        faq_lines.append("")
    faq_block = "\n".join(faq_lines).strip()
    article = (
        f"## Введение\n\n{intro}\n\n"
        f"{LOCK_START_TEMPLATE.format(term='важный термин')}важный термин<!--LOCK_END-->\n\n"
        f"{removable}\n\n"
        "## FAQ\n\n<!--FAQ_START-->\n"
        f"{faq_block}\n"
        "<!--FAQ_END-->"
    )
    trimmed = trim_text(article, min_chars=200, max_chars=400)
    assert "важный термин" in trimmed.text
    assert "<!--FAQ_START-->" in trimmed.text
    assert len("".join(trimmed.text.split())) <= 400
    assert trimmed.removed_paragraphs


def test_validator_detects_missing_keyword():
    text = (
        "## Введение\n\nТекст без маркеров.\n\n## FAQ\n\n<!--FAQ_START-->\n"
        "**Вопрос 1.** Как?\n\n**Ответ.** Так.\n\n<!--FAQ_END-->\n"
        "<script type=\"application/ld+json\">\n"
        '{"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": []}'
        "\n</script>"
    )
    with pytest.raises(ValidationError) as exc:
        validate_article(text, keywords=["ключ"], min_chars=10, max_chars=1000)
    assert exc.value.group == "keywords"


def test_validator_length_ignores_jsonld():
    payload = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"Вопрос {idx}?",
                "acceptedAnswer": {"@type": "Answer", "text": f"Ответ {idx}"},
            }
            for idx in range(1, 6)
        ],
    }
    faq_block = "\n".join(
        [
            "**Вопрос 1.** Вопрос 1?",
            "**Ответ.** Ответ 1",
            "",
            "**Вопрос 2.** Вопрос 2?",
            "**Ответ.** Ответ 2",
            "",
            "**Вопрос 3.** Вопрос 3?",
            "**Ответ.** Ответ 3",
            "",
            "**Вопрос 4.** Вопрос 4?",
            "**Ответ.** Ответ 4",
            "",
            "**Вопрос 5.** Вопрос 5?",
            "**Ответ.** Ответ 5",
            "",
        ]
    )
    intro_paragraph = (
        "Содержательный абзац с фактами и цифрами, объясняющий контекст и приводящий примеры. "
        "Практические советы включают последовательность действий, промежуточные выводы и точные формулировки."
    )
    intro = "\n\n".join([intro_paragraph for _ in range(22)])
    article = (
        "## Введение\n\n"
        f"{intro}\n\n"
        f"{LOCK_START_TEMPLATE.format(term='ключ')}ключ<!--LOCK_END--> фиксирует термин.\n\n"
        "## FAQ\n\n<!--FAQ_START-->\n"
        f"{faq_block}\n"
        "<!--FAQ_END-->\n"
        "<script type=\"application/ld+json\">\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "</script>"
    )
    article_no_jsonld = strip_jsonld(article)
    base_length = len("".join(article_no_jsonld.split()))
    full_length = len("".join(article.split()))
    assert full_length > base_length
    assert MIN_REQUIRED <= base_length <= MAX_REQUIRED
    result = validate_article(
        article,
        keywords=["ключ"],
        min_chars=MIN_REQUIRED,
        max_chars=MAX_REQUIRED,
    )
    assert result.length_ok
    assert result.jsonld_ok


def _stub_llm(monkeypatch):
    base_paragraph = (
        "Абзац с анализом показателей и практическими советами для семейного бюджета. "
        "Расчёт коэффициентов сопровождаем примерами и перечнем действий."
    )
    outline = ["Введение", "Аналитика", "Решения"]
    intro_block = []
    for idx in range(4):
        intro_block.append(
            f"{base_paragraph} Введение блок {idx + 1} показывает, как сформировать картину текущей ситуации и определить безопасные пределы долга."
        )
    intro_text = "\n\n".join(intro_block)

    main_blocks = []
    for idx in range(5):
        main_blocks.append(
            f"{base_paragraph} Аналитика блок {idx + 1} фокусируется на цифрах, добавляет формулы и объясняет, как применять их на практике."
        )
    main_text = "\n\n".join(main_blocks)

    outro_parts = []
    for idx in range(3):
        outro_parts.append(
            f"{base_paragraph} Решения блок {idx + 1} переводит выводы в план действий, перечисляет контрольные даты и роли участников."
        )
    outro_text = "\n\n".join(outro_parts)

    faq_entries = [
        {
            "q": "Как определить допустимую долговую нагрузку?",
            "a": "Сравните платежи с ежемесячным доходом и удерживайте коэффициент не выше 30–35%.",
        },
        {
            "q": "Какие данные нужны для расчёта?",
            "a": "Соберите сведения по кредитам, страховым взносам и коммунальным платежам за последний год.",
        },
        {
            "q": "Что делать при превышении порога?",
            "a": "Пересмотрите график платежей, договоритесь о реструктуризации и выделите обязательные траты.",
        },
        {
            "q": "Как планировать резерв?",
            "a": "Откладывайте не менее двух ежемесячных платежей на отдельный счёт с быстрым доступом.",
        },
        {
            "q": "Какие сервисы помогают контролю?",
            "a": "Используйте банковские дашборды и напоминания календаря, чтобы отслеживать даты и суммы.",
        },
    ]

    skeleton_payload = {
        "intro": intro_text,
        "main": main_blocks[:3],
        "faq": faq_entries,
        "outro": outro_text,
        "conclusion": outro_text,
    }
    skeleton_text = json.dumps(skeleton_payload, ensure_ascii=False)
    faq_payload = {"faq": faq_entries}

    def fake_call(self, *, step, messages, max_tokens=None, **kwargs):
        if step == PipelineStep.SKELETON:
            return GenerationResult(
                text=skeleton_text,
                model_used="stub-model",
                retry_used=False,
                fallback_used=None,
                fallback_reason=None,
                api_route="chat",
                schema="none",
                metadata={"usage_output_tokens": 1024},
            )
        faq_json = json.dumps(faq_payload, ensure_ascii=False)
        return GenerationResult(
            text=faq_json,
            model_used="stub-model",
            retry_used=False,
            fallback_used=None,
            fallback_reason=None,
            api_route="chat",
            schema="json",
            metadata={"usage_output_tokens": 256},
        )

    monkeypatch.setattr("deterministic_pipeline.DeterministicPipeline._call_llm", fake_call)


def test_pipeline_produces_valid_article(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_llm(monkeypatch)
    pipeline = DeterministicPipeline(
        topic="Долговая нагрузка семьи",
        base_outline=["Введение", "Аналитика", "Решения"],
        keywords=[f"ключ {idx}" for idx in range(1, 12)],
        min_chars=3500,
        max_chars=6000,
        messages=[{"role": "system", "content": "Системный промпт"}],
        model="stub-model",
        temperature=0.3,
        max_tokens=1800,
        timeout_s=60,
    )
    state = pipeline.run()
    length_no_spaces = len("".join(strip_jsonld(state.text).split()))
    soft_min, soft_max, _, _ = compute_soft_length_bounds(MIN_REQUIRED, MAX_REQUIRED)
    assert soft_min <= length_no_spaces <= soft_max
    assert state.validation and state.validation.is_valid
    assert state.text.count("**Вопрос") == 5


def test_pipeline_resume_falls_back_to_available_checkpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_llm(monkeypatch)
    pipeline = DeterministicPipeline(
        topic="Долговая нагрузка семьи",
        base_outline=["Введение", "Основная часть", "Вывод"],
        keywords=[f"ключ {idx}" for idx in range(1, 12)],
        min_chars=3500,
        max_chars=6000,
        messages=[{"role": "system", "content": "Системный промпт"}],
        model="stub-model",
        temperature=0.3,
        max_tokens=1800,
        timeout_s=60,
    )
    pipeline._run_skeleton()
    state = pipeline.resume(PipelineStep.FAQ)
    assert state.validation and state.validation.is_valid
    faq_entries = [entry for entry in state.logs if entry.step == PipelineStep.FAQ]
    assert faq_entries and faq_entries[-1].status == "ok"


def test_generate_article_returns_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_llm(monkeypatch)
    unique_name = f"test_{uuid.uuid4().hex}.md"
    outfile = Path("artifacts") / unique_name
    data = {
        "theme": "Долговая нагрузка семьи",
        "structure": ["Введение", "Основная часть", "Вывод"],
        "keywords": [f"ключ {idx}" for idx in range(1, 12)],
        "include_jsonld": True,
        "context_source": "off",
    }
    monkeypatch.setattr(
        "orchestrate._run_health_ping",
        lambda: {"ok": True, "message": "stub", "route": "responses", "fallback_used": False},
    )
    result = generate_article_from_payload(
        theme="finance",
        data=data,
        k=0,
        context_source="off",
        outfile=str(outfile),
    )
    metadata = result["metadata"]
    assert metadata["validation"]["passed"]
    assert Path(outfile).exists()
    assert metadata["pipeline_logs"]
    # cleanup
    Path(outfile).unlink(missing_ok=True)
    Path(outfile.with_suffix(".json")).unlink(missing_ok=True)


def test_gather_health_status_handles_missing_theme(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    status = gather_health_status(theme="")
    assert not status["ok"]
    assert not status["checks"]["theme_index"]["ok"]
    llm_ping = status["checks"]["llm_ping"]
    assert llm_ping["route"] == "responses"
    assert llm_ping["fallback_used"] is False
    assert not llm_ping["ok"]
