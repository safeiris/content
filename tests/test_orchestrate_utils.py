import json
import uuid
from pathlib import Path

import pytest

from config import LLM_ALLOW_FALLBACK, LLM_ROUTE
from deterministic_pipeline import DeterministicPipeline, PipelineStep
from faq_builder import build_faq_block
from keyword_injector import LOCK_START_TEMPLATE, inject_keywords
from length_limits import compute_soft_length_bounds
from length_trimmer import trim_text
from llm_client import GenerationResult
from orchestrate import generate_article_from_payload, gather_health_status
from validators import ValidationError, ValidationResult, strip_jsonld, validate_article

MIN_REQUIRED = 5200
MAX_REQUIRED = 6800

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
        validate_article(text, required_keywords=["ключ"], min_chars=10, max_chars=1000)
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
    intro = "\n\n".join([intro_paragraph for _ in range(32)])
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
        required_keywords=["ключ"],
        min_chars=MIN_REQUIRED,
        max_chars=MAX_REQUIRED,
    )
    assert result.length_ok
    assert result.jsonld_ok


def test_validator_rejects_placeholder_blocks():
    text = (
        "# Заголовок\n\n"
        "## Основной раздел\n\n"
        "Этот раздел будет дополнен после завершения генерации статьи.\n\n"
        "## FAQ\n\n"
        "<!--FAQ_START-->\n"
        "<!--FAQ_END-->\n"
    )

    with pytest.raises(ValidationError) as exc:
        validate_article(
            text,
            required_keywords=[],
            min_chars=100,
            max_chars=1200,
            faq_expected=0,
        )

    assert exc.value.group == "skeleton"


def _stub_llm(monkeypatch):
    base = (
        "Абзац с анализом показателей и практическими советами для семейного бюджета. "
        "Расчёт коэффициентов сопровождаем примерами и перечнем действий."
    )
    segments = [
        "## Введение",
        *[f"{base} Введение блок {idx}." for idx in range(1, 9)],
        "## Аналитика",
        *[f"{base} Аналитика блок {idx}." for idx in range(1, 9)],
        "## Решения",
        *[f"{base} Решения блок {idx}." for idx in range(1, 9)],
        "## FAQ",
        "<!--FAQ_START-->",
    ]
    for idx in range(1, 6):
        segments.append(f"**Вопрос {idx}.** Как действовать?")
        segments.append("**Ответ.** Соберите данные, проанализируйте метрики и сделайте выводы.")
        segments.append("")
    segments.append("<!--FAQ_END-->")
    draft_text = "\n\n".join(segments)
    refined_text = draft_text.replace("Соберите", "Соберите, систематизируйте")
    expansion_block = (
        "### Дополнительные действия\n\n"
        "Опишите, как распределить бюджет по категориям, учесть сезонные расходы и сформировать резерв."
        " Добавьте рекомендации по пересмотру планов при изменении дохода."
        "\n\n"
        "### Практические сценарии\n\n"
        "Разберите два примера семей с разным уровнем дохода: какие инструменты они используют для контроля долговой нагрузки,"
        " как перестраивают приоритеты и чего избегают. Укажите ожидаемый горизонт улучшений и метрики контроля."
    )

    def _strip_faq_block(text: str) -> str:
        if "<!--FAQ_START-->" in text:
            text = text.split("<!--FAQ_START-->", 1)[0]
        if "## FAQ" in text:
            text = text.split("## FAQ", 1)[0]
        return text.strip()

    def fake_validate(self, text):
        return ValidationResult(True, True, True, True, True, stats={"chars": len(text)})

    def fake_call(
        self,
        *,
        step,
        messages,
        max_tokens=None,
        allow_incomplete=False,
        previous_response_id=None,
        responses_format=None,
    ):
        if step is PipelineStep.SKELETON:
            if messages and "Черновик вышел" in messages[-1]["content"]:
                body_text = expansion_block
                usage = 320
            else:
                body_text = _strip_faq_block(draft_text)
                usage = 1800
            intro_text = body_text.split("\n\n", 1)[0].strip()
            if not intro_text:
                intro_text = "Вводный абзац"
            payload = {
                "intro": intro_text,
                "main_headers": [
                    "Введение",
                    "Аналитика",
                    "Решения",
                ],
                "sections": [
                    {"title": "Введение", "body": body_text},
                    {"title": "Аналитика", "body": body_text},
                    {"title": "Решения", "body": body_text},
                ],
                "faq": [
                    {
                        "q": f"Вопрос {idx}",
                        "a": f"Ответ {idx}. Подробное пояснение с примерами и выводами.",
                    }
                    for idx in range(1, 6)
                ],
                "conclusion": body_text,
                "conclusion_heading": "Вывод",
            }
            return GenerationResult(
                text=json.dumps(payload, ensure_ascii=False),
                model_used="stub-model",
                retry_used=False,
                fallback_used=None,
                metadata={"usage_output_tokens": usage},
            )
        injected_text = refined_text
        if messages:
            content = messages[-1].get("content", "")
            marker = "Текст:\n\n"
            if marker in content:
                injected_text = content.split(marker, 1)[1].strip()
                injected_text = injected_text.replace("Соберите", "Соберите, систематизируйте")
        return GenerationResult(
            text=injected_text,
            model_used="stub-model",
            retry_used=False,
            fallback_used=None,
            metadata={"usage_output_tokens": 420},
        )

    monkeypatch.setattr("deterministic_pipeline.DeterministicPipeline._call_llm", fake_call)
    monkeypatch.setattr("deterministic_pipeline.DeterministicPipeline._validate", fake_validate)
    monkeypatch.setattr(
        "deterministic_pipeline.validate_article",
        lambda *args, **kwargs: ValidationResult(True, True, True, True, True),
    )


def test_pipeline_produces_valid_article(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_llm(monkeypatch)
    pipeline = DeterministicPipeline(
        topic="Долговая нагрузка семьи",
        base_outline=["Введение", "Аналитика", "Решения"],
        required_keywords=[],
        min_chars=MIN_REQUIRED,
        max_chars=MAX_REQUIRED,
        messages=[{"role": "system", "content": "Системный промпт"}],
        model="stub-model",
        max_tokens=2600,
        timeout_s=60,
        faq_questions=5,
    )
    state = pipeline.run()
    length_no_spaces = len("".join(strip_jsonld(state.text).split()))
    soft_min, soft_max, _, _ = compute_soft_length_bounds(MIN_REQUIRED, MAX_REQUIRED)
    assert soft_min <= length_no_spaces <= soft_max
    assert state.validation and state.validation.is_valid
    assert state.text.count("**Вопрос") == 5




def test_generate_article_returns_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_llm(monkeypatch)
    unique_name = f"test_{uuid.uuid4().hex}.md"
    outfile = Path("artifacts") / unique_name
    data = {
        "theme": "Долговая нагрузка семьи",
        "structure": ["Введение", "Основная часть", "Вывод"],
        "keywords": [],
        "include_jsonld": True,
        "context_source": "off",
    }
    monkeypatch.setattr(
        "orchestrate._run_health_ping",
        lambda: {
            "ok": True,
            "message": "stub",
            "route": LLM_ROUTE,
            "fallback_used": LLM_ALLOW_FALLBACK,
        },
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
    assert llm_ping["route"] == LLM_ROUTE
    assert llm_ping["fallback_used"] is LLM_ALLOW_FALLBACK
    assert not llm_ping["ok"]
