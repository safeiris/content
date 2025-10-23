import json
import uuid
from pathlib import Path

from deterministic_pipeline import DeterministicPipeline, PipelineStep
from faq_builder import build_faq_block
from keyword_injector import LOCK_START_TEMPLATE, inject_keywords
import json
import uuid
from pathlib import Path

from deterministic_pipeline import DeterministicPipeline, PipelineStep
from keyword_injector import LOCK_START_TEMPLATE, inject_keywords
from length_trimmer import trim_text
from llm_client import GenerationResult
from orchestrate import generate_article_from_payload, gather_health_status
from validators import strip_jsonld, validate_article


def test_keyword_injection_adds_terms_section():
    base_text = "## Основная часть\n\nОписание практик.\n\n## FAQ\n\n<!--FAQ_START-->\n<!--FAQ_END-->\n"
    result = inject_keywords(base_text, ["ключевая фраза", "дополнительный термин"])
    assert "### Разбираемся в терминах" in result.text
    assert LOCK_START_TEMPLATE.format(term="ключевая фраза") in result.text
    assert result.coverage["дополнительный термин"]
    main_section = result.text.split("## FAQ", 1)[0]
    expected_phrase = (
        "Дополнительно рассматривается "
        + f"{LOCK_START_TEMPLATE.format(term='ключевая фраза')}ключевая фраза<!--LOCK_END-->"
        + " через прикладные сценарии."
    )
    assert expected_phrase in main_section


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
    article = (
        f"## Введение\n\n{intro}\n\n"
        f"{LOCK_START_TEMPLATE.format(term='важный термин')}важный термин<!--LOCK_END-->\n\n"
        f"{removable}\n\n"
        "## FAQ\n\n<!--FAQ_START-->\n**Вопрос 1.** Что важно?\n\n**Ответ.** Ответ с деталями.\n\n<!--FAQ_END-->"
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
    result = validate_article(text, keywords=["ключ"], min_chars=10, max_chars=1000)
    assert not result.keywords_ok


def test_validator_length_ignores_jsonld():
    payload = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"Вопрос {idx}",
                "acceptedAnswer": {"@type": "Answer", "text": f"Ответ {idx}"},
            }
            for idx in range(1, 6)
        ],
    }
    faq_block = "\n".join(
        [
            "**Вопрос 1.** Что?",
            "**Ответ.** Ответ.",
            "",
            "**Вопрос 2.** Что?",
            "**Ответ.** Ответ.",
            "",
            "**Вопрос 3.** Что?",
            "**Ответ.** Ответ.",
            "",
            "**Вопрос 4.** Что?",
            "**Ответ.** Ответ.",
            "",
            "**Вопрос 5.** Что?",
            "**Ответ.** Ответ.",
            "",
        ]
    )
    article = (
        "## Введение\n\n"
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
    min_chars = max(10, base_length - 5)
    max_chars = base_length + 5
    result = validate_article(article, keywords=["ключ"], min_chars=min_chars, max_chars=max_chars)
    assert result.length_ok
    assert result.jsonld_ok


def _stub_llm(monkeypatch):
    skeleton_body = "\n\n".join(
        [
            "Абзац с анализом показателей и практическими советами для семейного бюджета. "
            "Расчёт коэффициентов сопровождаем примерами и перечнем действий." for _ in range(45)
        ]
    )
    skeleton_text = (
        "## Введение\n\n"
        "Кратко объясняем, как долговая нагрузка влияет на решения семьи и почему ключ 1 помогает структурировать анализ.\n\n"
    "## Аналитика\n\n"
    f"{skeleton_body}\n\n"
    "## Решения\n\n"
    "Разбираем стратегии снижения нагрузки, контрольные точки и цифровые инструменты, уделяя внимание тому, как ключ 2 и ключ 3"
    " помогают планировать шаги.\n\n"
    "Создаём календарь контроля, в котором ключ 4 и ключ 5 отмечены как приоритетные метрики для семьи.\n\n"
    "## FAQ\n\n<!--FAQ_START-->\n<!--FAQ_END-->\n\n"
    "## Вывод\n\nПодводим итоги и фиксируем шаги для регулярного пересмотра бюджета, подчёркивая, как ключ 2 и ключ 3 помогают контролирова"
    "ть изменения."
)
    faq_payload = {
        "faq": [
            {
                "question": "Как определить допустимую долговую нагрузку?",
                "answer": "Сравните платежи с ежемесячным доходом и удерживайте коэффициент не выше 30–35%.",
            },
            {
                "question": "Какие данные нужны для расчёта?",
                "answer": "Соберите сведения по кредитам, страховым взносам и коммунальным платежам за последний год.",
            },
            {
                "question": "Что делать при превышении порога?",
                "answer": "Пересмотрите график платежей, договоритесь о реструктуризации и выделите обязательные траты.",
            },
            {
                "question": "Как планировать резерв?",
                "answer": "Откладывайте не менее двух ежемесячных платежей на отдельный счёт с быстрым доступом.",
            },
            {
                "question": "Какие сервисы помогают контролю?",
                "answer": "Используйте банковские дашборды и напоминания календаря, чтобы отслеживать даты и суммы.",
            },
        ]
    }

    def fake_call(self, *, step, messages, max_tokens=None):
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
    assert 3500 <= length_no_spaces <= 6000
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
    class DummyResponse:
        status_code = 200

    monkeypatch.setattr("orchestrate.httpx.get", lambda *args, **kwargs: DummyResponse())
    status = gather_health_status(theme="")
    assert not status["ok"]
    assert not status["checks"]["theme_index"]["ok"]
