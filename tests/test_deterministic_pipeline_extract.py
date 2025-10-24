import itertools
import json

from deterministic_pipeline import DeterministicPipeline, PipelineStep
from llm_client import GenerationResult
from validators import ValidationResult


def make_pipeline() -> DeterministicPipeline:
    return DeterministicPipeline(
        topic="Тест",
        base_outline=["Введение", "Основная часть", "Вывод"],
        required_keywords=[],
        min_chars=2800,
        max_chars=3600,
        messages=[{"role": "system", "content": "Системный промпт"}],
        model="stub-model",
        max_tokens=1800,
        timeout_s=30,
        faq_questions=5,
    )


def test_section_budget_target_close_to_goal():
    pipeline = make_pipeline()
    total_target = sum(item.target_chars for item in pipeline.section_budgets)
    assert pipeline.min_chars <= total_target <= pipeline.max_chars
    target_total = min(pipeline.max_chars, max(pipeline.min_chars, 6000))
    assert abs(total_target - target_total) <= 400
    assert pipeline.section_budgets[0].title == "Введение"


def test_pipeline_expands_short_draft(monkeypatch):
    pipeline = make_pipeline()

    def fake_validate(self, text):
        return ValidationResult(True, True, True, True, True, stats={"chars": len(text)})

    monkeypatch.setattr("deterministic_pipeline.DeterministicPipeline._validate", fake_validate)
    monkeypatch.setattr(
        "deterministic_pipeline.validate_article",
        lambda *args, **kwargs: ValidationResult(True, True, True, True, True),
    )

    responses = itertools.cycle(
        [
            "Короткий черновик без достаточной длины.",
            "Дополнительный блок, который расширяет текст и добавляет детали.",
            "Финальный вариант текста с достаточной длиной и структурой.",
        ]
    )

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
        text = next(responses)
        payload = {
            "intro": text,
            "main_headers": [
                "Введение",
                "Аналитика",
                "Решения",
            ],
            "sections": [
                {"title": "Введение", "body": text},
                {"title": "Аналитика", "body": text},
                {"title": "Решения", "body": text},
            ],
            "faq": [
                {
                    "q": f"Вопрос {idx}",
                    "a": f"Ответ {idx}. Подробное пояснение с примерами и выводами.",
                }
                for idx in range(1, 6)
            ],
            "conclusion": text,
            "conclusion_heading": "Вывод",
        }
        return GenerationResult(
            text=json.dumps(payload, ensure_ascii=False),
            model_used="stub-model",
            retry_used=False,
            fallback_used=None,
            metadata={"usage_output_tokens": 512 if step is PipelineStep.SKELETON else 128},
        )

    monkeypatch.setattr("deterministic_pipeline.DeterministicPipeline._call_llm", fake_call)

    state = pipeline.run()
    assert pipeline.checkpoints[PipelineStep.SKELETON]
    assert pipeline.checkpoints[PipelineStep.TRIM]
    assert state.validation.is_valid
    assert state.logs[-1].step is PipelineStep.TRIM
