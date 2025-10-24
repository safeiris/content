from deterministic_pipeline import (
    DeterministicPipeline,
    SkeletonBatchKind,
    SkeletonBatchPlan,
)
from config import LLM_ROUTE
from llm_client import GenerationResult


def make_pipeline() -> DeterministicPipeline:
    return DeterministicPipeline(
        topic="Тест",
        base_outline=["Введение", "Основная часть", "Вывод"],
        keywords=[],
        min_chars=1000,
        max_chars=2000,
        messages=[{"role": "system", "content": "Ты модель"}],
        model="stub-model",
        max_tokens=500,
        timeout_s=30,
    )


def test_extract_response_json_tolerates_wrapped_payload():
    pipeline = make_pipeline()
    raw = "<response_json>{\"intro\": \"text\", \"main\": []}</response_json> trailing"
    assert pipeline._extract_response_json(raw) == {"intro": "text", "main": []}


def test_extract_response_json_with_leading_text():
    pipeline = make_pipeline()
    raw = "Ответ: {\"faq\": [{\"q\": \"Q?\", \"a\": \"A!\"}] }"
    assert pipeline._extract_response_json(raw) == {"faq": [{"q": "Q?", "a": "A!"}]}


def test_build_batch_placeholder_provides_payload():
    pipeline = make_pipeline()
    outline = pipeline._prepare_outline()
    batch = SkeletonBatchPlan(kind=SkeletonBatchKind.MAIN, indices=[0, 1], label="main[1-2]")
    placeholder = pipeline._build_batch_placeholder(
        batch,
        outline=outline,
        target_indices=[0, 1],
    )
    assert pipeline._batch_has_payload(SkeletonBatchKind.MAIN, placeholder)


def test_run_skeleton_uses_placeholder_when_cap(monkeypatch):
    pipeline = make_pipeline()

    def fake_call(
        self,
        *,
        step,
        messages,
        max_tokens=None,
        previous_response_id=None,
        responses_format=None,
        allow_incomplete=False,
    ):
        return GenerationResult(
            text="",
            model_used="stub-model",
            retry_used=False,
            fallback_used=None,
            fallback_reason=None,
            api_route=LLM_ROUTE,
            schema="json",
            metadata={"status": "incomplete", "incomplete_reason": "max_output_tokens"},
        )

    monkeypatch.setattr("deterministic_pipeline.DeterministicPipeline._call_llm", fake_call)
    monkeypatch.setattr(
        "deterministic_pipeline.DeterministicPipeline._run_fallback_batch",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "deterministic_pipeline.DeterministicPipeline._run_cap_fallback_batch",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "deterministic_pipeline.DeterministicPipeline._render_skeleton_markdown",
        lambda self, payload: ("", {"outline": []}),
    )

    pipeline._run_skeleton()
    payload = pipeline.skeleton_payload
    assert payload["intro"]
    assert all(section for section in payload["main"])
    assert payload["conclusion"]
