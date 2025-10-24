from deterministic_pipeline import DeterministicPipeline


def make_pipeline() -> DeterministicPipeline:
    return DeterministicPipeline(
        topic="Тест",
        base_outline=["Введение", "Основная часть", "Вывод"],
        keywords=[],
        min_chars=1000,
        max_chars=2000,
        messages=[{"role": "system", "content": "Ты модель"}],
        model="stub-model",
        temperature=0.1,
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
