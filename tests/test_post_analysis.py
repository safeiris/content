from post_analysis import PostAnalysisRequirements, analyze


def _make_requirements(keywords):
    return PostAnalysisRequirements(
        min_chars=0,
        max_chars=10000,
        keywords=list(keywords),
        keyword_mode="strict",
        faq_questions=None,
        sources=[],
        style_profile="",
        length_sources=None,
        jsonld_enabled=False,
    )


def test_keyword_detection_respects_word_boundaries():
    requirements = _make_requirements(["банк", "страхование жизни"])
    text = (
        "Это банковский продукт. Страхование жизни доступно.\n\n"
        "FAQ\n"
        "1. Что такое страхование жизни?\n"
        "Ответ.\n"
        "2. Как оформить полис?\n"
        "Ответ.\n"
        "3. Какие преимущества?\n"
        "Ответ."
    )  # no standalone "банк"
    report = analyze(text, requirements=requirements, model="gpt", retry_count=0, fallback_used=False)

    assert "банк" in report["missing_keywords"]
    assert "страхование жизни" not in report["missing_keywords"]
    assert "keywords" in report["fail_reasons"]
    assert "faq" not in report["fail_reasons"]
    assert not report["meets_requirements"]


def test_keyword_normalization_handles_yo_and_nbsp():
    requirements = _make_requirements(["ёлка", "тёплый прием"])
    text = (
        "Елка украшена. Теплый\u00a0прием гостей организован.\n\n"
        "FAQ\n"
        "1. Что дарить?\n"
        "Ответ.\n"
        "2. Когда готовиться?\n"
        "Ответ.\n"
        "3. Как украсить дом?\n"
        "Ответ."
    )
    report = analyze(text, requirements=requirements, model="gpt", retry_count=0, fallback_used=False)

    assert not report["missing_keywords"], report["missing_keywords"]
    assert report["meets_requirements"]
    assert report["fail_reasons"] == []


def test_analyze_reports_length_sources_and_jsonld_requirement():
    requirements = PostAnalysisRequirements(
        min_chars=100,
        max_chars=500,
        keywords=[],
        keyword_mode="strict",
        faq_questions=5,
        sources=[],
        style_profile="",
        length_sources={"min": "user", "max": "profile"},
        jsonld_enabled=True,
    )
    text = "a" * 150
    report = analyze(text, requirements=requirements, model="gpt", retry_count=0, fallback_used=False)

    assert report["length"]["source"] == {"min": "user", "max": "profile"}
    assert report["length_limits_applied"]["source"] == {"min": "user", "max": "profile"}
    assert report["faq"]["jsonld_enabled"] is True
    assert report["faq"]["min"] == 5
    assert "faq" in report["fail_reasons"]
