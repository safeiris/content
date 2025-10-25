from skeleton_utils import normalize_skeleton_payload


def test_normalize_skeleton_payload_standardizes_keys():
    raw_payload = {
        "intro": "Intro",
        "main": "Section",
        "faq": {"q": "Q?", "a": "A!"},
        "outro": "Bye",
    }

    normalized = normalize_skeleton_payload(raw_payload)

    assert "outro" not in normalized
    assert normalized["conclusion"] == "Bye"
    assert len(normalized["main"]) == 3
    assert normalized["main"][0] == "Section"
    assert len(normalized["faq"]) == 5
    assert normalized["faq"][0] == {"q": "Q?", "a": "A!"}


def test_normalize_skeleton_payload_enforces_main_range():
    raw_payload = {
        "intro": "Intro",
        "main": [f"Block {idx}" for idx in range(10)],
        "faq": [],
        "conclusion": "Bye",
    }

    normalized = normalize_skeleton_payload(raw_payload)

    assert len(normalized["main"]) == 4
    assert normalized["main"][0] == "Block 0"


def test_normalize_skeleton_payload_fills_faq_from_seeds():
    raw_payload = {
        "intro": "Intro",
        "main": ["A", "B", "C"],
        "faq": [
            {"q": "Что такое кредитная нагрузка?", "a": "Это доля платежей от дохода."},
            {"q": "Как выбрать банк?", "a": "Сравните ставки и комиссии."},
        ],
        "conclusion": "Bye",
    }

    normalized = normalize_skeleton_payload(raw_payload)

    questions = [entry["q"] for entry in normalized["faq"]]
    assert len(questions) == 5
    assert any("долговую нагрузку" in question.lower() for question in questions)


def test_normalize_skeleton_payload_trims_to_top_five():
    raw_payload = {
        "intro": "Intro",
        "main": ["A", "B", "C"],
        "faq": [
            {"q": "Как считать процент по кредиту?", "a": "Рассчитывайте эффективную ставку."},
            {"q": "Что такое скоринг?", "a": "Скоринг оценивает кредитоспособность."},
            {"q": "Когда лучше подать заявку?", "a": "Когда доход стабилен и подтверждён."},
            {"q": "Как снизить переплату?", "a": "Вносите досрочные платежи."},
            {"q": "Зачем нужен поручитель?", "a": "Он повышает шанс одобрения."},
            {"q": "Какие документы нужны?", "a": "Паспорт и подтверждение дохода."},
        ],
        "keywords": ["процент", "скоринг", "доход"],
        "conclusion": "Bye",
    }

    normalized = normalize_skeleton_payload(raw_payload)

    questions = [entry["q"] for entry in normalized["faq"]]
    assert len(questions) == 5
    assert "Как снизить переплату?" not in questions
    assert any("процент" in q.lower() for q in questions)
    assert any("скоринг" in q.lower() for q in questions)
