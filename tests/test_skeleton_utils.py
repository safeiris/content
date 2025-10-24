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
    assert normalized["faq"] == [{"q": "Q?", "a": "A!"}]


def test_normalize_skeleton_payload_enforces_main_range():
    raw_payload = {
        "intro": "Intro",
        "main": [f"Block {idx}" for idx in range(10)],
        "faq": [],
        "conclusion": "Bye",
    }

    normalized = normalize_skeleton_payload(raw_payload)

    assert len(normalized["main"]) == 6
    assert normalized["main"][0] == "Block 0"
