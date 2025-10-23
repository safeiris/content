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
    assert normalized["main"] == ["Section"]
    assert normalized["faq"] == [{"q": "Q?", "a": "A!"}]
