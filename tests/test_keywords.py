from __future__ import annotations

import keywords


def test_merge_keywords_skips_auto_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(keywords, "KEYWORDS_ALLOW_AUTO", False, raising=False)

    manual_used, auto_used, final = keywords.merge_keywords(
        ["Ручной ключ"],
        ["Автоматический вариант"],
    )

    assert manual_used == ["ручной ключ"]
    assert auto_used == []
    assert final == manual_used


def test_merge_keywords_includes_auto_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(keywords, "KEYWORDS_ALLOW_AUTO", True, raising=False)

    manual_used, auto_used, final = keywords.merge_keywords(
        ["Manual"],
        ["Auto Candidate"],
    )

    assert manual_used == ["manual"]
    assert auto_used == ["auto candidate"]
    assert final == manual_used + auto_used
