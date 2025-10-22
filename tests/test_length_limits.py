from length_limits import resolve_length_limits


def test_resolve_uses_brief_values():
    payload = {"length_limits": {"min_chars": 1000, "max_chars": 1200}}
    result = resolve_length_limits("finance", payload)
    assert result.min_chars == 1000
    assert result.max_chars == 1200
    assert result.min_source == "brief"
    assert result.max_source == "brief"
    assert not result.warnings


def test_resolve_uses_profile_defaults_when_missing():
    result = resolve_length_limits("finance", {})
    assert result.min_chars == 3200
    assert result.max_chars == 5400
    assert result.min_source == "profile"
    assert result.max_source == "profile"
    assert result.profile_source.endswith("profiles/finance/settings.json")


def test_resolve_swaps_when_min_greater_than_max():
    payload = {"length_limits": {"min_chars": 6000, "max_chars": 3500}}
    result = resolve_length_limits("finance", payload)
    assert result.min_chars == 3500
    assert result.max_chars == 6000
    assert result.swapped
    assert result.min_source == "brief"
    assert result.max_source == "brief"
    assert any("поменяли" in warning.lower() or "перестав" in warning.lower() for warning in result.warnings)
