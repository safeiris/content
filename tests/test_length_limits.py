from length_limits import compute_soft_length_bounds, resolve_length_limits


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


def test_compute_soft_length_bounds_adds_reasonable_tolerance():
    soft_min, soft_max, tol_below, tol_above = compute_soft_length_bounds(3500, 6000)
    assert soft_min == 3430
    assert soft_max == 6120
    assert tol_below == 70
    assert tol_above == 120


def test_compute_soft_length_bounds_handles_inverted_values():
    soft_min, soft_max, _, _ = compute_soft_length_bounds(6000, 3500)
    assert soft_min == 3430
    assert soft_max == 6120
