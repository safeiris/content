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
    assert result.min_chars == 5200
    assert result.max_chars == 6800
    assert result.min_source == "profile"
    assert result.max_source == "profile"
    assert result.profile_source.endswith("profiles/finance/settings.json")


def test_resolve_swaps_when_min_greater_than_max():
    payload = {"length_limits": {"min_chars": 6800, "max_chars": 5200}}
    result = resolve_length_limits("finance", payload)
    assert result.min_chars == 5200
    assert result.max_chars == 6800
    assert result.swapped
    assert result.min_source == "brief"
    assert result.max_source == "brief"
    assert any("поменяли" in warning.lower() or "перестав" in warning.lower() for warning in result.warnings)


def test_compute_soft_length_bounds_adds_reasonable_tolerance():
    soft_min, soft_max, tol_below, tol_above = compute_soft_length_bounds(5200, 6800)
    assert soft_min == 5096
    assert soft_max == 6936
    assert tol_below == 104
    assert tol_above == 136


def test_compute_soft_length_bounds_handles_inverted_values():
    soft_min, soft_max, _, _ = compute_soft_length_bounds(6800, 5200)
    assert soft_min == 5096
    assert soft_max == 6936


def test_resolve_length_target_translates_to_soft_bounds():
    payload = {"length_target": 6000}
    result = resolve_length_limits("finance", payload)

    assert result.min_chars == 5880
    assert result.max_chars == 6120
    assert result.min_source == "brief"
    assert result.max_source == "brief"
    assert payload.get("_length_limits_requested", {}).get("target") == 6000
