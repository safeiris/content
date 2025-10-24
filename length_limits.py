from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import DEFAULT_MAX_LENGTH, DEFAULT_MIN_LENGTH


@dataclass(frozen=True)
class ResolvedLengthLimits:
    """Normalized length requirements with provenance metadata."""

    min_chars: int
    max_chars: int
    min_source: str
    max_source: str
    swapped: bool = False
    warnings: Tuple[str, ...] = ()
    profile_source: Optional[str] = None


SOFT_RANGE_PERCENT = 0.02
SOFT_RANGE_MIN_BELOW = 50
SOFT_RANGE_MIN_ABOVE = 100


def compute_soft_length_bounds(min_chars: int, max_chars: int) -> Tuple[int, int, int, int]:
    """Return relaxed length bounds around requested min/max values.

    The lower tolerance is at least ``SOFT_RANGE_MIN_BELOW`` characters or 2% of the
    minimum requirement. The upper tolerance is at least
    ``SOFT_RANGE_MIN_ABOVE`` characters or 2% of the maximum requirement. Both
    tolerances are additionally capped by zero to avoid negative bounds.
    """

    try:
        min_value = int(min_chars)
    except (TypeError, ValueError):  # defensive: keep behaviour predictable
        min_value = 0
    try:
        max_value = int(max_chars)
    except (TypeError, ValueError):
        max_value = 0

    if max_value < min_value:
        min_value, max_value = max_value, min_value

    min_value = max(0, min_value)
    max_value = max(0, max_value)

    lower_tolerance = 0
    if min_value > 0:
        lower_tolerance = min(
            min_value,
            max(SOFT_RANGE_MIN_BELOW, int(round(min_value * SOFT_RANGE_PERCENT))),
        )
    upper_tolerance = max(SOFT_RANGE_MIN_ABOVE, int(round(max_value * SOFT_RANGE_PERCENT)))

    soft_min = max(0, min_value - lower_tolerance)
    soft_max = max_value + upper_tolerance

    return soft_min, soft_max, lower_tolerance, upper_tolerance


def resolve_length_limits(theme: str, payload: Dict[str, Any]) -> ResolvedLengthLimits:
    """Determine min/max character limits using brief → profile → defaults."""

    brief_min, brief_max = _extract_brief_limits(payload)
    profile_min, profile_max, profile_source = _load_profile_limits(theme)

    min_value, min_source = _choose_limit(brief_min, profile_min, DEFAULT_MIN_LENGTH)
    max_value, max_source = _choose_limit(brief_max, profile_max, DEFAULT_MAX_LENGTH)

    swapped = False
    warnings: Tuple[str, ...] = ()

    if max_value < min_value:
        swapped = True
        min_value, max_value, min_source, max_source = (
            max_value,
            min_value,
            max_source,
            min_source,
        )
        warnings = (
            "Минимальный объём в брифе был больше максимального; значения переставлены местами.",
        )

    return ResolvedLengthLimits(
        min_chars=min_value,
        max_chars=max_value,
        min_source=min_source,
        max_source=max_source,
        swapped=swapped,
        warnings=warnings,
        profile_source=profile_source,
    )


def load_profile_length_limits(theme: str) -> Optional[Dict[str, int]]:
    """Expose raw profile defaults for UI consumers."""

    min_value, max_value, _ = _load_profile_limits(theme)
    if min_value is None and max_value is None:
        return None
    result: Dict[str, int] = {}
    if min_value is not None:
        result["min_chars"] = min_value
    if max_value is not None:
        result["max_chars"] = max_value
    return result or None


def _choose_limit(
    brief_value: Optional[int], profile_value: Optional[int], default_value: int
) -> Tuple[int, str]:
    if brief_value is not None:
        return brief_value, "brief"
    if profile_value is not None:
        return profile_value, "profile"
    return default_value, "default"


def _extract_brief_limits(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(payload, dict):
        return None, None

    length_block = payload.get("length_limits")
    min_value = None
    max_value = None
    if isinstance(length_block, dict):
        min_value = _safe_positive_int(
            _first_present(length_block, ["min_chars", "min", "min_length"])
        )
        max_value = _safe_positive_int(
            _first_present(length_block, ["max_chars", "max", "max_length"])
        )

    if min_value is None:
        min_value = _safe_positive_int(
            _first_present(payload, ["min_len", "min_chars", "min_length"])
        )
    if max_value is None:
        max_value = _safe_positive_int(
            _first_present(payload, ["max_len", "max_chars", "max_length"])
        )

    return min_value, max_value


def _first_present(container: Dict[str, Any], keys: Tuple[str, ...] | list[str]) -> Any:
    for key in keys:
        if key in container:
            return container[key]
    return None


def _load_profile_limits(theme: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    theme_slug = str(theme or "").strip()
    if not theme_slug:
        return None, None, None

    settings_path = Path("profiles") / theme_slug / "settings.json"
    if not settings_path.exists():
        return None, None, None

    try:
        raw_text = settings_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, json.JSONDecodeError):
        return None, None, settings_path.as_posix()

    if not isinstance(payload, dict):
        return None, None, settings_path.as_posix()

    length_section = payload.get("defaults")
    if isinstance(length_section, dict):
        length_section = length_section.get("length_limits")
    else:
        length_section = payload.get("length_limits")

    if not isinstance(length_section, dict):
        return None, None, settings_path.as_posix()

    min_value = _safe_positive_int(
        _first_present(length_section, ["min_chars", "min", "min_length"])
    )
    max_value = _safe_positive_int(
        _first_present(length_section, ["max_chars", "max", "max_length"])
    )
    return min_value, max_value, settings_path.as_posix()


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0:
        return None
    return candidate


__all__ = [
    "ResolvedLengthLimits",
    "resolve_length_limits",
    "load_profile_length_limits",
]
