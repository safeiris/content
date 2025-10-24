# -*- coding: utf-8 -*-
"""Helpers for normalizing skeleton payloads returned by LLM."""

from __future__ import annotations

import logging
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)

_CANONICAL_CONCLUSION_KEYS = ("conclusion", "outro", "ending", "final", "summary")
_DEFAULT_MAIN_PLACEHOLDER = (
    "Этот раздел будет расширен детальными рекомендациями в финальной версии статьи."
)


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _describe_keys(payload: Dict[str, Any]) -> str:
    descriptors = []
    if "intro" in payload:
        descriptors.append("intro")
    if "main" in payload:
        descriptors.append("main[]")
    if "faq" in payload:
        descriptors.append("faq[]")
    if "conclusion" in payload:
        descriptors.append("conclusion")
    return ",".join(descriptors)


def normalize_skeleton_payload(payload: Any) -> Any:
    """Return a normalized skeleton payload with canonical keys."""

    if not isinstance(payload, dict):
        return payload

    normalized: Dict[str, Any] = dict(payload)

    conclusion_value = None
    for key in _CANONICAL_CONCLUSION_KEYS:
        if key in normalized:
            value = normalized.get(key)
            if value is not None and str(value).strip():
                conclusion_value = value
                break
    if conclusion_value is not None:
        normalized["conclusion"] = conclusion_value
    for legacy_key in ("outro", "ending", "final", "summary"):
        normalized.pop(legacy_key, None)

    normalized_main = [
        str(item or "").strip() for item in _as_list(normalized.get("main")) if str(item or "").strip()
    ]
    if len(normalized_main) > 6:
        LOGGER.info("LOG:SKELETON_MAIN_TRIM normalize from=%d to=6", len(normalized_main))
        normalized_main = normalized_main[:6]
    while len(normalized_main) < 3:
        normalized_main.append(_DEFAULT_MAIN_PLACEHOLDER)
    normalized["main"] = normalized_main
    normalized["faq"] = _as_list(normalized.get("faq"))

    keys_descriptor = _describe_keys(normalized)
    LOGGER.info("LOG:SKELETON_NORMALIZED keys=%s", keys_descriptor)
    return normalized


__all__ = ["normalize_skeleton_payload"]
