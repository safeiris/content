"""Soft guardrails for structured model output."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

LOGGER = logging.getLogger("content_factory.guardrails")

_SCRIPT_RE = re.compile(r"<script[^>]*>(?P<body>.*?)</script>", re.DOTALL | re.IGNORECASE)
_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": '"',
    "\u2019": '"',
    "\u00ab": '"',
    "\u00bb": '"',
}


@dataclass(slots=True)
class GuardrailResult:
    """Outcome of JSON-LD parsing with repair attempts."""

    ok: bool
    faq_entries: List[Dict[str, str]]
    attempts: int
    degradation_flags: List[str] = field(default_factory=list)
    repaired_json: Optional[str] = None
    error: Optional[str] = None


def parse_and_repair_jsonld(raw_payload: Any, *, trace_id: Optional[str] = None, max_repairs: int = 2) -> GuardrailResult:
    """Parse JSON-LD payload with light-weight repair strategies."""

    text_candidate = _normalize_payload(raw_payload)
    if not text_candidate:
        return GuardrailResult(
            ok=False,
            faq_entries=[],
            attempts=0,
            degradation_flags=["jsonld_missing"],
            error="empty_payload",
        )

    attempts = 0
    candidates = _build_candidates(text_candidate)
    degradation_flags: List[str] = []

    for candidate in candidates:
        attempts += 1
        faq_entries, repaired = _try_parse_candidate(candidate)
        if faq_entries:
            return GuardrailResult(
                ok=True,
                faq_entries=faq_entries,
                attempts=attempts,
                degradation_flags=degradation_flags,
                repaired_json=repaired,
            )

    # Repair loop: we do not call external services here, but we try to coerce structures.
    repair_attempt = 0
    last_error = "parse_failed"
    while repair_attempt < max_repairs:
        repair_attempt += 1
        attempts += 1
        repaired_text = _apply_repair_heuristics(text_candidate, iteration=repair_attempt)
        faq_entries, repaired_json = _try_parse_candidate(repaired_text)
        if faq_entries:
            degradation_flags.append("jsonld_repaired")
            return GuardrailResult(
                ok=True,
                faq_entries=faq_entries,
                attempts=attempts,
                degradation_flags=degradation_flags,
                repaired_json=repaired_json,
            )
        last_error = "repair_failed"

    degradation_flags.append("jsonld_missing")
    LOGGER.warning("JSON-LD parsing failed", extra={"trace_id": trace_id, "attempts": attempts})
    return GuardrailResult(
        ok=False,
        faq_entries=[],
        attempts=attempts,
        degradation_flags=degradation_flags,
        error=last_error,
    )


def _normalize_payload(raw_payload: Any) -> str:
    if isinstance(raw_payload, (dict, list)):
        try:
            return json.dumps(raw_payload, ensure_ascii=False)
        except TypeError:
            return ""
    if not isinstance(raw_payload, str):
        return ""
    text = raw_payload.strip()
    if not text:
        return ""
    return text.translate(str.maketrans(_SMART_QUOTES))


def _build_candidates(text: str) -> List[str]:
    candidates = [text]
    script_match = _SCRIPT_RE.search(text)
    if script_match:
        body = script_match.group("body").strip()
        if body and body not in candidates:
            candidates.append(body)
    compact = re.sub(r",\s*(\]|\})", r"\1", text)
    if compact and compact not in candidates:
        candidates.append(compact)
    return candidates


def _try_parse_candidate(candidate: str) -> tuple[List[Dict[str, str]], Optional[str]]:
    try:
        document = json.loads(candidate)
    except json.JSONDecodeError:
        return [], None
    faq_entries = _extract_faq_entries(document)
    if faq_entries:
        try:
            repaired = json.dumps(document, ensure_ascii=False)
        except TypeError:
            repaired = candidate
        return faq_entries, repaired
    return [], None


def _extract_faq_entries(document: Any) -> List[Dict[str, str]]:
    entries: Iterable[Any] = []
    if isinstance(document, dict):
        if document.get("@type") == "FAQPage" and isinstance(document.get("mainEntity"), list):
            entries = document.get("mainEntity", [])
        elif isinstance(document.get("faq"), list):
            entries = document.get("faq", [])
        elif isinstance(document.get("items"), list):
            entries = document.get("items", [])
    elif isinstance(document, list):
        entries = document

    normalized: List[Dict[str, str]] = []
    for item in entries:
        if isinstance(item, dict):
            question = str(item.get("question") or item.get("name") or item.get("q") or "").strip()
            answer_field = item.get("answer") or item.get("acceptedAnswer") or item.get("a")
            if isinstance(answer_field, dict):
                answer = str(answer_field.get("text") or "").strip()
            else:
                answer = str(answer_field or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            question = str(item[0]).strip()
            answer = str(item[1]).strip()
        else:
            continue
        if question and answer:
            normalized.append({"question": question, "answer": answer})
    return normalized


def _apply_repair_heuristics(text: str, *, iteration: int) -> str:
    if iteration == 1:
        return text.replace("'", '"')
    normalized = re.sub(r",\s*(\]|\})", r"\1", text)
    normalized = normalized.replace("`", '"')
    return normalized


__all__ = ["GuardrailResult", "parse_and_repair_jsonld"]
