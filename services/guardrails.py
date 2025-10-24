"""Post-generation guardrails and repair routines."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Iterable, List, Tuple

LOGGER = logging.getLogger("content_factory.guardrails")

_SCRIPT_RE = re.compile(r"<script[^>]*>(?P<body>.*?)</script>", re.DOTALL | re.IGNORECASE)


def _strip_script_tag(payload: str) -> str:
    match = _SCRIPT_RE.search(payload)
    if not match:
        return payload
    return match.group("body").strip()


def _normalize_json(candidate: str) -> str:
    """Apply lightweight normalisation to improve JSON parsing odds."""

    normalized = candidate.strip()
    if not normalized:
        return normalized
    # Replace smart quotes and ensure standard quotes are used.
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2018", '"').replace("\u2019", '"')
    # Remove trailing commas before closing braces/brackets.
    normalized = re.sub(r",\s*(\]|\})", r"\1", normalized)
    return normalized


def _extract_faq_entries(document: Dict[str, object]) -> List[Dict[str, str]]:
    entities: Iterable[object] = []
    if isinstance(document, dict):
        if document.get("@type") == "FAQPage" and isinstance(document.get("mainEntity"), list):
            entities = document.get("mainEntity", [])
        elif isinstance(document.get("faq"), list):
            entities = document.get("faq", [])
    parsed: List[Dict[str, str]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if entity.get("@type") == "Question":
            question = str(entity.get("name", "")).strip()
            answer_block = entity.get("acceptedAnswer")
            if isinstance(answer_block, dict):
                answer_text = str(answer_block.get("text", "")).strip()
            else:
                answer_text = str(answer_block or "").strip()
        else:
            question = str(entity.get("question", "")).strip()
            answer_text = str(entity.get("answer", "")).strip()
        if question and answer_text:
            parsed.append({"question": question, "answer": answer_text})
    return parsed


def parse_jsonld_or_repair(text: str) -> Tuple[List[Dict[str, str]], int, bool]:
    """Parse FAQ JSON-LD, attempting repairs when needed."""

    attempts = 0
    degraded = False
    if not text:
        return [], attempts, degraded

    candidates = [text, _strip_script_tag(text)]
    normalized_candidates = []
    for candidate in candidates:
        normalized = _normalize_json(candidate)
        if normalized and normalized not in normalized_candidates:
            normalized_candidates.append(normalized)

    for candidate in normalized_candidates:
        attempts += 1
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError as exc:
            LOGGER.debug("Failed to parse JSON-LD candidate: %s", exc)
            continue
        faq_entries = _extract_faq_entries(document)
        if faq_entries:
            return faq_entries, attempts, degraded

    # Second pass: try to wrap bare arrays/objects into a FAQ schema.
    for candidate in normalized_candidates:
        attempts += 1
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(document, list):
            faq_entries = []
            for idx, item in enumerate(document, start=1):
                if isinstance(item, dict):
                    question = str(item.get("q") or item.get("question") or "").strip()
                    answer = str(item.get("a") or item.get("answer") or "").strip()
                    if question and answer:
                        faq_entries.append({"question": question, "answer": answer})
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    question = str(item[0]).strip()
                    answer = str(item[1]).strip()
                    if question and answer:
                        faq_entries.append({"question": question, "answer": answer})
            if faq_entries:
                degraded = True
                return faq_entries, attempts, degraded
        if isinstance(document, dict):
            faq_field = document.get("faq")
            if isinstance(faq_field, list):
                repaired = []
                for entry in faq_field:
                    if not isinstance(entry, dict):
                        continue
                    question = str(entry.get("q") or entry.get("question") or "").strip()
                    answer = str(entry.get("a") or entry.get("answer") or "").strip()
                    if question and answer:
                        repaired.append({"question": question, "answer": answer})
                if repaired:
                    degraded = True
                    return repaired, attempts, degraded

    # Parsing failed
    degraded = True
    LOGGER.warning("JSON-LD parsing failed after %d attempts", attempts)
    return [], attempts, degraded
