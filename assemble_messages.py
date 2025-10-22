# -*- coding: utf-8 -*-
"""Stage 1 orchestration helpers for future RAG integration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import (
    APPEND_STYLE_PROFILE_DEFAULT,
    STYLE_PROFILE_PATH,
    STYLE_PROFILE_VARIANT,
)
from rules_engine import build_prompt
from retrieval import estimate_tokens, load_index, search_topk

DEFAULT_CONTEXT_TOKEN_BUDGET = 2000
STYLE_PROFILE_INSERT_ANCHOR = "Если предоставлен блок CONTEXT"
STYLE_PROFILE_LIGHT_PATH = "profiles/finance/style_profile_light.md"
_STYLE_PROFILE_VARIANT_PATHS = {
    "full": STYLE_PROFILE_PATH,
    "light": STYLE_PROFILE_LIGHT_PATH,
}

STYLES_DIR = Path("profiles/finance/styles")


def _safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_style_profile(name: str) -> Optional[Dict[str, Any]]:
    profile_path = STYLES_DIR / f"{name}.json"
    if not profile_path.exists() or profile_path.stat().st_size < 2:
        return None
    try:
        payload = _safe_load_json(profile_path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def build_style_instruction(style_data: Dict[str, Any]) -> str:
    if not style_data or not style_data.get("enabled"):
        return ""

    style_slug = str(style_data.get("style", "sravni")).strip().lower() or "sravni"
    try:
        strength = float(style_data.get("strength", 0.6))
    except (TypeError, ValueError):
        strength = 0.6

    profile = _load_style_profile(style_slug)
    if not profile:
        return ""

    tone = str(profile.get("tone", "нейтральный, понятный")).strip() or "нейтральный, понятный"
    avg_len_raw = profile.get("avg_sentence_len", 15)
    try:
        avg_len = int(avg_len_raw)
    except (TypeError, ValueError):
        try:
            avg_len = int(float(avg_len_raw))
        except (TypeError, ValueError):
            avg_len = 15

    must_tokens = profile.get("must_use", []) or []
    avoid_tokens = profile.get("avoid_words", []) or []
    hints_raw = profile.get("style_hints", []) or []

    hints: List[str] = []
    for item in hints_raw if isinstance(hints_raw, (list, tuple, set)) else []:
        text = str(item).strip()
        if text:
            hints.append(text)

    must = ", ".join(str(item).strip() for item in must_tokens if str(item).strip())
    avoid = ", ".join(str(item).strip() for item in avoid_tokens if str(item).strip())

    parts = [f"Пиши в тоне: {tone}."]
    if strength > 0.4:
        parts.append(f"Средняя длина предложений — около {avg_len} слов.")
    if strength > 0.5 and hints:
        parts.append("Следуй схеме: " + "; ".join(hints) + ".")
    if strength > 0.7:
        if must:
            parts.append(f"Используй слова: {must}.")
        if avoid:
            parts.append(f"Избегай слов: {avoid}.")

    return " ".join(parts)


def _style_profile_file_for_variant(variant: str) -> Tuple[str, Path]:
    variant_key = variant if variant in _STYLE_PROFILE_VARIANT_PATHS else "full"
    relative_path = _STYLE_PROFILE_VARIANT_PATHS[variant_key]
    absolute_path = Path(__file__).resolve().parent / relative_path
    return relative_path, absolute_path


@dataclass
class ContextBundle:
    items: List[Dict[str, object]]
    total_tokens_est: int
    index_missing: bool
    context_used: bool
    token_budget_limit: int

    @staticmethod
    def token_budget_default() -> int:
        return DEFAULT_CONTEXT_TOKEN_BUDGET


def _style_profile_conditions_met(theme_slug: str, data: Dict[str, Any]) -> bool:
    """Return ``True`` when the Finance style profile must be injected."""

    return theme_slug.strip().lower() == "finance"


def _should_apply_style_profile(
    theme_slug: str,
    data: Dict[str, Any],
    *,
    override: Optional[bool] = None,
) -> bool:
    allow = APPEND_STYLE_PROFILE_DEFAULT if override is None else bool(override)
    if not allow:
        return False
    return _style_profile_conditions_met(theme_slug, data)


@lru_cache(maxsize=None)
def _read_style_profile_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _normalize_keywords(raw_keywords: Any) -> List[str]:
    if isinstance(raw_keywords, str):
        parts = [item.strip() for item in raw_keywords.split(",")]
        return [item for item in parts if item]
    if isinstance(raw_keywords, (list, tuple, set)):
        normalized: List[str] = []
        for item in raw_keywords:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    return []


def _apply_style_profile_placeholders(text: str, data: Dict[str, Any]) -> str:
    topic = str(data.get("theme") or "").strip()
    keywords = _normalize_keywords(data.get("keywords"))
    main_keyword = keywords[0] if keywords else ""
    lsi_keywords = ", ".join(keywords[1:]) if len(keywords) > 1 else ""

    replacements = {
        "[TOPIC]": topic,
        "[MAIN_KEYWORD]": main_keyword,
        "[LSI_KEYWORDS]": lsi_keywords,
    }

    result = text
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def _load_style_profile_text(data: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    variant = STYLE_PROFILE_VARIANT if STYLE_PROFILE_VARIANT in _STYLE_PROFILE_VARIANT_PATHS else "full"
    relative_path, absolute_path = _style_profile_file_for_variant(variant)
    raw_text = _read_style_profile_file(str(absolute_path)).strip()
    if not raw_text:
        return "", None, None
    rendered = _apply_style_profile_placeholders(raw_text, data).strip()
    return rendered, relative_path, variant


def _inject_style_profile(system_prompt: str, style_profile: str) -> Tuple[str, bool]:
    profile_text = style_profile.strip()
    if not profile_text:
        return system_prompt, False

    if STYLE_PROFILE_INSERT_ANCHOR in system_prompt:
        updated_prompt = system_prompt.replace(
            STYLE_PROFILE_INSERT_ANCHOR,
            f"{profile_text}\n\n{STYLE_PROFILE_INSERT_ANCHOR}",
            1,
        )
        return updated_prompt, True

    return f"{system_prompt.rstrip()}\n\n{profile_text}", True


def retrieve_context(
    theme_slug: str,
    query: str,
    *,
    k: int = 3,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> ContextBundle:
    budget_limit = token_budget if token_budget is not None else DEFAULT_CONTEXT_TOKEN_BUDGET

    try:
        index = load_index(theme_slug)
    except FileNotFoundError:
        return ContextBundle(
            items=[],
            total_tokens_est=0,
            index_missing=True,
            context_used=False,
            token_budget_limit=budget_limit,
        )

    search_query = (query or "").strip() or theme_slug
    results = search_topk(index=index, query=search_query, k=k)
    if not results:
        return ContextBundle(
            items=[],
            total_tokens_est=0,
            index_missing=False,
            context_used=False,
            token_budget_limit=budget_limit,
        )

    limited = list(results[:k]) if k > 0 else []

    def _token_sum(items: List[Dict[str, object]]) -> int:
        return sum(int(item.get("token_estimate", estimate_tokens(item.get("text", "")))) for item in items)

    if token_budget and limited:
        limited.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        while _token_sum(limited) > token_budget and len(limited) > 1:
            limited.pop()

    total = _token_sum(limited)
    return ContextBundle(
        items=limited,
        total_tokens_est=total,
        index_missing=False,
        context_used=bool(limited),
        token_budget_limit=budget_limit,
    )


def invalidate_style_profile_cache() -> None:
    """Reset cached style profile contents so that fresh edits are used."""

    _read_style_profile_file.cache_clear()


def retrieve_exemplars(theme_slug: str, query: str, k: int = 3) -> List[Dict[str, object]]:
    """Backward-compatible helper returning only exemplar items."""

    bundle = retrieve_context(theme_slug=theme_slug, query=query, k=k)
    return bundle.items


def _build_user_instruction(payload: Dict[str, Any]) -> str:
    base_instruction = "Сгенерируй текст по указанным параметрам."
    if not payload:
        instruction = base_instruction
    else:
        hints: List[str] = []
        facts_mode = str(payload.get("facts_mode", "")).strip().lower()
        if facts_mode == "cautious":
            hints.append(
                "Работай в режиме fact-check: ссылаться только на проверенные данные, избегать категоричных утверждений и уточнять, что читателю следует перепроверять цифры."
            )

        if payload.get("include_faq"):
            hints.append("Добавь блок FAQ с короткими ответами на типовые вопросы читателей.")
        if payload.get("include_table"):
            hints.append("Вставь краткую таблицу со сравнением ключевых параметров, если это уместно.")

        if hints:
            instruction = base_instruction + " " + " ".join(hints)
        else:
            instruction = base_instruction

    return instruction


def assemble_messages(
    data_path: str = "input_example.json",
    theme_slug: str = "finance",
    *,
    k: int = 3,
    exemplars: Optional[List[Dict[str, object]]] = None,
    data: Optional[Dict[str, Any]] = None,
    append_style_profile: Optional[bool] = None,
    context_source: str = "index.json",
    custom_context_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Prepare chat-style messages for an LLM call.

    Parameters
    ----------
    data_path: str
        Path to the JSON brief that mirrors ``input_example.json``.
    theme_slug: str
        Directory name under ``profiles/`` whose exemplars should be considered.
    k: int
        Number of exemplar clips to include in the CONTEXT block.
    """

    payload: Dict[str, Any]
    if data is not None:
        payload = data
    else:
        payload_path = Path(data_path)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    system_prompt = build_prompt(payload)

    style_profile_applied = False
    style_profile_source: Optional[str] = None
    style_profile_variant_used: Optional[str] = None
    if _should_apply_style_profile(theme_slug, payload, override=append_style_profile):
        profile_text, profile_source, profile_variant = _load_style_profile_text(payload)
        system_prompt, style_profile_applied = _inject_style_profile(system_prompt, profile_text)
        if style_profile_applied:
            style_profile_source = profile_source or STYLE_PROFILE_PATH
            style_profile_variant_used = profile_variant or "full"

    system_message: Dict[str, Any] = {"role": "system", "content": system_prompt}
    if style_profile_applied:
        system_message["style_profile_applied"] = True
        if style_profile_source:
            system_message["style_profile_source"] = style_profile_source
        if style_profile_variant_used:
            system_message["style_profile_variant"] = style_profile_variant_used

    messages: List[Dict[str, Any]] = [system_message]

    context_mode = (context_source or "index.json").strip().lower() or "index.json"
    exemplar_items = exemplars
    if context_mode == "custom":
        normalized_custom = (custom_context_text or "").strip()
        if normalized_custom:
            messages.append({"role": "system", "content": f"CONTEXT (CUSTOM):\n{normalized_custom}"})
    elif context_mode != "off":
        if exemplar_items is None:
            exemplar_items = retrieve_exemplars(
                theme_slug=theme_slug,
                query=payload.get("theme", ""),
                k=k,
            )
        if exemplar_items:
            fragments: List[str] = []
            for idx, item in enumerate(exemplar_items, start=1):
                path = str(item.get("path", "unknown"))
                text = str(item.get("text", ""))
                fragment = f"<<<EXEMPLAR #{idx} | {path}>>>\n{text.strip()}"
                fragments.append(fragment.strip())
            context_block = "\n\n".join(fragments)
            messages.append({"role": "system", "content": f"CONTEXT\n{context_block}"})

    user_instruction = _build_user_instruction(payload)
    style_instruction = build_style_instruction(payload.get("style", {})) if payload else ""
    if style_instruction:
        user_prompt = f"{style_instruction}\n\n{user_instruction}".strip()
    else:
        user_prompt = user_instruction
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _preview(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


if __name__ == "__main__":
    assembled = assemble_messages()
    system_payload = assembled[0]
    system_message = system_payload["content"]
    print("=== SYSTEM PROMPT PREVIEW ===")
    print(_preview(system_message))

    if system_payload.get("style_profile_applied"):
        print("\n[style profile applied from %s]" % system_payload.get("style_profile_source", STYLE_PROFILE_PATH))

    context_message = next(
        (msg["content"] for msg in assembled[1:] if msg["role"] == "system" and msg["content"].startswith("CONTEXT")),
        None,
    )

    if context_message:
        print("\n=== CONTEXT PREVIEW ===")
        print(_preview(context_message))
    else:
        print("\n(No CONTEXT exemplars attached. Retrieval stub currently returns 0 items.)")
