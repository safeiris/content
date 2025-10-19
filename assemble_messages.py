# -*- coding: utf-8 -*-
"""Stage 1 orchestration helpers for future RAG integration."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rules_engine import build_prompt
from retrieval import estimate_tokens, load_index, search_topk

DEFAULT_CONTEXT_TOKEN_BUDGET = 2000


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


def retrieve_exemplars(theme_slug: str, query: str, k: int = 3) -> List[Dict[str, object]]:
    """Backward-compatible helper returning only exemplar items."""

    bundle = retrieve_context(theme_slug=theme_slug, query=query, k=k)
    return bundle.items


def assemble_messages(
    data_path: str = "input_example.json",
    theme_slug: str = "finance",
    *,
    k: int = 3,
    exemplars: Optional[List[Dict[str, object]]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
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

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    exemplar_items = exemplars
    if exemplar_items is None:
        exemplar_items = retrieve_exemplars(theme_slug=theme_slug, query=payload.get("theme", ""), k=k)
    if exemplar_items:
        fragments: List[str] = []
        for idx, item in enumerate(exemplar_items, start=1):
            path = str(item.get("path", "unknown"))
            text = str(item.get("text", ""))
            fragment = f"<<<EXEMPLAR #{idx} | {path}>>>\n{text.strip()}"
            fragments.append(fragment.strip())
        context_block = "\n\n".join(fragments)
        messages.append({"role": "system", "content": f"CONTEXT\n{context_block}"})

    messages.append({"role": "user", "content": "Сгенерируй текст по указанным параметрам."})
    return messages


def _preview(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


if __name__ == "__main__":
    assembled = assemble_messages()
    system_message = assembled[0]["content"]
    print("=== SYSTEM PROMPT PREVIEW ===")
    print(_preview(system_message))

    context_message = next(
        (msg["content"] for msg in assembled[1:] if msg["role"] == "system" and msg["content"].startswith("CONTEXT")),
        None,
    )

    if context_message:
        print("\n=== CONTEXT PREVIEW ===")
        print(_preview(context_message))
    else:
        print("\n(No CONTEXT exemplars attached. Retrieval stub currently returns 0 items.)")
