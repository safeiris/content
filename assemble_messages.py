# -*- coding: utf-8 -*-
"""Stage 1 orchestration helpers for future RAG integration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rules_engine import build_prompt
from retrieval import load_index, search_topk


def retrieve_exemplars(theme_slug: str, query: str, k: int = 3) -> List[Dict[str, object]]:
    """Stub for exemplar retrieval.

    Stage 2 will plug in embedding-powered search here. The contract should
    return the top-``k`` items as dictionaries with the following shape::

        {"path": str, "text": str, "score": float}

    ``path`` is the source file, ``text`` is the fragment content (already
    trimmed to fit downstream token budgets), and ``score`` is a ranking weight
    where higher means more relevant.
    """

    try:
        index = load_index(theme_slug)
    except FileNotFoundError:
        return []

    search_query = (query or "").strip() or theme_slug
    results = search_topk(index=index, query=search_query, k=k)
    return [{"path": item.get("path"), "text": item.get("text", ""), "score": item.get("score", 0.0)} for item in results]


def assemble_messages(
    data_path: str = "input_example.json",
    theme_slug: str = "finance",
    *,
    k: int = 3,
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

    payload_path = Path(data_path)
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    system_prompt = build_prompt(data)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    exemplars = retrieve_exemplars(theme_slug=theme_slug, query=data.get("theme", ""), k=k)
    if exemplars:
        fragments: List[str] = []
        for idx, item in enumerate(exemplars, start=1):
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
