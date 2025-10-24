"""Resilient retrieval helper wrapping the legacy index utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from assemble_messages import ContextBundle, retrieve_context

LOGGER = logging.getLogger("content_factory.rag")


@dataclass
class RAGResult:
    documents: Sequence[dict]
    context_used: bool
    warnings: List[str]


class RAGClient:
    """Best-effort RAG client that never interrupts the pipeline."""

    def __init__(self, *, default_k: int = 3) -> None:
        self._default_k = max(0, int(default_k))

    def fetch(self, *, theme: str, query: str, k: int | None = None) -> RAGResult:
        effective_k = self._default_k if k is None else max(0, int(k))
        if effective_k == 0:
            return RAGResult(documents=[], context_used=False, warnings=["retrieval_disabled"])
        try:
            bundle: ContextBundle = retrieve_context(theme_slug=theme, query=query, k=effective_k)
        except FileNotFoundError:
            LOGGER.warning("RAG index missing for theme=%s", theme)
            return RAGResult(documents=[], context_used=False, warnings=["index_missing"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("RAG retrieval failed: %s", exc)
            return RAGResult(documents=[], context_used=False, warnings=["retrieval_error"])

        if not bundle.items:
            LOGGER.info("RAG returned no documents", extra={"theme": theme})
            return RAGResult(documents=[], context_used=False, warnings=["empty_result"])
        return RAGResult(documents=bundle.items, context_used=bundle.context_used, warnings=[])


__all__ = ["RAGClient", "RAGResult"]
