# -*- coding: utf-8 -*-
"""Utilities for building and querying exemplar embeddings."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from openai import OpenAI


DEFAULT_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_CLIP = 600


@dataclass
class IndexedClip:
    """Internal structure for items stored in the exemplar index."""

    path: str
    text: str
    embedding: Sequence[float]

    def as_dict(self) -> Dict[str, object]:
        return {"path": self.path, "text": self.text, "embedding": list(self.embedding)}


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("\n---\n", 1)
    if len(parts) == 2:
        return parts[1]
    return text


def _truncate_text(text: str, max_tokens: int = MAX_TOKENS_PER_CLIP) -> str:
    if not text:
        return ""
    import re

    tokens = re.split(r"(\s+)", text)
    token_budget = max_tokens
    seen_tokens = 0
    chunks: List[str] = []

    for chunk in tokens:
        if chunk.isspace():
            chunks.append(chunk)
            continue
        if not chunk:
            continue
        if seen_tokens >= token_budget:
            break
        seen_tokens += 1
        chunks.append(chunk)

    truncated = "".join(chunks).strip()
    if truncated:
        return truncated
    return text.strip()


def _embed_texts(texts: Sequence[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    client = _get_client()
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def _collect_clips(theme_slug: str) -> List[Path]:
    base_dir = Path("profiles") / theme_slug / "exemplars"
    if not base_dir.exists():
        raise FileNotFoundError(f"No exemplars directory found for theme '{theme_slug}' at {base_dir}.")
    return sorted(base_dir.rglob("*.md"))


def build_index(theme_slug: str) -> Path:
    """Build an embedding index for exemplar clips under ``profiles/<theme>/``."""

    clip_paths = _collect_clips(theme_slug)
    if not clip_paths:
        raise RuntimeError(f"No markdown clips found for theme '{theme_slug}'.")

    items: List[IndexedClip] = []
    lengths: List[int] = []
    for clip_path in clip_paths:
        raw_text = clip_path.read_text(encoding="utf-8")
        stripped = _strip_front_matter(raw_text)
        truncated = _truncate_text(stripped)
        lengths.append(len(truncated.split()))
        relative_path = clip_path.relative_to(Path("profiles") / theme_slug)
        items.append(IndexedClip(path=relative_path.as_posix(), text=truncated, embedding=[]))

    embeddings = _embed_texts([item.text for item in items])
    for item, embedding in zip(items, embeddings):
        item.embedding = embedding

    index = {
        "theme": theme_slug,
        "model": DEFAULT_MODEL,
        "clips": [item.as_dict() for item in items],
    }

    output_path = Path("profiles") / theme_slug / "index.json"
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_length = sum(lengths) / len(lengths)
    print(f"Indexed {len(items)} clips for theme '{theme_slug}'.")
    print(f"Average truncated length: {avg_length:.1f} tokens.")
    print(f"Index written to: {output_path}")

    return output_path


def load_index(theme_slug: str) -> Dict[str, object]:
    index_path = Path("profiles") / theme_slug / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found for theme '{theme_slug}' at {index_path}.")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding vectors must be of equal length.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_topk(index: Dict[str, object], query: str, k: int) -> List[Dict[str, object]]:
    if k <= 0:
        return []
    query = (query or "").strip()
    if not query:
        return []

    clips = index.get("clips") or []
    if not clips:
        return []

    query_embedding = _embed_texts([query])[0]

    scored = []
    for clip in clips:
        embedding = clip.get("embedding")
        if not isinstance(embedding, Iterable):
            continue
        try:
            score = _cosine_similarity(query_embedding, list(embedding))
        except ValueError:
            continue
        scored.append({"path": clip.get("path"), "text": clip.get("text", ""), "score": float(score)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: min(k, len(scored))]


def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Retrieval helper utilities")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Build an embedding index for a theme")
    index_parser.add_argument("--theme", required=True, help="Slug of the theme located under profiles/")

    args = parser.parse_args(argv)
    if args.command == "index":
        build_index(args.theme)
        return 0

    parser.print_help()
    return 1


def main() -> None:
    sys.exit(_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
