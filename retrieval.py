# -*- coding: utf-8 -*-
"""Utilities for building and querying exemplar embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI


DEFAULT_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_CLIP = 600

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Estimate token length using a lightweight heuristic."""

    if not text:
        return 0
    # Fallback to ASCII-aware splitting if the unicode class fails (older regex engines)
    try:
        tokens = _TOKEN_PATTERN.findall(text)
    except re.error:
        tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


@dataclass
class IndexedClip:
    """Internal structure for items stored in the exemplar index."""

    path: str
    text: str
    embedding: Sequence[float]
    content_hash: str = ""
    token_estimate: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "text": self.text,
            "embedding": list(self.embedding),
            "content_hash": self.content_hash,
            "token_estimate": self.token_estimate,
        }


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


def _truncate_text(text: str, max_tokens: int = MAX_TOKENS_PER_CLIP) -> Tuple[str, int]:
    if not text:
        return "", 0

    parts = re.split(r"(\s+)", text)
    seen_tokens = 0
    acc: List[str] = []

    for part in parts:
        if part.isspace():
            if acc:
                acc.append(part)
            continue
        if not part:
            continue
        if seen_tokens >= max_tokens:
            break
        seen_tokens += estimate_tokens(part)
        if seen_tokens > max_tokens:
            break
        acc.append(part)

    truncated = "".join(acc).strip()
    if not truncated:
        truncated = text.strip()
    token_estimate = estimate_tokens(truncated)
    return truncated, token_estimate


def _embed_texts(texts: Sequence[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    client = _get_client()
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def _collect_clips(theme_slug: str) -> List[Path]:
    base_dir = Path("profiles") / theme_slug / "exemplars"
    if not base_dir.exists():
        raise FileNotFoundError(f"No exemplars directory found for theme '{theme_slug}' at {base_dir}.")
    return sorted(base_dir.rglob("*.md"))


def build_index(theme_slug: str) -> Dict[str, object]:
    """Build an embedding index for exemplar clips under ``profiles/<theme>/``.

    Returns
    -------
    Dict[str, object]
        Aggregate information about the rebuilt index including statistics used by
        the API layer (clip count and average lengths).
    """

    clip_paths = _collect_clips(theme_slug)
    if not clip_paths:
        raise RuntimeError(f"No markdown clips found for theme '{theme_slug}'.")

    items: List[IndexedClip] = []
    lengths: List[int] = []
    token_lengths: List[int] = []
    cached = 0

    index_path = Path("profiles") / theme_slug / "index.json"
    existing_map: Dict[str, Dict[str, object]] = {}
    if index_path.exists():
        try:
            existing_index = json.loads(index_path.read_text(encoding="utf-8"))
            for clip in existing_index.get("clips", []):
                path = clip.get("path")
                if isinstance(path, str):
                    existing_map[path] = clip
        except json.JSONDecodeError:
            existing_map = {}

    embeddings_to_compute: List[str] = []
    pending_indices: List[int] = []

    for clip_path in clip_paths:
        raw_text = clip_path.read_text(encoding="utf-8")
        stripped = _strip_front_matter(raw_text)
        truncated, token_estimate = _truncate_text(stripped)
        lengths.append(len(truncated.split()))
        token_lengths.append(token_estimate)
        relative_path = clip_path.relative_to(Path("profiles") / theme_slug)
        rel_path = relative_path.as_posix()
        content_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()

        embedding: List[float] = []
        existing = existing_map.get(rel_path)
        if existing and existing.get("content_hash") == content_hash:
            prev_embedding = existing.get("embedding")
            if isinstance(prev_embedding, list) and prev_embedding:
                embedding = prev_embedding
                cached += 1
        if not embedding:
            embeddings_to_compute.append(truncated)
            pending_indices.append(len(items))

        clip = IndexedClip(
            path=rel_path,
            text=truncated,
            embedding=embedding,
            content_hash=content_hash,
            token_estimate=token_estimate,
        )
        items.append(clip)

    if embeddings_to_compute:
        new_embeddings = _embed_texts(embeddings_to_compute)
        for idx, vector in zip(pending_indices, new_embeddings):
            items[idx].embedding = vector

    index = {
        "theme": theme_slug,
        "model": DEFAULT_MODEL,
        "max_tokens_est": MAX_TOKENS_PER_CLIP,
        "clips": [],
    }

    for item in items:
        clip_dict = item.as_dict()
        clip_dict["content_hash"] = getattr(item, "content_hash", "")
        clip_dict["token_estimate"] = getattr(item, "token_estimate", estimate_tokens(item.text))
        index["clips"].append(clip_dict)

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    clip_count = len(items)
    avg_length = sum(lengths) / clip_count if clip_count else 0.0
    avg_tokens = sum(token_lengths) / clip_count if clip_count else 0.0
    print(f"Indexed {clip_count} clips for theme '{theme_slug}'.")
    print(f"Average truncated length: {avg_length:.1f} words; {avg_tokens:.1f} est. tokens.")
    if cached:
        print(f"cached: {cached}")

    return {
        "theme": theme_slug,
        "index_path": index_path.as_posix(),
        "clips": clip_count,
        "avg_truncated_words": round(avg_length, 1),
        "avg_truncated_tokens_est": round(avg_tokens, 1),
        "cached": cached,
    }


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
        scored.append(
            {
                "path": clip.get("path"),
                "text": clip.get("text", ""),
                "score": float(score),
                "token_estimate": int(clip.get("token_estimate") or estimate_tokens(clip.get("text", ""))),
            }
        )

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
