# -*- coding: utf-8 -*-
"""End-to-end pipeline: assemble prompt → call LLM → store artefacts."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from assemble_messages import assemble_messages, retrieve_exemplars
from llm_client import DEFAULT_MODEL, generate as llm_generate
from plagiarism_guard import is_too_similar


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an article using the configured LLM.")
    parser.add_argument("--theme", required=True, help="Theme slug (matches profiles/<theme>/...)")
    parser.add_argument("--data", required=True, help="Path to the JSON brief with generation parameters.")
    parser.add_argument(
        "--outfile",
        help="Optional path for the resulting markdown. Defaults to artifacts/<timestamp>__<theme>__article.md",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of exemplar clips to attach to CONTEXT (default: 3).")
    parser.add_argument("--model", help="Override model name (otherwise uses LLM_MODEL env or default).")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3).")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1400,
        dest="max_tokens",
        help="Max tokens for generation (default: 1400).",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per request in seconds (default: 60).")
    return parser.parse_args()


def _load_input(path: str) -> Dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Не найден файл входных данных: {payload_path}")
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {payload_path}: {exc}") from exc


def _resolve_model(cli_model: str | None) -> str:
    candidate = (cli_model or os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()
    return candidate or DEFAULT_MODEL


def _make_output_path(theme: str, outfile: str | None) -> Path:
    if outfile:
        return Path(outfile)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    filename = f"{timestamp}__{theme}__article.md"
    return Path("artifacts") / filename


def _write_outputs(markdown_path: Path, text: str, metadata: Dict[str, Any]) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text, encoding="utf-8")
    metadata_path = markdown_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarise(theme: str, k: int, model: str, text: str) -> None:
    chars = len(text)
    words = len(text.split()) if text.strip() else 0
    print(f"[orchestrate] theme={theme} k={k} model={model} length={chars} chars / {words} words")


def main() -> None:
    args = _parse_args()

    start_time = time.time()
    data = _load_input(args.data)

    model_name = _resolve_model(args.model)
    messages = assemble_messages(data_path=args.data, theme_slug=args.theme, k=args.k)
    exemplars = retrieve_exemplars(theme_slug=args.theme, query=data.get("theme", ""), k=args.k)
    clip_texts = [str(item.get("text", "")) for item in exemplars if item.get("text")]

    article_text = llm_generate(
        messages,
        model=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
    )

    retry_used = False
    retry_messages: List[Dict[str, str]] | None = None
    flagged = is_too_similar(article_text, clip_texts)
    if flagged:
        retry_used = True
        retry_messages = list(messages)
        retry_messages.append(
            {
                "role": "user",
                "content": "Перефразируй разделы, добавь списки и FAQ, избегай совпадений с примерами.",
            }
        )
        print("[orchestrate] Обнаружено совпадение с примерами, выполняю перегенерацию...", file=sys.stderr)
        article_text = llm_generate(
            retry_messages,
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
        )

    duration = time.time() - start_time
    markdown_path = _make_output_path(args.theme, args.outfile)
    messages_count = len(retry_messages) if retry_messages is not None else len(messages)

    metadata: Dict[str, Any] = {
        "theme": args.theme,
        "data_path": str(Path(args.data).resolve()),
        "model": model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout,
        "retrieval_k": args.k,
        "clips": [
            {"path": item.get("path"), "score": item.get("score")}
            for item in exemplars
        ],
        "plagiarism_flagged": flagged,
        "retry_used": retry_used,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "duration_seconds": round(duration, 3),
        "characters": len(article_text),
        "words": len(article_text.split()) if article_text.strip() else 0,
        "messages_count": messages_count,
    }

    _write_outputs(markdown_path, article_text, metadata)
    _summarise(args.theme, args.k, model_name, article_text)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
