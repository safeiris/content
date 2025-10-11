# -*- coding: utf-8 -*-
"""Minimal validator for Stage 1 prompt scaffolding."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rules_engine import build_prompt


def _load_input(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def validate_prompt(data_path: str = "input_example.json") -> None:
    payload_path = Path(data_path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Input file not found: {payload_path}")

    data = _load_input(payload_path)
    prompt = build_prompt(data)

    provided_sections: List[str] = data.get("structure", []) or []
    missing_sections = [section for section in provided_sections if f"- {section}" not in prompt]

    keywords: List[str] = [kw.strip() for kw in data.get("keywords", []) if isinstance(kw, str) and kw.strip()]
    lower_prompt = prompt.lower()
    missing_keywords = [kw for kw in keywords if kw.lower() not in lower_prompt]

    print("=== Prompt Validation ===")
    if missing_sections:
        print(f"Warnings: missing sections in prompt -> {', '.join(missing_sections)}")
    else:
        print("Sections: OK — все заявленные блоки присутствуют.")

    if missing_keywords:
        print(f"Warnings: keywords not surfaced -> {', '.join(missing_keywords)}")
    else:
        print("Keywords: OK — все ключевые слова упомянуты в инструкциях.")

    print("\nPreview:")
    print(_preview(prompt))


if __name__ == "__main__":
    validate_prompt()
