# -*- coding: utf-8 -*-
"""Utility to persist the assembled system prompt for manual review."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rules_engine import build_prompt


def export_prompt(data_path: str = "input_example.json", artifacts_dir: str = "artifacts") -> Path:
    payload_path = Path(data_path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Input file not found: {payload_path}")

    data = json.loads(payload_path.read_text(encoding="utf-8"))
    prompt = build_prompt(data)

    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"prompt_{timestamp}.txt"
    file_path.write_text(prompt, encoding="utf-8")
    return file_path


if __name__ == "__main__":
    path = export_prompt()
    print(f"Prompt saved to {path}")
