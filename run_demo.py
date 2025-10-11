# -*- coding: utf-8 -*-
import json
from pathlib import Path
from rules_engine import build_prompt

def main():
    demo_path = Path(__file__).resolve().parent / "input_example.json"
    if not demo_path.exists():
        raise FileNotFoundError(f"Не найден {demo_path}")
    data = json.loads(demo_path.read_text(encoding="utf-8"))
    prompt = build_prompt(data)

    print("=" * 80)
    print("СФОРМИРОВАННЫЙ СИСТЕМНЫЙ ПРОМПТ")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print("Как тестировать без UI:")
    print("- Измените input_example.json (tone/keywords/structure/объём)")
    print("- Повторно запустите: python run_demo.py")

if __name__ == "__main__":
    main()

