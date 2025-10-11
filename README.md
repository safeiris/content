# Ядро генерации — Этап 1 (демо без UI)

## Что это
Базовый модуль, который принимает параметры (тема, цель, ключевые слова, тон, структура, объём)
и собирает системный промпт для модели.

## Файлы
- `config.py` — дефолтные настройки.
- `helpers.py` — утилиты форматирования.
- `rules_engine.py` — сборка промпта (`build_prompt(data)`).
- `base_prompt.txt` — шаблон промпта.
- `input_example.json` — пример входных данных.
- `run_demo.py` — проверка без UI.

## Быстрый старт
```bash
python3 -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
# venv\Scripts\activate

python run_demo.py

