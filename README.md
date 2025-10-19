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
- `docs/implementation_overview.md` — техническая документация с обзором архитектуры и зависимостей.

## Быстрый старт
```bash
python3 -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
# venv\Scripts\activate

python run_demo.py

```

## RAG-ready (Stage 1)

### Profiles layout
- `profiles/<theme>/exemplars/` — тематические фрагменты по 200–700 слов (одно ключевое сообщение на файл, нейтральный Markdown). Пример: `profiles/finance/exemplars/{001_intro.md, 002_benefits.md, 003_faq.md}`.
- `profiles/<theme>/glossary.txt` — короткие термины и определения одной строкой.
- `profiles/<theme>/style_guide.md` — тон, do/don't, политика заголовков и локализации.

### Оркестрация без LLM
- `assemble_messages.py` — собирает системный промпт и добавляет `CONTEXT`, когда `retrieve_exemplars()` начнёт возвращать фрагменты.
- `retrieve_exemplars(theme_slug, query, k=3)` — контракт Stage 2: должен отдавать список словарей вида `{ "path": str, "text": str, "score": float }`, подобранных под токен-бюджет (~500–700 токенов суммарно).
- Итоговый формат сообщений: `[{"role": "system", ...}, опциональный CONTEXT, {"role": "user", "content": "Сгенерируй текст по указанным параметрам."}]`.

### Тестирование Stage 1
- `python run_demo.py` — базовый поток (без изменений).
- `python validate_prompt.py` — проверка наличия секций и ключевых слов в промпте.
- `python assemble_messages.py` — печатает предпросмотр system и CONTEXT (пока пусто).
- `python prompt_to_file.py` — сохраняет промпт в `./artifacts/prompt_<timestamp>.txt`.

### Сквозной прогон с генерацией статьи
1. Собери индекс (один раз или при изменении клипов): `python -m retrieval index --theme finance`
2. Запусти генерацию: `python orchestrate.py --theme finance --data input_example.json --k 3`
3. Результат смотри в `artifacts/*.md` и соседнем `.json` с метаданными.

### Ready for Stage 2, когда
- Структура профилей создана минимум для одной темы с 2–3 заготовками.
- `base_prompt.txt` знает про блок CONTEXT.
- Существуют стабы `retrieve_exemplars` и `assemble_messages` без внешних зависимостей.
- В проект не добавлены внешние библиотеки и сетевые вызовы.

