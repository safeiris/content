# -*- coding: utf-8 -*-

import os

# Тестовые ключи для учебной лаборатории
_DEFAULT_OPENAI_API_KEY = (
    "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"
)
OPENAI_API_KEY = (
    str(os.getenv("OPENAI_API_KEY", _DEFAULT_OPENAI_API_KEY)).strip() or _DEFAULT_OPENAI_API_KEY
)

# Дефолтные настройки ядра
DEFAULT_TONE = "экспертный, дружелюбный"
DEFAULT_STRUCTURE = ["Введение", "Основная часть", "FAQ", "Вывод"]

# Простая «норма» для SEO: ориентир по упоминаниям ключей на ~100 слов
DEFAULT_SEO_DENSITY = 2

# Рекомендуемые границы объёма (знаков)
DEFAULT_MIN_LENGTH = 1500
DEFAULT_MAX_LENGTH = 7000

# Стилевые профили
STYLE_PROFILE_PATH = "profiles/finance/style_profile.md"
STYLE_PROFILE_VARIANT = str(os.getenv("STYLE_PROFILE_VARIANT", "full")).strip().lower() or "full"
if STYLE_PROFILE_VARIANT not in {"full", "light"}:
    STYLE_PROFILE_VARIANT = "full"
APPEND_STYLE_PROFILE_DEFAULT = (
    str(os.getenv("APPEND_STYLE_PROFILE_DEFAULT", "true")).strip().lower() not in {"0", "false", "off", "no"}
)

