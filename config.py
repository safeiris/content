# -*- coding: utf-8 -*-

import os

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
APPEND_STYLE_PROFILE_DEFAULT = (
    str(os.getenv("APPEND_STYLE_PROFILE_DEFAULT", "true")).strip().lower() not in {"0", "false", "off", "no"}
)

