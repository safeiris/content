# -*- coding: utf-8 -*-

import os


def _env_int(name: str, default: int) -> int:
    value = str(os.getenv(name, "")).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float_list(name: str, default: str) -> tuple[float, ...]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        raw = default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    delays = []
    for part in parts:
        try:
            delays.append(float(part))
        except ValueError:
            continue
    if not delays:
        delays = [float(value) for value in default.split(",") if value]
    return tuple(delays)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off", "no"}

OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY", "")).strip()

USE_MOCK_LLM = _env_bool("USE_MOCK_LLM", False)
OFFLINE_MODE = _env_bool("OFFLINE_MODE", False)
PIPELINE_FAST_PATH = _env_bool("PIPELINE_FAST_PATH", False)
MODEL_PROVIDER = str(os.getenv("MODEL_PROVIDER", "openai")).strip() or "openai"

_FORCE_MODEL_RAW = str(os.getenv("FORCE_MODEL", os.getenv("LLM_FORCE_MODEL", "false"))).strip().lower()
FORCE_MODEL = _FORCE_MODEL_RAW in {"1", "true", "yes", "on"}

# GPT-5 Responses tuning
G5_MAX_OUTPUT_TOKENS_BASE = _env_int("G5_MAX_OUTPUT_TOKENS_BASE", 1500)
G5_MAX_OUTPUT_TOKENS_STEP1 = _env_int("G5_MAX_OUTPUT_TOKENS_STEP1", 2200)
G5_MAX_OUTPUT_TOKENS_STEP2 = _env_int("G5_MAX_OUTPUT_TOKENS_STEP2", 2600)
G5_MAX_OUTPUT_TOKENS_MAX = _env_int("G5_MAX_OUTPUT_TOKENS_MAX", 3000)
_DEFAULT_POLL_DELAYS = "0.3,0.6,1.0,1.5"
G5_POLL_INTERVALS = _env_float_list("G5_POLL_INTERVALS", _DEFAULT_POLL_DELAYS)
G5_POLL_MAX_ATTEMPTS = _env_int("G5_POLL_MAX_ATTEMPTS", len(G5_POLL_INTERVALS))
G5_ENABLE_PREVIOUS_ID_FETCH = _env_bool("G5_ENABLE_PREVIOUS_ID_FETCH", True)

# Дефолтные настройки ядра
DEFAULT_TONE = "экспертный, дружелюбный"
DEFAULT_STRUCTURE = ["Введение", "Основная часть", "FAQ", "Вывод"]

# Простая «норма» для SEO: ориентир по упоминаниям ключей на ~100 слов
DEFAULT_SEO_DENSITY = 2

# Рекомендуемые границы объёма (знаков)
DEFAULT_MIN_LENGTH = 3500
DEFAULT_MAX_LENGTH = 6000

# Максимальный объём пользовательского контекста (символов)
MAX_CUSTOM_CONTEXT_CHARS = 20_000

# Стилевые профили
STYLE_PROFILE_PATH = "profiles/finance/style_profile.md"
STYLE_PROFILE_VARIANT = str(os.getenv("STYLE_PROFILE_VARIANT", "full")).strip().lower() or "full"
if STYLE_PROFILE_VARIANT not in {"full", "light"}:
    STYLE_PROFILE_VARIANT = "full"
APPEND_STYLE_PROFILE_DEFAULT = (
    str(os.getenv("APPEND_STYLE_PROFILE_DEFAULT", "true")).strip().lower() not in {"0", "false", "off", "no"}
)

# Ключевые слова
KEYWORDS_ALLOW_AUTO = _env_bool("KEYWORDS_ALLOW_AUTO", False)
