from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from keyword_injector import LOCK_END, LOCK_START_TEMPLATE
from length_trimmer import TrimValidationError, trim_text
from validators import length_no_spaces

_FAQ_START = "<!--FAQ_START-->"
_JSONLD_PATTERN = re.compile(r"<script\s+type=\"application/ld\+json\">.*?</script>", re.DOTALL)

_FILLER_PARAGRAPHS: Tuple[str, ...] = (
    (
        "**Дополнительное пояснение.** Раскройте, как читатель может применить рекомендации в ближайшие дни, "
        "какие инструменты выбрать и как оценить первые результаты. Укажите контрольные точки, чтобы пользователь "
        "понимал, что продвижение идёт по плану."
        "\n\n"
        "**Практический пример.** Опишите типовой сценарий, когда команда сталкивается с нехваткой ресурсов. Поясните, "
        "как приоритизировать задачи, какие риски отслеживать и как быстро скорректировать стратегию, если показатели "
        "проседают."
    ),
    (
        "**Краткая памятка.** Сформулируйте три шага, которые можно выполнить сразу после прочтения статьи, и добавьте "
        "подсказки по коммуникации с коллегами или подрядчиком. Напомните, какие метрики проверять через неделю и через "
        "месяц, чтобы убедиться в эффективности."
        "\n\n"
        "**Мотивационный акцент.** Укрепите уверенность читателя, напомнив, что даже небольшие итерации дают накопительный "
        "эффект. Укажите, как документировать выводы и когда стоит возвращаться к материалу за дополнительными идеями."
    ),
)


@dataclass(frozen=True)
class LengthControllerResult:
    text: str
    length: int
    iterations: int
    adjusted: bool
    success: bool
    history: Tuple[int, ...]
    failure_reason: Optional[str] = None


def _split_jsonld_block(text: str) -> Tuple[str, str]:
    match = _JSONLD_PATTERN.search(text)
    if not match:
        return text, ""
    jsonld_block = match.group(0)
    before = text[: match.start()].rstrip()
    after = text[match.end() :].lstrip()
    article = before
    if after:
        article = f"{article}\n\n{after}" if article else after
    return article, jsonld_block.strip()


def _append_filler_block(text: str, filler: str) -> str:
    article, jsonld = _split_jsonld_block(text)
    insert_pos = article.find(_FAQ_START)
    if insert_pos < 0:
        insert_pos = len(article)
    before = article[:insert_pos].rstrip()
    after = article[insert_pos:].lstrip()
    filler_clean = filler.strip()
    pieces: List[str] = []
    if before:
        pieces.append(before)
    if filler_clean:
        pieces.append(filler_clean)
    if after:
        pieces.append(after)
    combined = "\n\n".join(pieces).rstrip() + "\n"
    if jsonld:
        combined = f"{combined.rstrip()}\n\n{jsonld}\n"
    return combined


def _select_filler(iteration: int) -> str:
    if not _FILLER_PARAGRAPHS:
        return ""
    index = iteration % len(_FILLER_PARAGRAPHS)
    return _FILLER_PARAGRAPHS[index]


_PRECISION_WORDS: Tuple[str, ...] = (
    "подробности",
    "пример",
    "детали",
    "вывод",
    "анализ",
    "решение",
    "инициатива",
    "практика",
)


def _build_precision_filler(delta: int) -> str:
    if delta <= 0:
        return ""
    target_core = max(0, delta - 1)
    collected: List[str] = []
    total = 0
    index = 0
    while total < target_core:
        word = _PRECISION_WORDS[index % len(_PRECISION_WORDS)] if _PRECISION_WORDS else "слово"
        index += 1
        clean = word.replace(" ", "")
        remaining = target_core - total
        if len(clean) > remaining:
            clean = clean[:remaining]
        collected.append(clean)
        total += len(clean)
    body = " ".join(item for item in collected if item).strip()
    if body:
        return f" {body}."
    return " ."


def _build_protection_mask(text: str, tokens: Iterable[str]) -> List[bool]:
    mask = [False] * len(text)
    for token in tokens:
        if not token:
            continue
        start = 0
        while True:
            idx = text.find(token, start)
            if idx == -1:
                break
            for pos in range(idx, min(idx + len(token), len(mask))):
                mask[pos] = True
            start = idx + len(token)
    return mask


def _truncate_to_exact_length(text: str, remove: int, protected: Iterable[str]) -> Optional[str]:
    if remove <= 0:
        return text
    article, jsonld = _split_jsonld_block(text)
    if not article.strip():
        return None
    tokens = [LOCK_START_TEMPLATE.format(term=term) for term in protected]
    tokens.extend([LOCK_END, _FAQ_START, "<!--FAQ_END-->", "## FAQ", "## Заключение"])
    tokens.extend(["**Вопрос", "**Ответ"])
    mask = _build_protection_mask(article, tokens)
    buffer: List[str] = []
    removed = 0
    for idx in range(len(article) - 1, -1, -1):
        char = article[idx]
        if removed >= remove:
            buffer.append(char)
            continue
        if mask[idx]:
            buffer.append(char)
            continue
        if char.isspace():
            buffer.append(char)
            continue
        removed += 1
    if removed < remove:
        return None
    rebuilt = "".join(reversed(buffer)).rstrip() + "\n"
    if jsonld:
        rebuilt = f"{rebuilt.rstrip()}\n\n{jsonld}\n"
    return rebuilt


def ensure_article_length(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    protected_blocks: Iterable[str] | None = None,
    max_iterations: int = 6,
    faq_expected: int = 5,
    exact_chars: Optional[int] = None,
) -> LengthControllerResult:
    """Ensure that article text fits the requested length range.

    The controller alternates between trimming (when the article is too long)
    and appending structured filler paragraphs (when the article is too short).
    It never raises ``TrimValidationError`` and always returns the best effort
    result, marking ``success`` when the final length falls inside the range.
    """

    protected = [str(term).strip() for term in (protected_blocks or []) if str(term).strip()]
    current_text = text
    history: List[int] = []
    iterations = 0
    adjusted = False
    failure_reason: Optional[str] = None

    current_min = max(0, int(min_chars))
    current_max = max(current_min, int(max_chars))

    while True:
        length_now = length_no_spaces(current_text)
        history.append(length_now)
        if current_min <= length_now <= current_max:
            break

        if iterations >= max_iterations:
            failure_reason = failure_reason or "max_iterations"
            break

        adjusted = True
        iterations += 1

        if length_now > current_max:
            try:
                trimmed = trim_text(
                    current_text,
                    min_chars=current_min,
                    max_chars=current_max,
                    protected_blocks=protected,
                    faq_expected=faq_expected,
                )
            except TrimValidationError as exc:  # pragma: no cover - defensive branch
                failure_reason = f"trim_failed:{exc}" if not failure_reason else failure_reason
                break

            new_length = length_no_spaces(trimmed.text)
            if new_length >= length_now:
                if current_max > current_min:
                    current_max = max(current_min, current_max - 50)
                    continue
                failure_reason = failure_reason or "trim_stalled"
                current_text = trimmed.text
                break

            current_text = trimmed.text
            continue

        filler = _select_filler(iterations - 1)
        if not filler:
            failure_reason = failure_reason or "no_filler"
            break
        current_text = _append_filler_block(current_text, filler)

    target_exact = exact_chars if exact_chars is not None else None
    final_length = length_no_spaces(current_text)
    if target_exact is not None:
        target_value = max(0, int(target_exact))
        delta = target_value - final_length
        if delta > 0:
            filler = _build_precision_filler(delta)
            if filler:
                current_text = _append_filler_block(current_text, filler)
                final_length = length_no_spaces(current_text)
                history.append(final_length)
                delta = target_value - final_length
        if delta < 0:
            trimmed_exact = _truncate_to_exact_length(current_text, -delta, protected)
            if trimmed_exact:
                current_text = trimmed_exact
                final_length = length_no_spaces(current_text)
                history.append(final_length)
            else:
                failure_reason = failure_reason or "exact_trim_failed"
        if target_value == final_length:
            current_min = current_max = target_value
    if history:
        history[-1] = final_length
    success = current_min <= final_length <= current_max
    return LengthControllerResult(
        text=current_text,
        length=final_length,
        iterations=iterations,
        adjusted=adjusted,
        success=success,
        history=tuple(history),
        failure_reason=failure_reason,
    )


__all__ = ["LengthControllerResult", "ensure_article_length"]
