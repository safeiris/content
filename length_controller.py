from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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


def ensure_article_length(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    protected_blocks: Iterable[str] | None = None,
    max_iterations: int = 6,
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
            return LengthControllerResult(
                text=current_text,
                length=length_now,
                iterations=iterations,
                adjusted=adjusted,
                success=True,
                history=tuple(history),
            )

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

    final_length = length_no_spaces(current_text)
    return LengthControllerResult(
        text=current_text,
        length=final_length,
        iterations=iterations,
        adjusted=adjusted,
        success=False,
        history=tuple(history),
        failure_reason=failure_reason,
    )


__all__ = ["LengthControllerResult", "ensure_article_length"]
