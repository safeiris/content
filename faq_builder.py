from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class FaqEntry:
    question: str
    answer: str
    anchor: str


@dataclass
class FaqBuildResult:
    text: str
    entries: List[FaqEntry]
    jsonld: str


def _sanitize_anchor(text: str) -> str:
    return "-" + "-".join(text.lower().split())


def _normalize_answer(answer: str) -> str:
    paragraphs = [part.strip() for part in answer.split("\n\n") if part.strip()]
    if not paragraphs:
        raise ValueError("Ответ пустой")
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:3]
    if any(len(p) < 20 for p in paragraphs):
        raise ValueError("Ответ слишком короткий")
    return "\n\n".join(paragraphs)


def _normalize_question(question: str, seen: set[str]) -> str:
    normalized = question.strip()
    if not normalized:
        raise ValueError("Вопрос пустой")
    if normalized.lower() in seen:
        raise ValueError("Дублирующийся вопрос")
    seen.add(normalized.lower())
    return normalized


def _normalize_entry(raw: Dict[str, str], seen: set[str]) -> FaqEntry:
    question_raw = raw.get("question") if "question" in raw else raw.get("q")
    answer_raw = raw.get("answer") if "answer" in raw else raw.get("a")
    question = _normalize_question(str(question_raw or ""), seen)
    answer = _normalize_answer(str(answer_raw or ""))
    anchor = str(raw.get("anchor") or _sanitize_anchor(question))
    return FaqEntry(question=question, answer=answer, anchor=anchor)


def _generate_generic_entries(topic: str, keywords: Sequence[str]) -> List[FaqEntry]:
    base_topic = topic or "теме"
    key_iter = list(keywords)[:5]
    templates = [
        "Как оценить основные риски, связанные с {topic}?",
        "Какие шаги помогут подготовиться к решению вопросов по {topic}?",
        "Какие цифры считать ориентиром, когда речь заходит о {topic}?",
        "Как использовать программы поддержки, если речь идёт о {topic}?",
        "Что делать, если ситуация с {topic} резко меняется?",
    ]
    answers = [
        "Начните с базовой диагностики: опишите текущую ситуацию, посчитайте ключевые показатели и зафиксируйте цели. "
        "Далее сопоставьте результаты с отраслевыми нормами и составьте план коррекции.",
        "Сформируйте пошаговый чек-лист. Включите в него анализ документов, консультации с экспертами и список сервисов, которые помогут собрать данные. "
        "По мере продвижения фиксируйте выводы, чтобы вернуться к ним на этапе принятия решения.",
        "Используйте диапазон значений из методических материалов и банковской аналитики. "
        "Сравните собственные показатели с усреднёнными и определите пороги, при которых стоит пересмотреть стратегию.",
        "Изучите федеральные и региональные программы, подходящие под ваш профиль. "
        "Составьте список требований, подготовьте пакет документов и оцените сроки рассмотрения, чтобы не потерять время.",
        "Создайте резервный план действий: определите, какие параметры контролировать ежемесячно, и заранее договоритесь о точках проверки. "
        "Если изменения превышают допустимый порог, инициируйте пересмотр стратегии и подключите независимую экспертизу.",
    ]

    entries: List[FaqEntry] = []
    for idx in range(5):
        keyword_hint = key_iter[idx] if idx < len(key_iter) else ""
        question = templates[idx].format(topic=base_topic)
        if keyword_hint:
            question = f"{question[:-1]} и {keyword_hint}?"
        answer = _normalize_answer(answers[idx])
        anchor = _sanitize_anchor(question)
        entries.append(FaqEntry(question=question, answer=answer, anchor=anchor))
    return entries


def _render_markdown(entries: Sequence[FaqEntry]) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        lines.append(f"**Вопрос {idx}.** {entry.question}")
        lines.append(
            "**Ответ.** "
            + entry.answer
            + " Это помогает не только понять детали, но и оформить решение в рабочем формате."
        )
        lines.append("")
    return "\n".join(lines).strip()


def _build_jsonld(entries: Sequence[FaqEntry]) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": entry.question,
                "acceptedAnswer": {"@type": "Answer", "text": entry.answer},
            }
            for entry in entries
        ],
    }
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'<script type="application/ld+json">\n{compact}\n</script>'


def build_faq_block(
    *,
    base_text: str,
    topic: str,
    keywords: Iterable[str],
    provided_entries: Sequence[Dict[str, str]] | None = None,
) -> FaqBuildResult:
    entries: List[FaqEntry] = []
    seen_questions: set[str] = set()
    if provided_entries:
        for entry in provided_entries:
            try:
                normalized = _normalize_entry(entry, seen_questions)
            except ValueError:
                continue
            entries.append(normalized)
            if len(entries) == 5:
                break
    if len(entries) < 5:
        for candidate in _generate_generic_entries(topic, list(keywords)):
            if len(entries) == 5:
                break
            if candidate.question.lower() in seen_questions:
                continue
            seen_questions.add(candidate.question.lower())
            entries.append(candidate)

    if len(entries) != 5:
        raise ValueError("Не удалось собрать пять валидных вопросов для FAQ")

    rendered = _render_markdown(entries)
    placeholder = "<!--FAQ_START-->"
    end_placeholder = "<!--FAQ_END-->"
    if placeholder not in base_text or end_placeholder not in base_text:
        raise ValueError("FAQ placeholder missing in base text")

    before, remainder = base_text.split(placeholder, 1)
    inside, after = remainder.split(end_placeholder, 1)
    inside = inside.strip()
    if inside:
        rendered = f"{inside}\n\n{rendered}".strip()
    merged = f"{before}{placeholder}\n{rendered}\n{end_placeholder}{after}"
    jsonld = _build_jsonld(entries)
    # JSON-LD валидация
    try:
        raw_json = jsonld.split("\n", 1)[1].rsplit("\n", 1)[0]
        payload = json.loads(raw_json)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Некорректный JSON-LD FAQ: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("@type") != "FAQPage" or len(payload.get("mainEntity", [])) != 5:
        raise ValueError("JSON-LD FAQ не соответствует схеме FAQPage")
    return FaqBuildResult(text=merged, entries=entries, jsonld=jsonld)
