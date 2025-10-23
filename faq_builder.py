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
        answer = answers[idx]
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
    if provided_entries:
        for idx, entry in enumerate(provided_entries, start=1):
            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()
            anchor = str(entry.get("anchor") or _sanitize_anchor(question))
            if not question or not answer:
                continue
            entries.append(FaqEntry(question=question, answer=answer, anchor=anchor))
    if len(entries) < 5:
        extra = _generate_generic_entries(topic, list(keywords))
        needed = 5 - len(entries)
        entries.extend(extra[:needed])
    entries = entries[:5]

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
    return FaqBuildResult(text=merged, entries=entries, jsonld=jsonld)
