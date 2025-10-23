from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional

from faq_builder import FaqBuildResult, build_faq_block
from keyword_injector import KeywordInjectionResult, build_term_pattern, inject_keywords
from length_trimmer import TrimResult, trim_text
from validators import ValidationResult, strip_jsonld, validate_article


class PipelineStep(str, Enum):
    SKELETON = "skeleton"
    KEYWORDS = "keywords"
    FAQ = "faq"
    TRIM = "trim"


@dataclass
class PipelineLogEntry:
    step: PipelineStep
    started_at: float
    finished_at: Optional[float] = None
    notes: Dict[str, object] = field(default_factory=dict)
    status: str = "pending"


@dataclass
class PipelineState:
    text: str
    jsonld: Optional[str]
    validation: Optional[ValidationResult]
    logs: List[PipelineLogEntry]
    checkpoints: Dict[PipelineStep, str]


class DeterministicPipeline:
    def __init__(
        self,
        *,
        topic: str,
        base_outline: List[str],
        keywords: Iterable[str],
        min_chars: int,
        max_chars: int,
        provided_faq: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.topic = topic
        self.base_outline = base_outline or ["Введение", "Основная часть", "Вывод"]
        self.keywords = list(keywords)
        self.normalized_keywords = [
            term
            for term in (str(item).strip() for item in self.keywords)
            if term
        ]
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.provided_faq = provided_faq or []
        self.logs: List[PipelineLogEntry] = []
        self.checkpoints: Dict[PipelineStep, str] = {}
        self.jsonld: Optional[str] = None
        self.locked_terms: List[str] = []
        self.jsonld_reserve: int = 0

    # Step helpers -----------------------------------------------------
    def _log(self, step: PipelineStep, status: str, **notes: object) -> None:
        entry = PipelineLogEntry(step=step, started_at=time.time(), status=status, notes=dict(notes))
        self.logs.append(entry)

    def _update_log(self, step: PipelineStep, status: str, **notes: object) -> None:
        for entry in reversed(self.logs):
            if entry.step == step:
                entry.status = status
                entry.finished_at = time.time()
                entry.notes.update(notes)
                return
        self.logs.append(
            PipelineLogEntry(step=step, started_at=time.time(), finished_at=time.time(), status=status, notes=dict(notes))
        )

    # Step implementations --------------------------------------------
    def _run_skeleton(self) -> str:
        self._log(PipelineStep.SKELETON, "running")
        intro = self._render_intro()
        body = self._render_body()
        outro = self._render_outro()
        faq_placeholder = "\n\n## FAQ\n\n<!--FAQ_START-->\n<!--FAQ_END-->\n\n"
        skeleton = f"## Введение\n\n{intro}\n\n## Основная часть\n\n{body}{faq_placeholder}## Вывод\n\n{outro}"
        self._update_log(
            PipelineStep.SKELETON,
            "ok",
            length=len(skeleton),
            **self._metrics(skeleton),
        )
        self.checkpoints[PipelineStep.SKELETON] = skeleton
        return skeleton

    def _render_intro(self) -> str:
        sentences = [
            f"{self.topic} влияет на финансовое здоровье семьи, поэтому начинаем с оценки текущей нагрузки и распределения платежей.",
            "Мы фиксируем ключевые показатели, объясняем критерии допустимых значений и даём быстрые советы по сбору данных.",
        ]
        return " ".join(sentences)

    def _render_body(self) -> str:
        if not self.base_outline:
            sections = ["Понимаем входные данные", "Разбираем метрики", "Готовим решения"]
        else:
            sections = [title for title in self.base_outline if title.lower() not in {"введение", "faq", "вывод"}]
            if not sections:
                sections = ["Понимаем входные данные", "Разбираем метрики", "Готовим решения"]
        paragraphs: List[str] = []
        for heading in sections:
            paragraphs.append(f"### {heading}")
            paragraphs.append(self._render_section_block(heading))
        return "\n\n".join(paragraphs)

    def _render_section_block(self, heading: str) -> str:
        sentences = [
            f"{heading} рассматриваем на реальных примерах, чтобы показать связь между цифрами и бытовыми решениями семьи.",
            "Отмечаем юридические нюансы, возможные риски и добавляем чек-лист действий, который можно выполнять по шагам.",
            "В конце указываем цифровые сервисы для автоматизации расчётов и напоминаний, чтобы снизить вероятность ошибок.",
        ]
        return " ".join(sentences)

    def _render_outro(self) -> str:
        sentences = [
            "В выводах собираем план действий, назначаем контрольные даты и распределяем ответственность между участниками.",
            "Дополняем материал рекомендациями по пересмотру стратегии и фиксируем признаки, при которых стоит обратиться к экспертам.",
        ]
        return " ".join(sentences)

    def _run_keywords(self, text: str) -> KeywordInjectionResult:
        self._log(PipelineStep.KEYWORDS, "running")
        result = inject_keywords(text, self.keywords)
        self.locked_terms = list(result.locked_terms)
        self._update_log(
            PipelineStep.KEYWORDS,
            "ok",
            coverage=result.coverage,
            inserted_section=result.inserted_section,
            **self._metrics(result.text),
        )
        self.checkpoints[PipelineStep.KEYWORDS] = result.text
        return result

    def _run_faq(self, text: str) -> FaqBuildResult:
        self._log(PipelineStep.FAQ, "running")
        faq_result = build_faq_block(
            base_text=text,
            topic=self.topic,
            keywords=self.keywords,
            provided_entries=self.provided_faq,
        )
        self.jsonld = faq_result.jsonld
        self.jsonld_reserve = len("".join(self.jsonld.split())) if self.jsonld else 0
        self._update_log(
            PipelineStep.FAQ,
            "ok",
            entries=[entry.question for entry in faq_result.entries],
            **self._metrics(faq_result.text),
        )
        self.checkpoints[PipelineStep.FAQ] = faq_result.text
        return faq_result

    def _run_trim(self, text: str) -> TrimResult:
        self._log(PipelineStep.TRIM, "running")
        reserve = self.jsonld_reserve if self.jsonld else 0
        target_max = max(self.min_chars, self.max_chars - reserve)
        result = trim_text(
            text,
            min_chars=self.min_chars,
            max_chars=target_max,
            protected_blocks=self.locked_terms,
        )
        self._update_log(
            PipelineStep.TRIM,
            "ok",
            removed=len(result.removed_paragraphs),
            **self._metrics(result.text),
        )
        self.checkpoints[PipelineStep.TRIM] = result.text
        return result

    # Public API -------------------------------------------------------
    def run(self) -> PipelineState:
        text = self._run_skeleton()
        keyword_result = self._run_keywords(text)
        faq_result = self._run_faq(keyword_result.text)
        trim_result = self._run_trim(faq_result.text)
        combined_text = trim_result.text
        if self.jsonld:
            combined_text = f"{combined_text.rstrip()}\n\n{self.jsonld}\n"
        validation = validate_article(
            combined_text,
            keywords=self.keywords,
            min_chars=self.min_chars,
            max_chars=self.max_chars,
        )
        self.logs.append(
            PipelineLogEntry(
                step=PipelineStep.TRIM,
                started_at=time.time(),
                finished_at=time.time(),
                status="validated" if validation.is_valid else "failed",
                notes={"stats": validation.stats, **self._metrics(combined_text)},
            )
        )
        return PipelineState(
            text=combined_text,
            jsonld=self.jsonld,
            validation=validation,
            logs=self.logs,
            checkpoints=self.checkpoints,
        )

    def resume(self, from_step: PipelineStep) -> PipelineState:
        order = [PipelineStep.SKELETON, PipelineStep.KEYWORDS, PipelineStep.FAQ, PipelineStep.TRIM]
        if from_step == PipelineStep.SKELETON:
            return self.run()

        requested_index = order.index(from_step)
        base_index = requested_index - 1
        fallback_index = base_index
        while fallback_index >= 0 and order[fallback_index] not in self.checkpoints:
            fallback_index -= 1

        if fallback_index < 0:
            message = (
                f"Чекпоинты отсутствуют для шага {from_step.value}. Полный перезапуск пайплайна."
            )
            self.logs.append(
                PipelineLogEntry(
                    step=from_step,
                    started_at=time.time(),
                    finished_at=time.time(),
                    status="error",
                    notes={"message": message},
                )
            )
            return self.run()

        base_step = order[fallback_index]
        base_text = self.checkpoints[base_step]
        if fallback_index != base_index:
            message = (
                f"Запрошено возобновление с шага {from_step.value}, но найден ближайший чекпоинт {base_step.value}."
            )
            self.logs.append(
                PipelineLogEntry(
                    step=from_step,
                    started_at=time.time(),
                    finished_at=time.time(),
                    status="error",
                    notes={
                        "message": message,
                        "requested": from_step.value,
                        "resumed_from": base_step.value,
                    },
                )
            )

        self._sync_locked_terms(base_text)

        text = base_text
        for step in order[fallback_index + 1 :]:
            if step == PipelineStep.KEYWORDS:
                text = self._run_keywords(text).text
            elif step == PipelineStep.FAQ:
                text = self._run_faq(text).text
            elif step == PipelineStep.TRIM:
                text = self._run_trim(text).text

        combined_text = text
        if self.jsonld:
            combined_text = f"{combined_text.rstrip()}\n\n{self.jsonld}\n"
        validation = validate_article(
            combined_text,
            keywords=self.keywords,
            min_chars=self.min_chars,
            max_chars=self.max_chars,
        )
        return PipelineState(
            text=combined_text,
            jsonld=self.jsonld,
            validation=validation,
            logs=self.logs,
            checkpoints=self.checkpoints,
        )

    # Metrics helpers --------------------------------------------------
    def _sync_locked_terms(self, text: str) -> None:
        pattern = re.compile(r"<!--LOCK_START term=\"([^\"]+)\"-->")
        self.locked_terms = pattern.findall(text)

    def _count_faq_entries(self, text: str) -> int:
        if "<!--FAQ_START-->" not in text or "<!--FAQ_END-->" not in text:
            return 0
        block = text.split("<!--FAQ_START-->", 1)[1].split("<!--FAQ_END-->", 1)[0]
        return len(re.findall(r"\*\*Вопрос\s+\d+\.\*\*", block))

    def _metrics(self, text: str) -> Dict[str, object]:
        article = strip_jsonld(text)
        chars_no_spaces = len(re.sub(r"\s+", "", article))
        keywords_found = 0
        for term in self.normalized_keywords:
            if build_term_pattern(term).search(article):
                keywords_found += 1
        return {
            "chars_no_spaces": chars_no_spaces,
            "keywords_found": keywords_found,
            "keywords_total": len(self.normalized_keywords),
            "faq_count": self._count_faq_entries(article),
        }
