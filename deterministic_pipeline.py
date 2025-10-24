"""LLM-driven content pipeline with explicit step-level guarantees."""

from __future__ import annotations

import json
import logging
import math
import re
import textwrap
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    G5_MAX_OUTPUT_TOKENS_MAX,
    G5_MAX_OUTPUT_TOKENS_STEP1,
    SKELETON_BATCH_SIZE_MAIN,
    SKELETON_FAQ_BATCH,
    TAIL_FILL_MAX_TOKENS,
)
from llm_client import GenerationResult, generate as llm_generate
from faq_builder import _normalize_entry
from keyword_injector import (
    KeywordCoverage,
    KeywordInjectionResult,
    LOCK_END,
    LOCK_START_TEMPLATE,
    build_term_pattern,
    evaluate_keyword_coverage,
    inject_keywords,
)
from length_controller import ensure_article_length
from length_limits import compute_soft_length_bounds
from length_trimmer import TrimResult, TrimValidationError, trim_text
from skeleton_utils import normalize_skeleton_payload
from validators import (
    ValidationError,
    ValidationResult,
    length_no_spaces,
    strip_jsonld,
    validate_article,
)


LOGGER = logging.getLogger("content_factory.pipeline")

FAQ_START = "<!--FAQ_START-->"
FAQ_END = "<!--FAQ_END-->"

_JSONLD_PATTERN = re.compile(r"<script\s+type=\"application/ld\+json\">.*?</script>", re.DOTALL)

ARTICLE_HARD_CHAR_CAP = 7200

_TEMPLATE_SNIPPETS = [
    "рассматриваем на реальных примерах, чтобы показать связь между цифрами",
    "Отмечаем юридические нюансы, возможные риски и добавляем чек-лист",
    "В выводах собираем план действий, назначаем контрольные даты",
]


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
class SectionBudget:
    title: str
    target_chars: int


@dataclass
class PipelineState:
    text: str
    jsonld: Optional[str]
    validation: Optional[ValidationResult]
    logs: List[PipelineLogEntry]
    checkpoints: Dict[PipelineStep, str]
    model_used: Optional[str] = None
    fallback_used: Optional[str] = None
    fallback_reason: Optional[str] = None
    api_route: Optional[str] = None
    token_usage: Optional[float] = None
    skeleton_payload: Optional[Dict[str, object]] = None


class PipelineStepError(RuntimeError):
    """Raised when a particular pipeline step fails irrecoverably."""

    def __init__(self, step: PipelineStep, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.step = step
        self.status_code = status_code


class SkeletonBatchKind(str, Enum):
    INTRO = "intro"
    MAIN = "main"
    FAQ = "faq"
    CONCLUSION = "conclusion"


@dataclass
class SkeletonBatchPlan:
    kind: SkeletonBatchKind
    indices: List[int] = field(default_factory=list)
    label: str = ""
    tail_fill: bool = False


@dataclass
class SkeletonOutline:
    intro_heading: str
    main_headings: List[str]
    conclusion_heading: str
    has_faq: bool

    def all_headings(self) -> List[str]:
        headings = [self.intro_heading]
        headings.extend(self.main_headings)
        headings.append(self.conclusion_heading)
        return headings

    def update_main_headings(self, new_headings: Sequence[str]) -> None:
        cleaned = [str(item).strip() for item in new_headings if str(item or "").strip()]
        if not cleaned:
            return
        current_len = len(self.main_headings)
        if current_len == 0:
            self.main_headings = cleaned
            return
        adjusted = list(self.main_headings)
        for idx, heading in enumerate(cleaned):
            if idx < len(adjusted):
                adjusted[idx] = heading
            else:
                adjusted.append(heading)
        if len(adjusted) < current_len:
            adjusted.extend(self.main_headings[len(adjusted) :])
        self.main_headings = adjusted


@dataclass
class SkeletonVolumeEstimate:
    predicted_tokens: int
    start_max_tokens: int
    cap_tokens: Optional[int]
    intro_tokens: int
    conclusion_tokens: int
    per_main_tokens: int
    per_faq_tokens: int
    requires_chunking: bool


@dataclass
class SkeletonAssembly:
    outline: SkeletonOutline
    intro: Optional[str] = None
    conclusion: Optional[str] = None
    main_sections: List[Optional[str]] = field(default_factory=list)
    faq_entries: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.main_sections:
            self.main_sections = [None] * len(self.outline.main_headings)

    def apply_intro(
        self,
        intro_text: Optional[str],
        main_headers: Optional[Sequence[str]] = None,
        conclusion_heading: Optional[str] = None,
    ) -> None:
        if intro_text and intro_text.strip():
            self.intro = intro_text.strip()
        if main_headers:
            self.outline.update_main_headings(main_headers)
            if len(self.main_sections) != len(self.outline.main_headings):
                self.main_sections = [None] * len(self.outline.main_headings)
        if conclusion_heading and conclusion_heading.strip():
            self.outline.conclusion_heading = conclusion_heading.strip()

    def apply_main(self, index: int, body: str, *, heading: Optional[str] = None) -> None:
        if heading and heading.strip() and 0 <= index < len(self.outline.main_headings):
            self.outline.main_headings[index] = heading.strip()
        if 0 <= index < len(self.main_sections) and body and body.strip():
            self.main_sections[index] = body.strip()

    def apply_faq(self, question: str, answer: str) -> None:
        if question.strip() and answer.strip():
            self.faq_entries.append({"q": question.strip(), "a": answer.strip()})

    def apply_conclusion(self, conclusion_text: Optional[str]) -> None:
        if conclusion_text and conclusion_text.strip():
            self.conclusion = conclusion_text.strip()

    def missing_main_indices(self) -> List[int]:
        return [idx for idx, body in enumerate(self.main_sections) if not body]

    def missing_faq_count(self, target_total: int) -> int:
        return max(0, target_total - len(self.faq_entries))

    def outline_snapshot(self) -> List[str]:
        headings = [self.outline.intro_heading]
        headings.extend(self.outline.main_headings)
        headings.append(self.outline.conclusion_heading)
        return headings

    def build_payload(self) -> Dict[str, object]:
        main_blocks = [body or "" for body in self.main_sections]
        payload: Dict[str, object] = {
            "intro": self.intro or "",
            "main": main_blocks,
            "faq": list(self.faq_entries),
            "conclusion": self.conclusion or "",
        }
        return payload

class DeterministicPipeline:
    """Pipeline that orchestrates LLM calls and post-processing steps."""

    def __init__(
        self,
        *,
        topic: str,
        base_outline: Sequence[str],
        required_keywords: Iterable[str],
        preferred_keywords: Optional[Iterable[str]] = None,
        min_chars: int,
        max_chars: int,
        messages: Sequence[Dict[str, object]],
        model: str,
        max_tokens: int,
        timeout_s: int,
        backoff_schedule: Optional[List[float]] = None,
        provided_faq: Optional[List[Dict[str, str]]] = None,
        jsonld_requested: bool = True,
        faq_questions: Optional[int] = None,
    ) -> None:
        if not model or not str(model).strip():
            raise PipelineStepError(PipelineStep.SKELETON, "Не указана модель для генерации.")

        self.topic = topic.strip() or "Тема"
        self.base_outline = list(base_outline) if base_outline else ["Введение", "Основная часть", "Вывод"]
        required_clean = [str(term).strip() for term in required_keywords if str(term).strip()]
        preferred_clean = [
            str(term).strip()
            for term in (preferred_keywords or [])
            if str(term).strip()
        ]
        seen_terms: set[str] = set()
        self.required_keywords: List[str] = []
        for term in required_clean:
            if term not in seen_terms:
                self.required_keywords.append(term)
                seen_terms.add(term)
        self.preferred_keywords: List[str] = []
        for term in preferred_clean:
            if term not in seen_terms:
                self.preferred_keywords.append(term)
                seen_terms.add(term)
        self.keywords = list(self.required_keywords + self.preferred_keywords)
        self.normalized_keywords = list(self.keywords)
        self.normalized_required_keywords = list(self.required_keywords)
        self.normalized_preferred_keywords = list(self.preferred_keywords)
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.messages = [dict(message) for message in messages]
        self.model = str(model).strip()
        self.max_tokens = int(max_tokens) if max_tokens else 0
        self.timeout_s = int(timeout_s)
        self.backoff_schedule = list(backoff_schedule) if backoff_schedule else None
        self.provided_faq = provided_faq or []
        self.jsonld_requested = bool(jsonld_requested)
        try:
            faq_target = int(faq_questions) if faq_questions is not None else 5
        except (TypeError, ValueError):
            faq_target = 5
        if faq_target < 0:
            faq_target = 0
        self.faq_target = faq_target

        self.logs: List[PipelineLogEntry] = []
        self.checkpoints: Dict[PipelineStep, str] = {}
        self.jsonld: Optional[str] = None
        self.locked_terms: List[str] = []
        self.jsonld_reserve: int = 0
        self.skeleton_payload: Optional[Dict[str, object]] = None
        self._skeleton_faq_entries: List[Dict[str, str]] = []
        self.keywords_coverage_percent: float = 0.0
        self.keywords_required_coverage_percent: float = 0.0

        self._model_used: Optional[str] = None
        self._fallback_used: Optional[str] = None
        self._fallback_reason: Optional[str] = None
        self._api_route: Optional[str] = None
        self._token_usage: Optional[float] = None

        self.section_budgets: List[SectionBudget] = self._compute_section_budgets()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

    def _register_llm_result(self, result: GenerationResult, usage: Optional[float]) -> None:
        if result.model_used:
            self._model_used = result.model_used
        elif self._model_used is None:
            self._model_used = self.model
        if result.fallback_used:
            self._fallback_used = result.fallback_used
        if result.fallback_reason:
            self._fallback_reason = result.fallback_reason
        if result.api_route:
            self._api_route = result.api_route
        if usage is not None:
            self._token_usage = usage

    def _compute_section_budgets(self) -> List[SectionBudget]:
        titles = [str(item or "").strip() for item in self.base_outline if str(item or "").strip()]
        if len(titles) < 3:
            titles = ["Введение", "Основная часть", "Вывод"]
        intro = titles[0]
        conclusion = titles[-1]
        main_titles = titles[1:-1] or ["Основная часть"]

        target_total = min(self.max_chars, max(self.min_chars, 6000))
        intro_weight = 0.18
        conclusion_weight = 0.16
        remaining_weight = max(0.0, 1.0 - intro_weight - conclusion_weight)
        if not main_titles:
            intro_weight = 0.25
            conclusion_weight = 0.25
            remaining_weight = 0.5
        main_weight = remaining_weight / len(main_titles) if main_titles else 0.0

        weights: List[float] = [intro_weight]
        weights.extend([main_weight] * len(main_titles))
        weights.append(conclusion_weight)

        sections = [intro, *main_titles, conclusion]
        budgets: List[SectionBudget] = []
        allocated = 0
        for title, weight in zip(sections, weights):
            portion = max(0, int(round(target_total * weight)))
            budgets.append(SectionBudget(title=title, target_chars=portion))
            allocated += portion

        diff = target_total - allocated
        idx = 0
        while diff != 0 and budgets:
            adjust = 1 if diff > 0 else -1
            budgets[idx % len(budgets)].target_chars = max(0, budgets[idx % len(budgets)].target_chars + adjust)
            diff -= adjust
            idx += 1

        return budgets

    def _validate(self, text: str) -> ValidationResult:
        return validate_article(
            text,
            required_keywords=self.required_keywords,
            preferred_keywords=self.preferred_keywords,
        )

    def _prompt_length(self, messages: Sequence[Dict[str, object]]) -> int:
        length = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                length += len(content)
        return length

    def _approx_prompt_tokens(self, messages: Sequence[Dict[str, object]]) -> int:
        """Rough token estimate based on message character count."""

        total_chars = self._prompt_length(messages)
        if total_chars <= 0:
            return 0
        # Empirical heuristic: ~4 characters per token for mixed Russian text.
        return max(1, int(math.ceil(total_chars / 4.0)))

    def _should_force_single_main_batches(
        self,
        outline: SkeletonOutline,
        estimate: "SkeletonVolumeEstimate",
    ) -> bool:
        """Return True when main batches must be generated one-by-one."""

        approx_tokens = self._approx_prompt_tokens(self.messages)
        if approx_tokens >= 3400:
            return True
        if estimate.requires_chunking:
            return True
        if len(outline.main_headings) >= 5 and estimate.cap_tokens:
            return True
        return False

    def _extract_usage(self, result: GenerationResult) -> Optional[float]:
        metadata = result.metadata or {}
        if not isinstance(metadata, dict):
            return None
        candidates = [
            metadata.get("usage_output_tokens"),
            metadata.get("token_usage"),
            metadata.get("output_tokens"),
        ]
        usage_block = metadata.get("usage")
        if isinstance(usage_block, dict):
            candidates.append(usage_block.get("output_tokens"))
            candidates.append(usage_block.get("total_tokens"))
        for candidate in candidates:
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None

    def _call_llm(
        self,
        *,
        step: PipelineStep,
        messages: Sequence[Dict[str, object]],
        max_tokens: Optional[int] = None,
        override_model: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        responses_format: Optional[Dict[str, object]] = None,
        allow_incomplete: bool = False,
    ) -> GenerationResult:
        prompt_len = self._prompt_length(messages)
        limit = max_tokens if max_tokens and max_tokens > 0 else self.max_tokens
        if not limit or limit <= 0:
            limit = 700
        attempt = 0
        model_to_use = (override_model or self.model).strip()
        while attempt < 3:
            attempt += 1
            LOGGER.info(
                "LOG:LLM_REQUEST step=%s model=%s prompt_len=%d attempt=%d max_tokens=%d",
                step.value,
                model_to_use,
                prompt_len,
                attempt,
                limit,
            )
            try:
                result = llm_generate(
                    list(messages),
                    model=model_to_use,
                    max_tokens=limit,
                    timeout_s=self.timeout_s,
                    backoff_schedule=self.backoff_schedule,
                    previous_response_id=previous_response_id,
                    responses_text_format=responses_format,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("LOG:LLM_ERROR step=%s message=%s", step.value, exc)
                raise PipelineStepError(step, f"Сбой при обращении к модели ({step.value}): {exc}") from exc

            usage = self._extract_usage(result)
            metadata = result.metadata or {}
            status = str(metadata.get("status") or "ok")
            incomplete_reason = metadata.get("incomplete_reason") or ""
            LOGGER.info(
                "LOG:LLM_RESPONSE step=%s tokens_used=%s status=%s",
                step.value,
                "%.0f" % usage if isinstance(usage, (int, float)) else "unknown",
                status,
            )
            if status.lower() != "incomplete" and not incomplete_reason:
                self._register_llm_result(result, usage)
                return result

            if allow_incomplete:
                LOGGER.warning(
                    "LLM_INCOMPLETE_RETURN step=%s status=%s reason=%s",
                    step.value,
                    status or "incomplete",
                    incomplete_reason or "",
                )
                self._register_llm_result(result, usage)
                return result

            if attempt >= 3:
                message = "Модель не завершила генерацию (incomplete)."
                LOGGER.error(
                    "LLM_INCOMPLETE_ABORT step=%s status=%s reason=%s",
                    step.value,
                    status or "incomplete",
                    incomplete_reason or "",
                )
                raise PipelineStepError(step, message)

            LOGGER.warning(
                "LLM_RETRY_incomplete step=%s attempt=%d status=%s reason=%s",
                step.value,
                attempt,
                status or "incomplete",
                incomplete_reason or "",
            )
            limit = max(200, int(limit * 0.9))

        raise PipelineStepError(step, "Не удалось получить ответ от модели.")

    def _check_template_text(self, text: str, step: PipelineStep) -> None:
        lowered = text.lower()
        if lowered.count("дополнительно рассматривается") >= 3:
            raise PipelineStepError(step, "Обнаружен шаблонный текст 'Дополнительно рассматривается'.")
        for snippet in _TEMPLATE_SNIPPETS:
            if snippet in lowered:
                raise PipelineStepError(step, "Найден служебный шаблонный фрагмент, генерация отклонена.")

    def _metrics(self, text: str) -> Dict[str, object]:
        article = strip_jsonld(text)
        chars_no_spaces = length_no_spaces(article)
        keywords_found = 0
        for term in self.normalized_keywords:
            if build_term_pattern(term).search(article):
                keywords_found += 1
        return {
            "chars_no_spaces": chars_no_spaces,
            "keywords_found": keywords_found,
            "keywords_total": len(self.normalized_keywords),
        }

    def _prepare_outline(self) -> SkeletonOutline:
        outline = [str(segment or "").strip() for segment in self.base_outline if str(segment or "").strip()]
        if not outline:
            outline = ["Введение", "Основная часть", "FAQ", "Вывод"]
        intro_heading = outline[0]
        conclusion_heading = outline[-1] if len(outline) > 1 else "Вывод"
        faq_markers = {"faq", "f.a.q.", "вопросы и ответы"}
        has_faq = any(entry.strip().lower() in faq_markers for entry in outline)
        main_candidates: List[str] = []
        for entry in outline[1:-1]:
            normalized = entry.strip()
            if normalized.lower() in faq_markers:
                continue
            if normalized:
                main_candidates.append(normalized)
        if not main_candidates:
            main_candidates = ["Основная часть"]
        return SkeletonOutline(
            intro_heading=intro_heading,
            main_headings=main_candidates,
            conclusion_heading=conclusion_heading,
            has_faq=has_faq,
        )

    def _predict_skeleton_volume(self, outline: SkeletonOutline) -> SkeletonVolumeEstimate:
        cap = G5_MAX_OUTPUT_TOKENS_MAX if G5_MAX_OUTPUT_TOKENS_MAX > 0 else None
        requested_min = int(self.min_chars) if self.min_chars > 0 else DEFAULT_MIN_LENGTH
        requested_max = int(self.max_chars) if self.max_chars > 0 else DEFAULT_MAX_LENGTH
        min_chars = max(2000, requested_min)
        hard_cap = ARTICLE_HARD_CHAR_CAP if ARTICLE_HARD_CHAR_CAP > 0 else None
        max_candidate = max(min_chars + 400, requested_max)
        if hard_cap is not None:
            max_candidate = min(max_candidate, hard_cap)
        max_chars = max_candidate
        avg_chars = max(min_chars, int(round((min_chars + max_chars) / 2)))
        approx_tokens = max(1400, int(avg_chars / 3.1))
        main_count = max(1, len(outline.main_headings))
        faq_count = self.faq_target if outline.has_faq else 0
        intro_tokens = max(160, int(approx_tokens * 0.12))
        conclusion_tokens = max(140, int(approx_tokens * 0.1))
        faq_pool = max(0, int(approx_tokens * 0.2)) if faq_count else 0
        per_faq_tokens = max(70, int(faq_pool / faq_count)) if faq_count else 0
        allocated_faq = per_faq_tokens * faq_count
        remaining_for_main = max(
            approx_tokens - intro_tokens - conclusion_tokens - allocated_faq,
            250 * main_count,
        )
        per_main_tokens = max(250, int(remaining_for_main / main_count)) if main_count else 0
        predicted = intro_tokens + conclusion_tokens + per_main_tokens * main_count + per_faq_tokens * faq_count
        start_max = int(predicted * 1.25)
        step1_cap = G5_MAX_OUTPUT_TOKENS_STEP1 if G5_MAX_OUTPUT_TOKENS_STEP1 > 0 else 1200
        if cap is not None and cap > 0:
            start_max = min(start_max, cap)
        start_max = min(start_max, step1_cap)
        start_max = max(900, start_max)
        requires_chunking = bool(cap is not None and predicted > cap)
        LOGGER.info(
            "SKELETON_ESTIMATE predicted=%d start_max=%d cap=%s → resolved max_output_tokens=%d",
            predicted,
            start_max,
            cap if cap is not None else "-",
            start_max,
        )
        return SkeletonVolumeEstimate(
            predicted_tokens=predicted,
            start_max_tokens=start_max,
            cap_tokens=cap,
            intro_tokens=intro_tokens,
            conclusion_tokens=conclusion_tokens,
            per_main_tokens=per_main_tokens,
            per_faq_tokens=per_faq_tokens,
            requires_chunking=requires_chunking,
        )

    def _build_skeleton_batches(
        self,
        outline: SkeletonOutline,
        estimate: SkeletonVolumeEstimate,
    ) -> List[SkeletonBatchPlan]:
        batches: List[SkeletonBatchPlan] = [SkeletonBatchPlan(kind=SkeletonBatchKind.INTRO, label="intro")]
        main_count = len(outline.main_headings)
        if main_count > 0:
            if self._should_force_single_main_batches(outline, estimate):
                batch_size = 1
            else:
                batch_size = max(1, min(SKELETON_BATCH_SIZE_MAIN, main_count))
            start = 0
            while start < main_count:
                end = min(start + batch_size, main_count)
                indices = list(range(start, end))
                if len(indices) == 1:
                    label = f"main[{indices[0] + 1}]"
                else:
                    label = f"main[{indices[0] + 1}-{indices[-1] + 1}]"
                batches.append(
                    SkeletonBatchPlan(
                        kind=SkeletonBatchKind.MAIN,
                        indices=indices,
                        label=label,
                    )
                )
                start = end
        if outline.has_faq and self.faq_target > 0:
            total_faq = self.faq_target
            produced = 0
            first_batch = min(SKELETON_FAQ_BATCH, total_faq)
            if first_batch > 0:
                indices = list(range(produced, produced + first_batch))
                label = (
                    f"faq[{indices[0] + 1}-{indices[-1] + 1}]"
                    if len(indices) > 1
                    else f"faq[{indices[0] + 1}]"
                )
                batches.append(
                    SkeletonBatchPlan(
                        kind=SkeletonBatchKind.FAQ,
                        indices=indices,
                        label=label,
                    )
                )
                produced += first_batch
            while produced < total_faq:
                remaining = total_faq - produced
                chunk_size = min(SKELETON_FAQ_BATCH, remaining)
                indices = list(range(produced, produced + chunk_size))
                label = (
                    f"faq[{indices[0] + 1}-{indices[-1] + 1}]"
                    if len(indices) > 1
                    else f"faq[{indices[0] + 1}]"
                )
                batches.append(
                    SkeletonBatchPlan(
                        kind=SkeletonBatchKind.FAQ,
                        indices=indices,
                        label=label,
                    )
                )
                produced += chunk_size
        batches.append(SkeletonBatchPlan(kind=SkeletonBatchKind.CONCLUSION, label="conclusion"))
        return batches

    def _format_batch_label(
        self, kind: SkeletonBatchKind, indices: Sequence[int], *, suffix: str = ""
    ) -> str:
        base: str
        if kind == SkeletonBatchKind.MAIN:
            if not indices:
                base = "main"
            elif len(indices) == 1:
                base = f"main[{indices[0] + 1}]"
            else:
                base = f"main[{indices[0] + 1}-{indices[-1] + 1}]"
        elif kind == SkeletonBatchKind.FAQ:
            if not indices:
                base = "faq"
            elif len(indices) == 1:
                base = f"faq[{indices[0] + 1}]"
            else:
                base = f"faq[{indices[0] + 1}-{indices[-1] + 1}]"
        else:
            base = kind.value
        return base + suffix

    def _can_split_batch(self, kind: SkeletonBatchKind, indices: Sequence[int]) -> bool:
        return kind in (SkeletonBatchKind.MAIN, SkeletonBatchKind.FAQ) and len(indices) > 1

    def _split_batch_indices(self, indices: Sequence[int]) -> Tuple[List[int], List[int]]:
        materialized = [int(idx) for idx in indices]
        if len(materialized) <= 1:
            return list(materialized), []
        keep_size = max(1, len(materialized) - 1)
        keep = materialized[:keep_size]
        remainder = materialized[keep_size:]
        if not keep:
            keep = [materialized[0]]
            remainder = materialized[1:]
        return keep, remainder

    def _batch_has_payload(self, kind: SkeletonBatchKind, payload: object) -> bool:
        if kind == SkeletonBatchKind.INTRO:
            if not isinstance(payload, dict):
                return False
            intro = str(payload.get("intro") or "").strip()
            headers = payload.get("main_headers")
            has_headers = bool(
                isinstance(headers, list)
                and any(str(item or "").strip() for item in headers)
            )
            conclusion_heading = str(payload.get("conclusion_heading") or "").strip()
            return bool(intro or has_headers or conclusion_heading)
        if kind == SkeletonBatchKind.MAIN:
            if not isinstance(payload, dict):
                return False
            sections = payload.get("sections")
            if not isinstance(sections, list):
                alt_main = payload.get("main")
                if isinstance(alt_main, list):
                    for entry in alt_main:
                        if isinstance(entry, dict):
                            body = str(entry.get("body") or entry.get("text") or "").strip()
                            title = str(entry.get("title") or entry.get("heading") or "").strip()
                            if body or title:
                                return True
                        else:
                            if str(entry or "").strip():
                                return True
                return False
            for section in sections:
                if not isinstance(section, dict):
                    continue
                body = str(section.get("body") or "").strip()
                title = str(section.get("title") or "").strip()
                if body or title:
                    return True
            return False
        if kind == SkeletonBatchKind.FAQ:
            if not isinstance(payload, dict):
                return False
            faq_items = payload.get("faq")
            if not isinstance(faq_items, list):
                return False
            for entry in faq_items:
                if not isinstance(entry, dict):
                    continue
                question = str(entry.get("q") or "").strip()
                answer = str(entry.get("a") or "").strip()
                if question or answer:
                    return True
            return False
        if kind == SkeletonBatchKind.CONCLUSION:
            if not isinstance(payload, dict):
                return False
            conclusion = str(payload.get("conclusion") or "").strip()
            return bool(conclusion)
        return False

    def _metadata_response_id(self, metadata: Mapping[str, object]) -> str:
        if not isinstance(metadata, dict):
            return ""
        candidates = (
            "response_id",
            "id",
            "responseId",
            "responseID",
            "request_id",
            "requestId",
        )
        for key in candidates:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        fallback = metadata.get("previous_response_id")
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return ""

    def _apply_inline_faq(self, payload: object, assembly: SkeletonAssembly) -> None:
        if not isinstance(payload, dict):
            return
        faq_items = payload.get("faq")
        if not isinstance(faq_items, list):
            return
        existing_questions = {entry.get("q") for entry in assembly.faq_entries}
        for entry in faq_items:
            if assembly.missing_faq_count(self.faq_target) == 0:
                break
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("q") or entry.get("question") or "").strip()
            answer = str(entry.get("a") or entry.get("answer") or "").strip()
            if not question or not answer:
                continue
            if question in existing_questions:
                continue
            assembly.apply_faq(question, answer)
            existing_questions.add(question)

    def _batch_schema(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        item_count: int,
    ) -> Dict[str, object]:
        if batch.kind == SkeletonBatchKind.INTRO:
            main_count = max(0, len(outline.main_headings))
            schema = {
                "type": "object",
                "properties": {
                    "intro": {"type": "string"},
                    "main_headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": main_count,
                    },
                    "conclusion_heading": {"type": "string"},
                },
                "required": ["intro", "main_headers", "conclusion_heading"],
                "additionalProperties": False,
            }
        elif batch.kind == SkeletonBatchKind.MAIN:
            min_items = max(0, int(item_count))
            schema = {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "body": {"type": "string"},
                            },
                            "required": ["title", "body"],
                            "additionalProperties": False,
                        },
                        "minItems": min_items,
                    }
                },
                "required": ["sections"],
                "additionalProperties": False,
            }
        elif batch.kind == SkeletonBatchKind.FAQ:
            min_items = max(0, int(item_count))
            schema = {
                "type": "object",
                "properties": {
                    "faq": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "q": {"type": "string"},
                                "a": {"type": "string"},
                            },
                            "required": ["q", "a"],
                            "additionalProperties": False,
                        },
                        "minItems": min_items,
                    }
                },
                "required": ["faq"],
                "additionalProperties": False,
            }
        else:
            schema = {
                "type": "object",
                "properties": {"conclusion": {"type": "string"}},
                "required": ["conclusion"],
                "additionalProperties": False,
            }
        name_map = {
            SkeletonBatchKind.INTRO: "seo_article_intro_batch",
            SkeletonBatchKind.MAIN: "seo_article_main_batch",
            SkeletonBatchKind.FAQ: "seo_article_faq_batch",
            SkeletonBatchKind.CONCLUSION: "seo_article_conclusion_batch",
        }
        format_block = {
            "type": "json_schema",
            "name": name_map.get(batch.kind, "seo_article_skeleton_batch"),
            "schema": schema,
            "strict": True,
        }
        return self._prepare_format_block(format_block, batch=batch)

    def _prepare_format_block(
        self,
        format_block: Dict[str, object],
        *,
        batch: SkeletonBatchPlan,
    ) -> Dict[str, object]:
        fmt_type = str(format_block.get("type") or "").lower()
        if fmt_type == "json_schema":
            schema = format_block.get("schema")
            if not isinstance(schema, dict):
                raise PipelineStepError(
                    PipelineStep.SKELETON,
                    "Некорректная схема ответа для батча скелета.",
                )
            self._enforce_schema_defaults(schema)
        format_name = str(format_block.get("name") or "")
        if (
            batch.kind in (
                SkeletonBatchKind.MAIN,
                SkeletonBatchKind.FAQ,
                SkeletonBatchKind.CONCLUSION,
            )
            and format_name.strip() == "seo_article_skeleton"
        ):
            replacement_map = {
                SkeletonBatchKind.MAIN: "seo_article_main_batch",
                SkeletonBatchKind.FAQ: "seo_article_faq_batch",
                SkeletonBatchKind.CONCLUSION: "seo_article_conclusion_batch",
            }
            replacement = replacement_map.get(batch.kind, "seo_article_skeleton_batch")
            LOGGER.warning(
                "LOG:BAT_FMT_FIXUP kind=%s label=%s from=%s to=%s",
                batch.kind.value,
                batch.label or self._format_batch_label(batch.kind, batch.indices),
                format_name,
                replacement,
            )
            format_block = dict(format_block)
            format_block["name"] = replacement
        return format_block

    def _enforce_schema_defaults(self, schema: Dict[str, object], path: str = "$") -> None:
        if not isinstance(schema, dict):
            raise PipelineStepError(
                PipelineStep.SKELETON,
                f"Невалидная структура схемы по пути {path}.",
            )
        node_type = str(schema.get("type") or "")
        if node_type == "object":
            properties = schema.get("properties")
            if isinstance(properties, dict):
                if "additionalProperties" not in schema:
                    schema["additionalProperties"] = False
                required = schema.get("required")
                if isinstance(required, list):
                    missing = [
                        str(field)
                        for field in required
                        if field not in properties
                    ]
                    if missing:
                        missing_fields = ", ".join(sorted(missing))
                        raise PipelineStepError(
                            PipelineStep.SKELETON,
                            f"Схема {path} содержит обязательные поля без описания: {missing_fields}",
                        )
                for key, value in properties.items():
                    if isinstance(value, dict):
                        self._enforce_schema_defaults(value, f"{path}.{key}")
            else:
                schema.setdefault("properties", {})
                schema.setdefault("additionalProperties", False)
        if node_type == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                self._enforce_schema_defaults(items, f"{path}[]")
        for keyword in ("allOf", "anyOf", "oneOf"):
            collection = schema.get(keyword)
            if isinstance(collection, list):
                for index, value in enumerate(collection):
                    if isinstance(value, dict):
                        self._enforce_schema_defaults(
                            value, f"{path}.{keyword}[{index}]"
                        )

    def _extract_response_json(self, raw_text: str) -> Optional[object]:
        candidate = (raw_text or "").strip()
        if not candidate:
            return None
        if "<response_json>" in candidate and "</response_json>" in candidate:
            try:
                candidate = candidate.split("<response_json>", 1)[1].split("</response_json>", 1)[0]
            except Exception:  # pragma: no cover - defensive
                candidate = candidate
        candidate = candidate.strip()
        decoder = json.JSONDecoder()
        for start_index in (0, candidate.find("{"), candidate.find("[")):
            if start_index is None or start_index < 0:
                continue
            snippet = candidate[start_index:].strip()
            if not snippet:
                continue
            try:
                parsed, _ = decoder.raw_decode(snippet)
                return parsed
            except json.JSONDecodeError:
                continue
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            balanced = match.group(0)
            try:
                return json.loads(balanced)
            except json.JSONDecodeError:
                pass
        return None

    def _normalize_intro_batch(
        self, payload: object, outline: SkeletonOutline
    ) -> Tuple[Dict[str, object], List[str]]:
        normalized: Dict[str, object] = {}
        missing: List[str] = []
        if not isinstance(payload, dict):
            return normalized, ["intro", "main_headers"]
        intro_text = str(payload.get("intro") or "").strip()
        headers_raw = payload.get("main_headers")
        if isinstance(headers_raw, list):
            headers = [str(item or "").strip() for item in headers_raw if str(item or "").strip()]
        else:
            headers = []
        conclusion_heading = str(payload.get("conclusion_heading") or "").strip()
        if not intro_text:
            missing.append("intro")
        alt_main = payload.get("main")
        needs_headers = False
        if len(headers) < len(outline.main_headings) or (
            isinstance(alt_main, list) and len(alt_main) > len(headers)
        ):
            derived: List[str] = []
            if isinstance(alt_main, list):
                for position, item in enumerate(alt_main):
                    if isinstance(item, dict):
                        candidate = str(item.get("title") or item.get("heading") or "").strip()
                        if candidate:
                            derived.append(candidate)
                    elif isinstance(item, str):
                        text_value = item.strip()
                        if text_value:
                            base_heading = outline.main_headings[0] if outline.main_headings else "Основная часть"
                            derived.append(f"{base_heading} #{position + 1}")
            if derived:
                for idx, value in enumerate(derived):
                    if idx < len(headers):
                        if value:
                            headers[idx] = value
                    else:
                        headers.append(value)
            needs_headers = len(headers) < len(outline.main_headings) and not (
                isinstance(alt_main, list)
                and any(str(item or "").strip() for item in alt_main)
            )
        if needs_headers:
            missing.append("main_headers")
        normalized["intro"] = intro_text
        normalized["main_headers"] = headers
        normalized["conclusion_heading"] = conclusion_heading or outline.conclusion_heading
        return normalized, missing

    def _normalize_main_batch(
        self,
        payload: object,
        target_indices: Sequence[int],
        outline: SkeletonOutline,
    ) -> Tuple[List[Tuple[int, str, str]], List[int]]:
        normalized: List[Tuple[int, str, str]] = []
        missing: List[int] = []
        if not isinstance(payload, dict):
            return normalized, list(target_indices)
        sections = payload.get("sections")
        if not isinstance(sections, list):
            alt_main = payload.get("main")
            if isinstance(alt_main, list):
                converted: List[Dict[str, str]] = []
                for item in alt_main:
                    if isinstance(item, dict):
                        title = str(item.get("title") or item.get("heading") or "").strip()
                        body = str(
                            item.get("body") or item.get("text") or item.get("content") or ""
                        ).strip()
                        if not body and not title:
                            continue
                        converted.append({"title": title, "body": body or title})
                    else:
                        text_value = str(item or "").strip()
                        if not text_value:
                            continue
                        converted.append({"title": "", "body": text_value})
                sections = converted
            else:
                return normalized, list(target_indices)
        max_count = len(target_indices)
        for position, section in enumerate(sections[:max_count]):
            if not isinstance(section, dict):
                missing.append(target_indices[position])
                continue
            target_index = target_indices[position]
            title = str(section.get("title") or outline.main_headings[target_index]).strip()
            body = str(section.get("body") or "").strip()
            if not body:
                missing.append(target_index)
                continue
            normalized.append((target_index, title or outline.main_headings[target_index], body))
        if len(normalized) < len(target_indices):
            for index in target_indices[len(normalized) :]:
                if index not in missing:
                    missing.append(index)
        return normalized, missing

    def _normalize_faq_batch(
        self,
        payload: object,
        target_indices: Sequence[int],
    ) -> Tuple[List[Tuple[int, str, str]], List[int]]:
        normalized: List[Tuple[int, str, str]] = []
        missing: List[int] = []
        if not isinstance(payload, dict):
            return normalized, list(target_indices)
        faq_items = payload.get("faq")
        if not isinstance(faq_items, list):
            return normalized, list(target_indices)
        max_count = len(target_indices)
        for position, entry in enumerate(faq_items[:max_count]):
            if not isinstance(entry, dict):
                missing.append(target_indices[position])
                continue
            question = str(entry.get("q") or "").strip()
            answer = str(entry.get("a") or "").strip()
            target_index = target_indices[position]
            if not question or not answer:
                missing.append(target_index)
                continue
            normalized.append((target_index, question, answer))
        if len(normalized) < len(target_indices):
            for index in target_indices[len(normalized) :]:
                if index not in missing:
                    missing.append(index)
        return normalized, missing

    def _normalize_conclusion_batch(self, payload: object) -> Tuple[str, bool]:
        if not isinstance(payload, dict):
            return "", True
        conclusion = str(payload.get("conclusion") or "").strip()
        return conclusion, not bool(conclusion)

    def _batch_token_budget(
        self,
        batch: SkeletonBatchPlan,
        estimate: SkeletonVolumeEstimate,
        item_count: int,
    ) -> int:
        cap = estimate.cap_tokens or estimate.start_max_tokens
        if batch.kind == SkeletonBatchKind.INTRO:
            base = estimate.intro_tokens + max(estimate.per_main_tokens, 300)
        elif batch.kind == SkeletonBatchKind.MAIN:
            base = max(estimate.per_main_tokens * max(1, item_count), 400)
        elif batch.kind == SkeletonBatchKind.FAQ:
            base = max(estimate.per_faq_tokens * max(1, item_count), 320)
        else:
            base = estimate.conclusion_tokens + max(estimate.per_faq_tokens, 120)
        allowance = cap if cap and cap > 0 else estimate.start_max_tokens
        if allowance and allowance > 0:
            base = min(base, allowance)
        return max(400, int(base * 1.1))

    def _build_batch_messages(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        target_indices: Sequence[int],
        tail_fill: bool = False,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        messages = [dict(message) for message in self.messages]
        outline_text = ", ".join(outline.all_headings())
        general_lines = [
            "Ты создаёшь детерминированный SEO-скелет статьи.",
            f"Тема: {self.topic}.",
            f"Общий объём: {self.min_chars}–{self.max_chars} символов без пробелов.",
            f"План разделов: {outline_text}.",
        ]
        if self.normalized_keywords:
            general_lines.append(
                "Сохраняй точные упоминания ключевых слов: " + ", ".join(self.normalized_keywords) + "."
            )

        lines: List[str] = list(general_lines)
        if batch.kind == SkeletonBatchKind.INTRO:
            lines.append("На вводный абзац выделен лимит до 450 слов — не превышай его.")
            lines.extend(
                [
                    "Сформируй вводный абзац без приветствий и уточни заголовки основных разделов.",
                    "Верни JSON вида {\"intro\": str, \"main_headers\": [..], \"conclusion_heading\": str}.",
                    "main_headers должен содержать столько элементов, сколько основных разделов в плане, порядок сохраняется.",
                    "intro — один абзац с 3–4 предложениями, без списков.",
                ]
            )
        elif batch.kind == SkeletonBatchKind.MAIN:
            headings = [outline.main_headings[index] for index in target_indices]
            already_ready = [
                outline.main_headings[idx]
                for idx, body in enumerate(assembly.main_sections)
                if body and idx not in target_indices
            ]
            lines.append(
                "Нужно детализировать разделы: "
                + "; ".join(
                    f"{index + 1}. {heading}" for index, heading in zip(target_indices, headings)
                )
                + "."
            )
            if already_ready:
                lines.append(
                    "Эти разделы уже готовы и переписывать их не нужно: "
                    + "; ".join(already_ready)
                    + "."
                )
            lines.append(
                "Каждый раздел держи в рамках до четырёх абзацев по 3–4 предложения — без лишней воды."
            )
            lines.extend(
                [
                    "Каждый раздел — 3–5 абзацев по 3–4 предложения, с примерами, рисками и расчётами.",
                    "Верни JSON {\"sections\": [{\"title\": str, \"body\": str}, ...]} в порядке указанного списка.",
                ]
            )
            if tail_fill:
                lines.append("Верни только недостающие разделы без повтора уже написанных частей.")
        elif batch.kind == SkeletonBatchKind.FAQ:
            start_number = target_indices[0] + 1 if target_indices else 1
            lines.append(
                "Подготовь новые элементы FAQ с практичными ответами (минимум два предложения каждый)."
            )
            lines.append("Каждый ответ — максимум 2–3 предложения, избегай повторов.")
            lines.append(
                "Верни JSON {\"faq\": [{\"q\": str, \"a\": str}, ...]} в количестве, равном запросу."
            )
            lines.append(
                "Продолжай нумерацию вопросов, начиная с пункта №%d." % start_number
            )
            if tail_fill:
                lines.append("Добавь только недостающие вопросы и ответы.")
        else:
            lines.append("Заключение ограничь объёмом до 180 слов.")
            lines.extend(
                [
                    "Сделай связный вывод и обозначь план действий, ссылаясь на ключевые идеи статьи.",
                    "Верни JSON {\"conclusion\": str} с одним абзацем.",
                ]
            )

        lines.append("Ответ заверни в теги <response_json>...</response_json> без комментариев.")
        user_payload = textwrap.dedent("\n".join(lines)).strip()
        messages.append({"role": "user", "content": user_payload})
        format_block = self._batch_schema(batch, outline=outline, item_count=len(target_indices))
        return messages, format_block

    def _parse_fallback_main(
        self,
        text: str,
        *,
        target_index: int,
        outline: SkeletonOutline,
    ) -> Optional[Dict[str, object]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title_buffer: List[str] = []
        body_buffer: List[str] = []
        current = ""
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("заголовок"):
                current = "title"
                content = line.split(":", 1)[1] if ":" in line else ""
                title_buffer = [content.strip()]
                continue
            if lowered.startswith("текст"):
                current = "body"
                content = line.split(":", 1)[1] if ":" in line else ""
                body_buffer = [content.strip()]
                continue
            if current == "title":
                title_buffer.append(line)
            elif current == "body":
                body_buffer.append(line)
        title = " ".join(part for part in title_buffer if part).strip()
        if not title and 0 <= target_index < len(outline.main_headings):
            title = outline.main_headings[target_index]
        body = "\n".join(part for part in body_buffer if part).strip()
        if not body:
            return None
        return {"sections": [{"title": title, "body": body}]}

    def _parse_fallback_faq(self, text: str) -> Optional[Dict[str, object]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        question_parts: List[str] = []
        answer_parts: List[str] = []
        current = ""
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("вопрос"):
                current = "question"
                content = line.split(":", 1)[1] if ":" in line else ""
                question_parts = [content.strip()]
                continue
            if lowered.startswith("ответ"):
                current = "answer"
                content = line.split(":", 1)[1] if ":" in line else ""
                answer_parts = [content.strip()]
                continue
            if current == "question":
                question_parts.append(line)
            elif current == "answer":
                answer_parts.append(line)
        question = " ".join(part for part in question_parts if part).strip()
        answer = "\n".join(part for part in answer_parts if part).strip()
        if not question or not answer:
            return None
        return {"faq": [{"q": question, "a": answer}]}

    def _run_fallback_batch(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        target_indices: Sequence[int],
        max_tokens: int,
        previous_response_id: Optional[str],
    ) -> Tuple[Optional[object], Optional[GenerationResult]]:
        if not target_indices:
            return None, None
        messages = [dict(message) for message in self.messages]
        outline_text = ", ".join(outline.all_headings())
        base_lines = [
            "Ты создаёшь детерминированный SEO-скелет статьи.",
            f"Тема: {self.topic}.",
            f"Общий объём: {self.min_chars}–{self.max_chars} символов без пробелов.",
            f"План разделов: {outline_text}.",
        ]
        if self.normalized_keywords:
            base_lines.append(
                "Сохраняй точные упоминания ключевых слов: " + ", ".join(self.normalized_keywords) + "."
            )
        lines: List[str] = list(base_lines)
        lines.append(
            (
                "Выведи ровно одну секцию для статьи, кратко и без JSON. "
                "Только текст. Потом парсится и сериализуется в нужный JSON-фрагмент."
            )
        )
        if batch.kind == SkeletonBatchKind.MAIN:
            target_index = target_indices[0]
            heading = outline.main_headings[target_index] if target_index < len(outline.main_headings) else "Раздел"
            ready = [
                outline.main_headings[idx]
                for idx, body in enumerate(assembly.main_sections)
                if body and idx != target_index
            ]
            lines.append(
                f"Нужно срочно раскрыть раздел №{target_index + 1}: {heading}."
            )
            if ready:
                lines.append(
                    "Эти разделы уже готовы, не переписывай их: " + "; ".join(ready) + "."
                )
            lines.extend(
                [
                    "Сформируй ровно один новый раздел основной части.",
                    "Верни строго две строки меток:",
                    "Заголовок: <краткое название раздела>",
                    "Текст: <развёрнутый текст раздела на 3–5 абзацев>",
                    "Не добавляй списков и дополнительных пояснений.",
                ]
            )
        elif batch.kind == SkeletonBatchKind.FAQ:
            start_number = target_indices[0] + 1
            lines.extend(
                [
                    f"Нужно дополнить FAQ одним пунктом с номером {start_number}.",
                    "Верни строго две строки меток:",
                    "Вопрос: <формулировка вопроса>",
                    "Ответ: <полный ответ из 2–3 предложений>",
                    "Не добавляй иных строк и маркеров.",
                ]
            )
        else:
            return None, None
        lines.append("Ответ дай без JSON и без тегов <response_json>.")
        user_payload = textwrap.dedent("\n".join(lines)).strip()
        messages.append({"role": "user", "content": user_payload})
        format_block = {"type": "output_text", "name": "output_text"}
        result = self._call_llm(
            step=PipelineStep.SKELETON,
            messages=messages,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            responses_format=format_block,
            allow_incomplete=True,
        )
        text = result.text.strip()
        if not text:
            return None, result
        if batch.kind == SkeletonBatchKind.MAIN:
            payload = self._parse_fallback_main(
                text,
                target_index=target_indices[0],
                outline=outline,
            )
        else:
            payload = self._parse_fallback_faq(text)
        if payload is None:
            return None, result
        LOGGER.info(
            "FALLBACK_ROUTE used=output_text kind=%s label=%s",
            batch.kind.value,
            batch.label,
        )
        return payload, result

    def _run_cap_fallback_batch(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        target_indices: Sequence[int],
        max_tokens: int,
        previous_response_id: Optional[str],
    ) -> Tuple[Optional[object], Optional[GenerationResult]]:
        if batch.kind not in (SkeletonBatchKind.MAIN, SkeletonBatchKind.FAQ):
            return None, None
        if not target_indices:
            return None, None
        messages = [dict(message) for message in self.messages]
        outline_text = ", ".join(outline.all_headings())
        lines: List[str] = [
            "Аварийный режим: сформируй минимальный JSON-фрагмент.",
            f"Тема: {self.topic}.",
            f"План разделов: {outline_text}.",
        ]
        if batch.kind == SkeletonBatchKind.MAIN:
            target_index = target_indices[0]
            heading = (
                outline.main_headings[target_index]
                if target_index < len(outline.main_headings)
                else f"Раздел {target_index + 1}"
            )
            ready_sections = [
                outline.main_headings[idx]
                for idx, body in enumerate(assembly.main_sections)
                if body and idx != target_index
            ]
            if ready_sections:
                lines.append("Уже готовы: " + "; ".join(ready_sections) + ".")
            lines.extend(
                [
                    f"Нужен краткий текст для раздела №{target_index + 1}: {heading}.",
                    "Ответ верни в JSON: {\"sections\": [{\"title\": \"...\", \"body\": \"...\"}]}",
                    "Минимум два предложения в body.",
                ]
            )
        else:
            start_number = target_indices[0] + 1
            lines.extend(
                [
                    f"Добавь пункт FAQ №{start_number}.",
                    "Ответ в формате JSON: {\"faq\": [{\"q\": \"...\", \"a\": \"...\"}]}",
                    "Вопрос и ответ должны быть осмысленными.",
                ]
            )
        user_payload = textwrap.dedent("\n".join(lines)).strip()
        messages.append({"role": "user", "content": user_payload})
        schema = self._batch_schema(
            batch,
            outline=outline,
            item_count=len(target_indices) or 1,
        )
        format_block = {"type": "json_object", "schema": schema}
        emergency_tokens = max(320, min(max_tokens, 900))
        LOGGER.info(
            "CAP_FALLBACK kind=%s label=%s tokens=%d",
            batch.kind.value,
            batch.label,
            emergency_tokens,
        )
        result = self._call_llm(
            step=PipelineStep.SKELETON,
            messages=messages,
            max_tokens=emergency_tokens,
            previous_response_id=previous_response_id,
            responses_format=format_block,
            allow_incomplete=True,
        )
        payload_obj = self._extract_response_json(result.text)
        if not self._batch_has_payload(batch.kind, payload_obj):
            return None, result
        return payload_obj, result

    def _build_batch_placeholder(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        target_indices: Sequence[int],
    ) -> Optional[object]:
        if batch.kind == SkeletonBatchKind.INTRO:
            headers = list(outline.main_headings)
            if not headers:
                headers = [f"Раздел {idx + 1}" for idx in range(max(1, len(target_indices)))]
            return {
                "intro": (
                    f"Введение по теме «{self.topic}» будет расширено после снятия ограничений."  # noqa: B950
                ),
                "main_headers": headers,
                "conclusion_heading": outline.conclusion_heading
                or "Заключение",
            }
        if batch.kind == SkeletonBatchKind.MAIN:
            sections: List[Dict[str, str]] = []
            indices = list(target_indices) or [0]
            for index in indices:
                heading = (
                    outline.main_headings[index]
                    if 0 <= index < len(outline.main_headings)
                    else f"Раздел {index + 1}"
                )
                body = (
                    f"Раздел «{heading}» будет детализирован в финальной версии статьи."
                )
                sections.append({"title": heading, "body": body})
            return {"sections": sections}
        if batch.kind == SkeletonBatchKind.FAQ:
            faq_entries: List[Dict[str, str]] = []
            indices = list(target_indices) or [0]
            for index in indices:
                number = index + 1
                question = f"Что важно помнить по пункту №{number}?"
                answer = (
                    "Ответ будет расширен после полного завершения генерации скелета."
                )
                faq_entries.append({"q": question, "a": answer})
            return {"faq": faq_entries}
        if batch.kind == SkeletonBatchKind.CONCLUSION:
            return {
                "conclusion": (
                    "Вывод будет дополнен после завершения основной генерации текста."
                )
            }
        return None

    def _tail_fill_batch(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        estimate: SkeletonVolumeEstimate,
        missing_items: Sequence[int],
        metadata: Dict[str, object],
    ) -> None:
        pending = [int(item) for item in missing_items if isinstance(item, int)]
        if not pending:
            return
        if batch.kind == SkeletonBatchKind.MAIN:
            self._tail_fill_main_sections(
                indices=pending,
                outline=outline,
                assembly=assembly,
                estimate=estimate,
            )
            return
        previous_id = self._metadata_response_id(metadata)
        if not previous_id and batch.kind == SkeletonBatchKind.FAQ:
            budget = self._batch_token_budget(batch, estimate, 1)
            max_tokens = max(320, min(budget, TAIL_FILL_MAX_TOKENS))
            for index in pending:
                fallback_plan = SkeletonBatchPlan(
                    kind=SkeletonBatchKind.FAQ,
                    indices=[index],
                    label=self._format_batch_label(SkeletonBatchKind.FAQ, [index]),
                    tail_fill=True,
                )
                payload, result = self._run_fallback_batch(
                    fallback_plan,
                    outline=outline,
                    assembly=assembly,
                    target_indices=[index],
                    max_tokens=max_tokens,
                    previous_response_id=None,
                )
                if payload is None or result is None:
                    continue
                normalized_entries, _ = self._normalize_faq_batch(payload, [index])
                for _, question, answer in normalized_entries:
                    assembly.apply_faq(question, answer)
            return
        if not previous_id:
            return
        if batch.kind in (SkeletonBatchKind.MAIN, SkeletonBatchKind.FAQ):
            groups = [[index] for index in pending]
        else:
            groups = [list(pending)]

        for group in groups:
            if not group:
                continue
            tail_plan = SkeletonBatchPlan(
                kind=batch.kind,
                indices=list(group),
                label=batch.label + "#tail",
                tail_fill=True,
            )
            messages, format_block = self._build_batch_messages(
                tail_plan,
                outline=outline,
                assembly=assembly,
                target_indices=list(group),
                tail_fill=True,
            )
            budget = self._batch_token_budget(batch, estimate, len(group))
            max_tokens = max(400, min(budget, TAIL_FILL_MAX_TOKENS, 800))
            LOGGER.info(
                "TAIL_FILL missing=%d items=%s max_tokens=%d",
                len(group),
                ",".join(str(item + 1) for item in group),
                max_tokens,
            )
            attempts = 0
            payload: Optional[object] = None
            tail_metadata: Dict[str, object] = {}
            current_limit = max_tokens
            result: Optional[GenerationResult] = None
            while attempts < 3:
                attempts += 1
                result = self._call_llm(
                    step=PipelineStep.SKELETON,
                    messages=messages,
                    max_tokens=current_limit,
                    previous_response_id=previous_id,
                    responses_format=format_block,
                    allow_incomplete=True,
                )
                tail_metadata = result.metadata or {}
                payload = self._extract_response_json(result.text)
                status = str(tail_metadata.get("status") or "")
                reason = str(tail_metadata.get("incomplete_reason") or "")
                is_incomplete = status.lower() == "incomplete" or bool(reason)
                if not is_incomplete or self._batch_has_payload(batch.kind, payload):
                    break
                current_limit = max(200, int(current_limit * 0.85))
            if result is None:
                raise PipelineStepError(
                    PipelineStep.SKELETON,
                    "Сбой tail-fill: модель не ответила.",
                )
            if batch.kind == SkeletonBatchKind.MAIN:
                normalized, missing = self._normalize_main_batch(payload, list(group), outline)
                for index, heading, body in normalized:
                    assembly.apply_main(index, body, heading=heading)
                if missing:
                    raise PipelineStepError(
                        PipelineStep.SKELETON,
                        "Не удалось достроить все разделы основной части.",
                    )
            elif batch.kind == SkeletonBatchKind.FAQ:
                normalized, missing = self._normalize_faq_batch(payload, list(group))
                for _, question, answer in normalized:
                    assembly.apply_faq(question, answer)
                if missing:
                    raise PipelineStepError(
                        PipelineStep.SKELETON,
                        "Не удалось достроить все элементы FAQ.",
                    )
            elif batch.kind == SkeletonBatchKind.INTRO:
                normalized, missing_fields = self._normalize_intro_batch(payload, outline)
                if normalized.get("intro"):
                    headers = normalized.get("main_headers") or []
                    if len(headers) < len(outline.main_headings):
                        headers = headers + outline.main_headings[len(headers) :]
                    assembly.apply_intro(
                        normalized.get("intro"),
                        headers,
                        normalized.get("conclusion_heading"),
                    )
                if missing_fields:
                    raise PipelineStepError(
                        PipelineStep.SKELETON,
                        "Не удалось завершить вводный блок скелета.",
                    )
            else:
                conclusion_text, missing = self._normalize_conclusion_batch(payload)
                if conclusion_text:
                    assembly.apply_conclusion(conclusion_text)
                if missing:
                    raise PipelineStepError(
                        PipelineStep.SKELETON,
                        "Не удалось завершить вывод скелета.",
                    )
            self._apply_inline_faq(payload, assembly)
            for key, value in tail_metadata.items():
                if value:
                    metadata[key] = value

    def _tail_fill_main_sections(
        self,
        *,
        indices: Sequence[int],
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        estimate: SkeletonVolumeEstimate,
    ) -> None:
        targets = [idx for idx in indices if 0 <= idx < len(outline.main_headings)]
        if not targets:
            return
        LOGGER.info("LOG:MAIN_TAIL_FILL start missing=%d", len(targets))
        for index in targets:
            heading = (
                outline.main_headings[index]
                if index < len(outline.main_headings)
                else f"Блок {index + 1}"
            )
            messages, format_block = self._build_main_tail_fill_messages(
                outline=outline,
                assembly=assembly,
                target_index=index,
                heading=heading,
            )
            budget = max(600, min(estimate.per_main_tokens + 160, 900))
            result = self._call_llm(
                step=PipelineStep.SKELETON,
                messages=messages,
                max_tokens=budget,
                responses_format=format_block,
            )
            payload = self._extract_response_json(result.text)
            format_used = "json"
            resolved_heading = heading
            body_text = ""
            if isinstance(payload, dict):
                section_payload: Optional[Dict[str, object]] = None
                section_block = payload.get("section")
                if isinstance(section_block, dict):
                    section_payload = section_block
                else:
                    sections = payload.get("sections")
                    if isinstance(sections, list) and sections:
                        candidate = sections[0]
                        if isinstance(candidate, dict):
                            section_payload = candidate
                if section_payload is not None:
                    heading_candidate = str(
                        section_payload.get("title")
                        or section_payload.get("heading")
                        or ""
                    ).strip()
                    body_candidate = str(
                        section_payload.get("body")
                        or section_payload.get("text")
                        or ""
                    ).strip()
                    if heading_candidate:
                        resolved_heading = heading_candidate
                    if body_candidate:
                        body_text = body_candidate
            if not body_text:
                fallback = self._parse_fallback_main(
                    result.text,
                    target_index=index,
                    outline=outline,
                )
                if fallback and isinstance(fallback, dict):
                    sections = fallback.get("sections")
                    if isinstance(sections, list) and sections:
                        candidate = sections[0]
                        if isinstance(candidate, dict):
                            heading_candidate = str(
                                candidate.get("title")
                                or candidate.get("heading")
                                or ""
                            ).strip()
                            body_candidate = str(
                                candidate.get("body")
                                or candidate.get("text")
                                or ""
                            ).strip()
                            if heading_candidate:
                                resolved_heading = heading_candidate
                            if body_candidate:
                                body_text = body_candidate
                                format_used = "text"
            if not body_text:
                body_text = (
                    "Этот раздел будет дополнен подробными рекомендациями по теме."
                    " Пока используем короткий абзац-заглушку."
                )
                format_used = "stub"
            assembly.apply_main(index, body_text, heading=resolved_heading)
            usage = self._extract_usage(result)
            tokens_repr = "-" if usage is None else f"{int(round(usage))}"
            LOGGER.info(
                "LOG:MAIN_TAIL_FILL_ACCEPT idx=%d tokens=%s format=%s",
                index + 1,
                tokens_repr,
                format_used,
            )

    def _build_main_tail_fill_messages(
        self,
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        target_index: int,
        heading: str,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        messages = [dict(message) for message in self.messages]
        existing_sections: List[str] = []
        for idx, title in enumerate(outline.main_headings):
            if idx == target_index:
                continue
            body = assembly.main_sections[idx] if idx < len(assembly.main_sections) else None
            if body and str(body).strip():
                existing_sections.append(f"{idx + 1}. {title}")
        lines = [
            "Ты дополняешь основной раздел SEO-статьи.",
            f"Тема: {self.topic}.",
            "Нужно подготовить одну новую секцию основной части, без повторов уже готовых блоков.",
            f"Целевой заголовок: {heading}.",
        ]
        if existing_sections:
            lines.append("Уже готовые разделы: " + "; ".join(existing_sections) + ".")
        lines.extend(
            [
                "Секция должна содержать 3–4 абзаца по 3–4 предложения, с цифрами, рисками и действиями.",
                "Сконцентрируйся на практических советах и логичной структуре.",
                "Верни JSON {\"section\": {\"title\": str, \"body\": str}} без пояснений.",
                "Строго одна новая секция основной части, без повторов.",
            ]
        )
        if self.normalized_keywords:
            lines.append(
                "По возможности используй ключевые слова: "
                + ", ".join(self.normalized_keywords)
                + "."
            )
        user_payload = textwrap.dedent("\n".join(lines)).strip()
        messages.append({"role": "user", "content": user_payload})
        schema = {
            "type": "object",
            "properties": {
                "section": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title", "body"],
                    "additionalProperties": False,
                }
            },
            "required": ["section"],
            "additionalProperties": False,
        }
        format_block = {
            "type": "json_schema",
            "name": "seo_article_main_batch_one",
            "schema": schema,
            "strict": True,
        }
        plan = SkeletonBatchPlan(
            kind=SkeletonBatchKind.MAIN,
            indices=[target_index],
            label=f"main[{target_index + 1}]",
            tail_fill=True,
        )
        return messages, self._prepare_format_block(format_block, batch=plan)

    def _append_main_heading_to_base_outline(self, heading: str) -> None:
        normalized = str(heading or "").strip()
        if not normalized:
            return
        if normalized in self.base_outline:
            return
        faq_markers = {"faq", "f.a.q.", "вопросы и ответы"}
        insert_pos = len(self.base_outline)
        for idx in range(1, len(self.base_outline)):
            marker = str(self.base_outline[idx] or "").strip().lower()
            if marker in faq_markers:
                insert_pos = idx
                break
        if insert_pos <= 0:
            insert_pos = 1
        if insert_pos >= len(self.base_outline):
            self.base_outline.append(normalized)
        else:
            self.base_outline.insert(insert_pos, normalized)

    def _generate_additional_main_sections(
        self,
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        estimate: SkeletonVolumeEstimate,
        count: int,
    ) -> List[str]:
        generated: List[str] = []
        target_total = max(0, int(count))
        for _ in range(target_total):
            new_heading = f"Дополнительный блок {len(outline.main_headings) + 1}"
            outline.main_headings.append(new_heading)
            assembly.main_sections.append(None)
            self._append_main_heading_to_base_outline(new_heading)
            new_index = len(assembly.main_sections) - 1
            self._tail_fill_main_sections(
                indices=[new_index],
                outline=outline,
                assembly=assembly,
                estimate=estimate,
            )
            body_text = assembly.main_sections[new_index]
            if isinstance(body_text, str) and body_text.strip():
                generated.append(body_text.strip())
            else:
                placeholder = (
                    "Дополнительный раздел будет дополнен подробными рекомендациями "
                    "в обновлении статьи."
                )
                assembly.apply_main(new_index, placeholder, heading=new_heading)
                generated.append(placeholder)
        return generated

    def _finalize_main_sections(
        self,
        payload: Dict[str, object],
        *,
        outline: SkeletonOutline,
        assembly: SkeletonAssembly,
        estimate: SkeletonVolumeEstimate,
    ) -> Dict[str, object]:
        main_blocks = payload.get("main")
        if not isinstance(main_blocks, list):
            main_blocks = []
        sanitized = [str(item or "").strip() for item in main_blocks if str(item or "").strip()]
        before_len = len(sanitized)
        result = list(sanitized)
        if len(result) > 6:
            LOGGER.info("LOG:SKELETON_MAIN_TRIMMED from=%d to=6", len(result))
            result = result[:6]
        needed = max(0, min(3 - len(result), 3))
        if needed > 0:
            additional = self._generate_additional_main_sections(
                outline=outline,
                assembly=assembly,
                estimate=estimate,
                count=needed,
            )
            if additional:
                result.extend(additional)
            LOGGER.info(
                "LOG:SKELETON_MAIN_AUTOFIX needed=%d before=%d after=%d",
                needed,
                before_len,
                len(result),
            )
        payload["main"] = result
        return payload

    def _render_skeleton_markdown(self, payload: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
        if not isinstance(payload, dict):
            raise ValueError("Структура скелета не является объектом")

        intro = str(payload.get("intro") or "").strip()
        raw_main = payload.get("main")
        if not isinstance(raw_main, list):
            raw_main = []
        conclusion = str(payload.get("conclusion") or "").strip()
        faq = payload.get("faq")
        if not intro or not conclusion:
            raise ValueError("Скелет не содержит обязательных полей intro/main/conclusion")

        normalized_main: List[str] = [
            str(item or "").strip() for item in raw_main if str(item or "").strip()
        ]
        if len(normalized_main) > 6:
            normalized_main = normalized_main[:6]
        placeholders_needed = max(0, 3 - len(normalized_main))
        if placeholders_needed:
            for _ in range(placeholders_needed):
                normalized_main.append(
                    "Этот раздел будет расширен детальными рекомендациями в финальной версии статьи."
                )
            LOGGER.warning(
                "LOG:SKELETON_MAIN_PLACEHOLDER applied count=%d",
                placeholders_needed,
            )

        expected_faq = self.faq_target if self.faq_target > 0 else 0
        if expected_faq > 0:
            if not isinstance(faq, list) or len(faq) != expected_faq:
                raise ValueError(
                    f"Скелет FAQ должен содержать ровно {expected_faq} элементов"
                )
        else:
            faq = []

        normalized_faq: List[Dict[str, str]] = []
        for idx, entry in enumerate(faq, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"FAQ элемент №{idx} имеет неверный формат")
            question = str(entry.get("q") or "").strip()
            answer = str(entry.get("a") or "").strip()
            if not question or not answer:
                raise ValueError(f"FAQ элемент №{idx} пуст")
            normalized_faq.append({"question": question, "answer": answer})

        outline = [segment.strip() for segment in self.base_outline if segment.strip()]
        outline = [
            entry
            for entry in outline
            if entry.lower() not in {"faq", "f.a.q.", "вопросы и ответы"}
        ]
        if len(outline) < 3:
            outline = ["Введение", "Основная часть", "Вывод"]

        intro_heading = outline[0]
        conclusion_heading = outline[-1]
        main_headings = outline[1:-1]
        if not main_headings:
            main_headings = ["Основная часть"]
        if len(main_headings) < len(normalized_main):
            extra = len(normalized_main) - len(main_headings)
            for index in range(extra):
                main_headings.append(f"Блок {len(main_headings) + 1}")
        elif len(main_headings) > len(normalized_main):
            main_headings = main_headings[: len(normalized_main)]

        lines: List[str] = [f"# {self.topic}", ""]

        def _append_section(heading: str, content: str) -> None:
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part.strip()]
            if not paragraphs:
                raise ValueError(f"Раздел '{heading}' пуст")
            lines.append(f"## {heading}")
            for paragraph in paragraphs:
                lines.append(paragraph)
                lines.append("")

        _append_section(intro_heading, intro)
        for heading, body in zip(main_headings, normalized_main):
            _append_section(heading, body)
        _append_section(conclusion_heading, conclusion)

        lines.append("## FAQ")
        lines.append(FAQ_START)
        lines.append(FAQ_END)
        markdown = "\n".join(lines).strip()
        outline_summary = [intro_heading, *main_headings, conclusion_heading]
        self._skeleton_faq_entries = normalized_faq
        return markdown, {"outline": outline_summary, "faq": normalized_faq}

    def _render_faq_markdown(self, entries: Sequence[Dict[str, str]]) -> str:
        lines: List[str] = []
        for index, entry in enumerate(entries, start=1):
            question = entry.get("question", "").strip()
            answer = entry.get("answer", "").strip()
            lines.append(f"**Вопрос {index}.** {question}")
            lines.append(f"**Ответ.** {answer}")
            lines.append("")
        return "\n".join(lines).strip()

    def _build_jsonld(self, entries: Sequence[Dict[str, str]]) -> str:
        payload = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": entry.get("question", ""),
                    "acceptedAnswer": {"@type": "Answer", "text": entry.get("answer", "")},
                }
                for entry in entries
            ],
        }
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f'<script type="application/ld+json">\n{compact}\n</script>'

    def _merge_faq(self, base_text: str, faq_block: str) -> str:
        if FAQ_START not in base_text or FAQ_END not in base_text:
            raise PipelineStepError(PipelineStep.FAQ, "В тексте нет маркеров FAQ для замены.")
        before, remainder = base_text.split(FAQ_START, 1)
        inside, after = remainder.split(FAQ_END, 1)
        inside = inside.strip()
        merged = f"{before}{FAQ_START}\n{faq_block}\n{FAQ_END}{after}"
        return merged

    def _sanitize_entries(self, entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        sanitized: List[Dict[str, str]] = []
        seen_questions: set[str] = set()
        for entry in entries:
            try:
                normalized = _normalize_entry(dict(entry), seen_questions)
            except ValueError as exc:
                raise PipelineStepError(
                    PipelineStep.FAQ,
                    f"Некорректный FAQ: {exc}.",
                ) from exc
            question = normalized.question.strip()
            answer = normalized.answer.strip()
            lowered = (question + " " + answer).lower()
            if "дополнительно рассматривается" in lowered:
                raise PipelineStepError(
                    PipelineStep.FAQ,
                    "FAQ содержит шаблонную фразу 'Дополнительно рассматривается'.",
                )
            sanitized.append({"question": question, "answer": answer})
        return sanitized

    def _parse_faq_entries(self, raw_text: str) -> List[Dict[str, str]]:
        candidate = raw_text.strip()
        if not candidate:
            raise PipelineStepError(PipelineStep.FAQ, "Модель вернула пустой блок FAQ.")
        data: Optional[Dict[str, object]] = None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match:
                data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise PipelineStepError(PipelineStep.FAQ, "Ответ модели не является корректным JSON.")
        entries = data.get("faq")
        if not isinstance(entries, list):
            raise PipelineStepError(PipelineStep.FAQ, "В ответе отсутствует массив faq.")
        sanitized = self._sanitize_entries(entries)
        expected = self.faq_target if self.faq_target > 0 else 0
        if expected > 0:
            if len(sanitized) != expected:
                raise PipelineStepError(
                    PipelineStep.FAQ,
                    f"FAQ должно содержать ровно {expected} пар вопросов и ответов.",
                )
            return sanitized
        return []

    def _build_faq_messages(self, base_text: str) -> List[Dict[str, str]]:
        hints: List[str] = []
        if self.provided_faq:
            provided_preview = json.dumps(
                [
                    {
                        "question": str(entry.get("question", "")).strip(),
                        "answer": str(entry.get("answer", "")).strip(),
                    }
                    for entry in self.provided_faq
                    if str(entry.get("question", "")).strip() and str(entry.get("answer", "")).strip()
                ],
                ensure_ascii=False,
                indent=2,
            )
            hints.append(
                "Используй следующие пары как ориентир и улучшай формулировки, если нужно:\n" + provided_preview
            )
        if self.normalized_keywords:
            hints.append(
                "По возможности вплетай ключевые слова: " + ", ".join(self.normalized_keywords) + "."
            )

        user_instructions = [
            "Ниже приведена статья без блока FAQ. Сформируй пять уникальных вопросов и ответов.",
            "Верни результат в формате JSON: {\"faq\": [{\"question\": \"...\", \"answer\": \"...\"}, ...]}.",
            "Ответы должны быть развернутыми, практичными и без повторов.",
            "Не используй клише вроде 'Дополнительно рассматривается'.",
        ]
        if hints:
            user_instructions.extend(hints)
        payload = "\n".join(user_instructions)
        article_block = f"СТАТЬЯ:\n{base_text.strip()}"
        return [
            {
                "role": "system",
                "content": (
                    "Ты опытный финансовый редактор. Сформируй полезный FAQ без повторов,"
                    " обеспечь, чтобы вопросы отличались по фокусу и помогали читателю действовать."
                ),
            },
            {"role": "user", "content": f"{payload}\n\n{article_block}"},
        ]

    def _sync_locked_terms(self, text: str) -> None:
        pattern = re.compile(r"<!--LOCK_START term=\"([^\"]+)\"-->")
        discovered = pattern.findall(text)
        seen: set[str] = set()
        self.locked_terms = []
        for term in discovered:
            if term and term not in seen:
                self.locked_terms.append(term)
                seen.add(term)
        coverage: KeywordCoverage = evaluate_keyword_coverage(
            text,
            self.required_keywords,
            preferred=self.preferred_keywords,
        )
        self.keywords_required_coverage_percent = coverage.required_percent
        self.keywords_coverage_percent = coverage.overall_percent


    @staticmethod
    def _split_jsonld_block(text: str) -> Tuple[str, str]:
        match = _JSONLD_PATTERN.search(text)
        if not match:
            return text, ""
        jsonld = match.group(0).strip()
        before = text[: match.start()].rstrip()
        after = text[match.end() :].lstrip()
        article = before
        if after:
            article = f"{article}\n\n{after}" if article else after
        return article, jsonld

    @staticmethod
    def _is_intro_heading(title: str) -> bool:
        normalized = title.strip().lower()
        return normalized in {"введение", "вступление", "introduction", "intro"}

    @staticmethod
    def _is_conclusion_heading(title: str) -> bool:
        normalized = title.strip().lower()
        return normalized in {"заключение", "вывод", "итоги", "заключение и рекомендации"}

    @staticmethod
    def _inject_sentence_into_paragraph(body: str, sentence: str, *, tail: bool) -> str:
        paragraphs = [segment for segment in re.split(r"\n\s*\n", body.strip()) if segment.strip()]
        if not paragraphs:
            return body
        index = -1 if tail else 0
        target = paragraphs[index].rstrip()
        separator = " " if target.endswith(tuple(".!?")) else ". "
        paragraphs[index] = target + separator + sentence.strip()
        return "\n\n".join(paragraphs)

    @staticmethod
    def _compose_reinforcement_sentence(terms: Sequence[str], *, intro: bool) -> str:
        if not terms:
            return ""
        if len(terms) == 1:
            base = terms[0]
            if intro:
                return f"Отдельно подчёркиваем {base} — это помогает задать нужный контекст."
            return f"Дополнительно фиксируем {base}, чтобы итоговый вывод оставался конкретным."
        head = ", ".join(terms[:-1])
        last = terms[-1]
        joined = f"{head} и {last}" if head else last
        if intro:
            return (
                f"Также отмечаем {joined} — эти акценты помогают читателю сразу увидеть ключевые ориентиры."
            )
        return f"В финале напоминаем про {joined}, чтобы сохранить целостность аргументации."

    def _reinforce_keywords(self, text: str, missing_terms: Sequence[str]) -> str:
        if not missing_terms:
            return text
        article, jsonld = self._split_jsonld_block(text)
        if not article.strip():
            return text
        terms = [term for term in missing_terms if term]
        if not terms:
            return text
        midpoint = (len(terms) + 1) // 2
        intro_terms = terms[:midpoint]
        conclusion_terms = terms[midpoint:]

        pattern = re.compile(
            r"(?P<header>##\s+(?P<title>[^\n]+))\n(?P<body>.*?)(?=\n##\s+|\Z)",
            re.DOTALL,
        )

        def _replace(match: re.Match[str]) -> str:
            header = match.group("header")
            title = match.group("title")
            body = match.group("body")
            if self._is_intro_heading(title) and intro_terms:
                sentence = self._compose_reinforcement_sentence(intro_terms, intro=True)
                updated_body = self._inject_sentence_into_paragraph(body, sentence, tail=False)
                return f"{header}\n{updated_body}"
            if self._is_conclusion_heading(title) and conclusion_terms:
                sentence = self._compose_reinforcement_sentence(conclusion_terms, intro=False)
                updated_body = self._inject_sentence_into_paragraph(body, sentence, tail=True)
                return f"{header}\n{updated_body}"
            return match.group(0)

        updated_article = pattern.sub(_replace, article)
        if updated_article == article:
            return text
        updated_article = updated_article.rstrip() + "\n"
        if jsonld:
            return f"{updated_article}\n{jsonld}\n"
        return updated_article

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _run_skeleton(self) -> str:
        self._log(PipelineStep.SKELETON, "running")
        outline = self._prepare_outline()
        estimate = self._predict_skeleton_volume(outline)
        batches = self._build_skeleton_batches(outline, estimate)
        assembly = SkeletonAssembly(outline=outline)
        metadata_snapshot: Dict[str, object] = {}
        last_result: Optional[GenerationResult] = None

        pending_batches = deque(batches)
        scheduled_main_indices: Set[int] = set()
        parse_none_streaks: Dict[str, int] = {}
        for plan in pending_batches:
            if plan.kind == SkeletonBatchKind.MAIN:
                scheduled_main_indices.update(plan.indices)
        split_serial = 0
        tail_fill_allowed = False

        while pending_batches:
            batch = pending_batches.popleft()
            if batch.kind in (SkeletonBatchKind.FAQ, SkeletonBatchKind.CONCLUSION):
                filled_main = sum(
                    1
                    for body in assembly.main_sections
                    if isinstance(body, str) and body.strip()
                )
                has_pending_main = any(
                    plan.kind == SkeletonBatchKind.MAIN for plan in pending_batches
                )
                if filled_main < 3 and has_pending_main:
                    LOGGER.info(
                        "LOG:SCHEDULER_BLOCK main underflow=%d target_min=3 → continue_main",
                        filled_main,
                    )
                    pending_batches.append(batch)
                    continue
            if not batch.label:
                batch.label = self._format_batch_label(batch.kind, batch.indices)
            active_indices = list(batch.indices)
            limit_override: Optional[int] = None
            override_to_cap = False
            retries = 0
            consecutive_empty_incomplete = 0
            payload_obj: Optional[object] = None
            metadata_snapshot = {}
            result: Optional[GenerationResult] = None
            last_max_tokens = estimate.start_max_tokens
            continuation_id: Optional[str] = None
            batch_partial = False
            first_attempt_for_batch = True
            best_payload_obj: Optional[object] = None
            best_result: Optional[GenerationResult] = None
            best_metadata_snapshot: Dict[str, object] = {}
            last_reason_lower = ""
            forced_tail_indices: List[int] = []

            while True:
                messages, format_block = self._build_batch_messages(
                    batch,
                    outline=outline,
                    assembly=assembly,
                    target_indices=active_indices,
                    tail_fill=batch.tail_fill,
                )
                base_budget = self._batch_token_budget(batch, estimate, len(active_indices) or 1)
                if first_attempt_for_batch:
                    max_tokens_to_use = estimate.start_max_tokens
                else:
                    max_tokens_to_use = base_budget
                if limit_override is not None:
                    if override_to_cap:
                        max_tokens_to_use = max(max_tokens_to_use, limit_override)
                    else:
                        max_tokens_to_use = min(max_tokens_to_use, limit_override)
                last_max_tokens = max_tokens_to_use
                first_attempt_for_batch = False
                request_prev_id = continuation_id or ""
                result = self._call_llm(
                    step=PipelineStep.SKELETON,
                    messages=messages,
                    max_tokens=max_tokens_to_use,
                    previous_response_id=continuation_id,
                    responses_format=format_block,
                    allow_incomplete=True,
                )
                last_result = result
                metadata_snapshot = result.metadata or {}
                response_id_candidate = self._metadata_response_id(metadata_snapshot)
                if response_id_candidate:
                    continuation_id = response_id_candidate
                payload_obj = self._extract_response_json(result.text)
                status = str(metadata_snapshot.get("status") or "")
                reason = str(metadata_snapshot.get("incomplete_reason") or "")
                reason_lower = reason.strip().lower()
                last_reason_lower = reason_lower
                if reason_lower == "max_output_tokens":
                    tail_fill_allowed = True
                is_incomplete = status.lower() == "incomplete" or bool(reason)
                has_payload = self._batch_has_payload(batch.kind, payload_obj)
                if payload_obj is not None and has_payload:
                    best_payload_obj = payload_obj
                    best_result = result
                    best_metadata_snapshot = dict(metadata_snapshot)
                metadata_prev_id = str(
                    metadata_snapshot.get("previous_response_id")
                    or request_prev_id
                    or ""
                )
                schema_label = str(result.schema or "")
                schema_is_none = schema_label.endswith(".none")
                parse_none_count = 0
                if metadata_prev_id:
                    if schema_is_none and is_incomplete and not has_payload:
                        parse_none_count = parse_none_streaks.get(metadata_prev_id, 0) + 1
                        parse_none_streaks[metadata_prev_id] = parse_none_count
                    else:
                        parse_none_streaks.pop(metadata_prev_id, None)
                if not is_incomplete or has_payload:
                    batch_partial = bool(is_incomplete and has_payload)
                    if metadata_prev_id:
                        parse_none_streaks.pop(metadata_prev_id, None)
                    if request_prev_id and request_prev_id != metadata_prev_id:
                        parse_none_streaks.pop(request_prev_id, None)
                    break
                consecutive_empty_incomplete += 1
                should_autosplit = False
                if self._can_split_batch(batch.kind, active_indices) and len(active_indices) > 1:
                    if reason_lower == "max_output_tokens" and consecutive_empty_incomplete >= 1:
                        should_autosplit = True
                    elif consecutive_empty_incomplete >= 2 and parse_none_count >= 2:
                        should_autosplit = True
                if should_autosplit:
                    keep, remainder = self._split_batch_indices(active_indices)
                    original_size = len(active_indices)
                    if remainder:
                        split_serial += 1
                        remainder_label = self._format_batch_label(
                            batch.kind,
                            remainder,
                            suffix=f"#split{split_serial}",
                        )
                        pending_batches.appendleft(
                            SkeletonBatchPlan(
                                kind=batch.kind,
                                indices=list(remainder),
                                label=remainder_label,
                                tail_fill=batch.tail_fill,
                            )
                        )
                    LOGGER.info(
                        "BATCH_AUTOSPLIT kind=%s label=%s from=%d to=%d",
                        batch.kind.value,
                        batch.label or self._format_batch_label(batch.kind, active_indices),
                        original_size,
                        len(keep) or 0,
                    )
                    active_indices = keep
                    batch.indices = list(keep)
                    batch.label = self._format_batch_label(batch.kind, keep)
                    limit_override = None
                    override_to_cap = False
                    retries = 0
                    consecutive_empty_incomplete = 0
                    first_attempt_for_batch = True
                    if metadata_prev_id:
                        parse_none_streaks.pop(metadata_prev_id, None)
                    continue
                should_trigger_fallback = (
                    len(active_indices) == 1
                    and consecutive_empty_incomplete >= 2
                    and (
                        reason_lower == "max_output_tokens"
                        or (metadata_prev_id and parse_none_count >= 2)
                    )
                )
                if should_trigger_fallback:
                    fallback_payload, fallback_result = self._run_fallback_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        target_indices=active_indices,
                        max_tokens=max_tokens_to_use,
                        previous_response_id=continuation_id,
                    )
                    if fallback_payload is not None and fallback_result is not None:
                        payload_obj = fallback_payload
                        result = fallback_result
                        last_result = result
                        metadata_snapshot = result.metadata or {}
                        response_id_candidate = self._metadata_response_id(metadata_snapshot)
                        if response_id_candidate:
                            continuation_id = response_id_candidate
                        batch_partial = True
                        if metadata_prev_id:
                            parse_none_streaks.pop(metadata_prev_id, None)
                        break
                    if reason_lower == "max_output_tokens":
                        LOGGER.warning(
                            "SKELETON_FALLBACK_SKIPPED kind=%s label=%s reason=max_output_tokens",
                            batch.kind.value,
                            batch.label or self._format_batch_label(batch.kind, active_indices),
                        )
                        break
                    raise PipelineStepError(
                        PipelineStep.SKELETON,
                        "Fallback не дал валидный ответ для скелета.",
                    )
                retries += 1
                if retries >= 3:
                    LOGGER.warning(
                        "SKELETON_INCOMPLETE_WITHOUT_CONTENT kind=%s label=%s status=%s reason=%s",
                        batch.kind.value,
                        batch.label or self._format_batch_label(batch.kind, active_indices),
                        status or "incomplete",
                        reason or "",
                    )
                    break
                if reason_lower == "max_output_tokens":
                    cap_limit = estimate.cap_tokens or estimate.start_max_tokens
                    if cap_limit and cap_limit > 0:
                        limit_override = cap_limit
                    else:
                        limit_override = estimate.start_max_tokens
                    override_to_cap = True
                else:
                    limit_override = max(200, int(last_max_tokens * 0.85))
                    override_to_cap = False

            target_indices = list(active_indices)

            if (
                payload_obj is None
                and best_payload_obj is None
                and last_reason_lower == "max_output_tokens"
                and batch.kind in (SkeletonBatchKind.MAIN, SkeletonBatchKind.FAQ)
                and active_indices
            ):
                tail_fill_allowed = True
                cap_payload, cap_result = self._run_cap_fallback_batch(
                    batch,
                    outline=outline,
                    assembly=assembly,
                    target_indices=active_indices,
                    max_tokens=last_max_tokens,
                    previous_response_id=continuation_id,
                )
                if cap_payload is not None and cap_result is not None:
                    payload_obj = cap_payload
                    result = cap_result
                    metadata_snapshot = cap_result.metadata or {}
                    response_id_candidate = self._metadata_response_id(metadata_snapshot)
                    if response_id_candidate:
                        continuation_id = response_id_candidate
                        parse_none_streaks.pop(response_id_candidate, None)
                    batch_partial = True

            if payload_obj is None and best_payload_obj is not None:
                payload_obj = best_payload_obj
                result = best_result
                metadata_snapshot = dict(best_metadata_snapshot)
                batch_partial = True
                response_id_candidate = self._metadata_response_id(metadata_snapshot)
                if response_id_candidate:
                    parse_none_streaks.pop(response_id_candidate, None)
            if (
                payload_obj is None
                and best_payload_obj is None
                and last_reason_lower == "max_output_tokens"
            ):
                placeholder = self._build_batch_placeholder(
                    batch,
                    outline=outline,
                    target_indices=active_indices,
                )
                if placeholder is not None:
                    payload_obj = placeholder
                    batch_partial = True
                    if batch.kind in (SkeletonBatchKind.MAIN, SkeletonBatchKind.FAQ):
                        tail_fill_allowed = True
                        forced_tail_indices = list(active_indices)
                    LOGGER.warning(
                        "SKELETON_PLACEHOLDER_APPLIED kind=%s label=%s",
                        batch.kind.value,
                        batch.label,
                    )
            if payload_obj is None and batch.kind in (SkeletonBatchKind.MAIN, SkeletonBatchKind.FAQ) and active_indices:
                fallback_payload, fallback_result = self._run_fallback_batch(
                    batch,
                    outline=outline,
                    assembly=assembly,
                    target_indices=active_indices,
                    max_tokens=last_max_tokens,
                    previous_response_id=continuation_id,
                )
                if fallback_payload is not None and fallback_result is not None:
                    payload_obj = fallback_payload
                    result = fallback_result
                    metadata_snapshot = fallback_result.metadata or {}
                    response_id_candidate = self._metadata_response_id(metadata_snapshot)
                    if response_id_candidate:
                        continuation_id = response_id_candidate
                        parse_none_streaks.pop(response_id_candidate, None)
                    batch_partial = True
            if payload_obj is None:
                raise PipelineStepError(
                    PipelineStep.SKELETON,
                    "Скелет не содержит данных после генерации.",
                )

            if batch.kind == SkeletonBatchKind.INTRO:
                normalized, missing_fields = self._normalize_intro_batch(payload_obj, outline)
                intro_text = normalized.get("intro", "")
                headers = normalized.get("main_headers") or []
                if len(headers) < len(outline.main_headings):
                    headers = headers + outline.main_headings[len(headers) :]
                assembly.apply_intro(intro_text, headers, normalized.get("conclusion_heading"))
                current_total = len(assembly.main_sections)
                new_indices = [
                    idx for idx in range(current_total) if idx not in scheduled_main_indices
                ]
                if new_indices:
                    start_pos = 0
                    if self._should_force_single_main_batches(outline, estimate):
                        batch_size = 1
                    else:
                        batch_size = max(1, SKELETON_BATCH_SIZE_MAIN)
                    while start_pos < len(new_indices):
                        chunk = new_indices[start_pos : start_pos + batch_size]
                        if not chunk:
                            break
                        chunk_label = self._format_batch_label(SkeletonBatchKind.MAIN, chunk)
                        pending_batches.append(
                            SkeletonBatchPlan(
                                kind=SkeletonBatchKind.MAIN,
                                indices=list(chunk),
                                label=chunk_label,
                            )
                        )
                        scheduled_main_indices.update(chunk)
                        start_pos += batch_size
                if missing_fields:
                    if tail_fill_allowed:
                        self._tail_fill_batch(
                            batch,
                            outline=outline,
                            assembly=assembly,
                            estimate=estimate,
                            missing_items=[0],
                            metadata=metadata_snapshot,
                        )
                    else:
                        LOGGER.info(
                            "LOG:TAIL_FILL_SKIPPED reason=max_tokens_not_hit kind=%s items=%s",
                            batch.kind.value,
                            "0",
                        )
            elif batch.kind == SkeletonBatchKind.MAIN:
                normalized_sections, missing_indices = self._normalize_main_batch(
                    payload_obj, target_indices, outline
                )
                for index, heading, body in normalized_sections:
                    assembly.apply_main(index, body, heading=heading)
                if forced_tail_indices:
                    merged = list(dict.fromkeys(missing_indices + forced_tail_indices))
                else:
                    merged = list(missing_indices)
                if merged:
                    if not tail_fill_allowed:
                        LOGGER.info(
                            "LOG:TAIL_FILL_FORCED reason=missing_content kind=%s items=%s",
                            batch.kind.value,
                            ",".join(str(idx) for idx in merged),
                        )
                    self._tail_fill_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        estimate=estimate,
                        missing_items=merged,
                        metadata=metadata_snapshot,
                    )
            elif batch.kind == SkeletonBatchKind.FAQ:
                normalized_entries, missing_faq = self._normalize_faq_batch(payload_obj, target_indices)
                for _, question, answer in normalized_entries:
                    assembly.apply_faq(question, answer)
                if forced_tail_indices:
                    merged_faq = list(dict.fromkeys(missing_faq + forced_tail_indices))
                else:
                    merged_faq = list(missing_faq)
                if merged_faq:
                    if not tail_fill_allowed:
                        LOGGER.info(
                            "LOG:TAIL_FILL_FORCED reason=missing_content kind=%s items=%s",
                            batch.kind.value,
                            ",".join(str(idx) for idx in merged_faq),
                        )
                    self._tail_fill_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        estimate=estimate,
                        missing_items=merged_faq,
                        metadata=metadata_snapshot,
                    )
            else:
                conclusion_text, missing_flag = self._normalize_conclusion_batch(payload_obj)
                assembly.apply_conclusion(conclusion_text)
                if missing_flag:
                    if tail_fill_allowed:
                        self._tail_fill_batch(
                            batch,
                            outline=outline,
                            assembly=assembly,
                            estimate=estimate,
                            missing_items=[0],
                            metadata=metadata_snapshot,
                        )
                    else:
                        LOGGER.info(
                            "LOG:TAIL_FILL_SKIPPED reason=max_tokens_not_hit kind=%s items=%s",
                            batch.kind.value,
                            "0",
                        )

            self._apply_inline_faq(payload_obj, assembly)
            LOGGER.info(
                "BATCH_ACCEPT state=%s kind=%s label=%s",
                "partial" if batch_partial else "complete",
                batch.kind.value,
                batch.label,
            )

        if not assembly.intro:
            raise PipelineStepError(PipelineStep.SKELETON, "Не удалось получить вводный блок скелета.")
        if not assembly.conclusion:
            raise PipelineStepError(PipelineStep.SKELETON, "Не удалось получить вывод скелета.")
        missing_main = assembly.missing_main_indices()
        if missing_main:
            LOGGER.warning(
                "LOG:SKELETON_MAIN_GAPS missing=%s",
                ",".join(str(idx + 1) for idx in missing_main),
            )
            if tail_fill_allowed:
                self._tail_fill_main_sections(
                    indices=missing_main,
                    outline=outline,
                    assembly=assembly,
                    estimate=estimate,
                )
                missing_main = assembly.missing_main_indices()
            else:
                LOGGER.info(
                    "LOG:TAIL_FILL_SKIPPED reason=max_tokens_not_hit kind=%s items=%s",
                    "main",
                    ",".join(str(idx) for idx in missing_main),
                )
            if missing_main:
                for index in missing_main:
                    heading = (
                        outline.main_headings[index]
                        if 0 <= index < len(outline.main_headings)
                        else f"Раздел {index + 1}"
                    )
                    placeholder = (
                        f"Раздел «{heading}» будет дополнен после завершения генерации статьи."
                    )
                    assembly.apply_main(index, placeholder, heading=heading)
                LOGGER.warning(
                    "LOG:SKELETON_MAIN_PLACEHOLDER_FINAL count=%d",
                    len(missing_main),
                )
        if outline.has_faq and assembly.missing_faq_count(self.faq_target):
            raise PipelineStepError(
                PipelineStep.SKELETON,
                "Не удалось собрать полный FAQ на этапе скелета.",
            )

        payload = assembly.build_payload()
        if outline.has_faq and self.faq_target > 0:
            faq_items = payload.get("faq", [])
            if isinstance(faq_items, list) and len(faq_items) > self.faq_target:
                payload["faq"] = faq_items[: self.faq_target]

        normalized_payload = normalize_skeleton_payload(payload)
        normalized_payload = self._finalize_main_sections(
            normalized_payload,
            outline=outline,
            assembly=assembly,
            estimate=estimate,
        )
        markdown, summary = self._render_skeleton_markdown(normalized_payload)
        snapshot = dict(normalized_payload)
        snapshot["outline"] = summary.get("outline", [])
        if "faq" in summary:
            snapshot["faq"] = summary.get("faq", [])
        self.skeleton_payload = snapshot
        self._skeleton_faq_entries = [
            {"question": entry.get("q", ""), "answer": entry.get("a", "")}
            for entry in normalized_payload.get("faq", [])
        ]

        self._check_template_text(markdown, PipelineStep.SKELETON)
        route = last_result.api_route if last_result is not None else "responses"
        LOGGER.info("SKELETON_OK route=%s", route)
        self._update_log(
            PipelineStep.SKELETON,
            "ok",
            length=len(markdown),
            metadata_status=metadata_snapshot.get("status") or "ok",
            **self._metrics(markdown),
        )
        self.checkpoints[PipelineStep.SKELETON] = markdown
        return markdown

    def _run_keywords(self, text: str) -> KeywordInjectionResult:
        self._log(PipelineStep.KEYWORDS, "running")
        result = inject_keywords(
            text,
            self.required_keywords,
            preferred=self.preferred_keywords,
        )
        self.locked_terms = list(result.locked_terms)
        self.keywords_required_coverage_percent = result.coverage_percent
        self.keywords_coverage_percent = result.overall_coverage_percent
        missing_required = sorted(result.missing_terms)
        missing_preferred = sorted(result.missing_preferred)
        LOGGER.info(
            "KEYWORDS_COVERAGE required=%.0f%% overall=%.0f%% missing_required=%s missing_preferred=%s",
            result.coverage_percent,
            result.overall_coverage_percent,
            ",".join(missing_required) if missing_required else "-",
            ",".join(missing_preferred) if missing_preferred else "-",
        )
        if missing_required:
            raise PipelineStepError(
                PipelineStep.KEYWORDS,
                "Не удалось обеспечить 100% покрытие обязательных ключей: "
                + ", ".join(missing_required),
            )
        LOGGER.info(
            "KEYWORDS_OK coverage_required=%.2f%% coverage_overall=%.2f%%",
            result.coverage_percent,
            result.overall_coverage_percent,
        )
        self._update_log(
            PipelineStep.KEYWORDS,
            "ok",
            KEYWORDS_COVERAGE=result.coverage_report,
            KEYWORDS_COVERAGE_PERCENT=result.coverage_percent,
            KEYWORDS_COVERAGE_OVERALL=result.overall_coverage_percent,
            KEYWORDS_MISSING=missing_required,
            KEYWORDS_MISSING_PREFERRED=missing_preferred,
            inserted_section=result.inserted_section,
            **self._metrics(result.text),
        )
        self.checkpoints[PipelineStep.KEYWORDS] = result.text
        return result

    def _run_faq(self, text: str) -> str:
        self._log(PipelineStep.FAQ, "running")
        entries_source: List[Dict[str, str]] = []
        if self._skeleton_faq_entries:
            entries_source = list(self._skeleton_faq_entries)
        elif self.provided_faq:
            entries_source = [
                {"question": str(item.get("question", "")).strip(), "answer": str(item.get("answer", "")).strip()}
                for item in self.provided_faq
                if isinstance(item, dict)
            ]

        if entries_source and self.faq_target > 0:
            sanitized = self._sanitize_entries(entries_source)
            if len(sanitized) != self.faq_target:
                raise PipelineStepError(
                    PipelineStep.FAQ,
                    f"FAQ должно содержать ровно {self.faq_target} пар вопросов и ответов.",
                )
            faq_block = self._render_faq_markdown(sanitized)
            merged_text = self._merge_faq(text, faq_block)
            self.jsonld = self._build_jsonld(sanitized)
            self.jsonld_reserve = len(self.jsonld.replace(" ", "")) if self.jsonld else 0
            LOGGER.info("FAQ_OK entries=%s", ",".join(entry["question"] for entry in sanitized))
            self._update_log(
                PipelineStep.FAQ,
                "ok",
                entries=[entry["question"] for entry in sanitized],
                **self._metrics(merged_text),
            )
            self.checkpoints[PipelineStep.FAQ] = merged_text
            return merged_text

        if self.faq_target <= 0:
            LOGGER.info("FAQ_SKIPPED disabled for текущий запрос")
            self._update_log(
                PipelineStep.FAQ,
                "skipped",
                entries=[],
                **self._metrics(text),
            )
            self.checkpoints[PipelineStep.FAQ] = text
            self.jsonld = None
            self.jsonld_reserve = 0
            return text

        raise PipelineStepError(
            PipelineStep.FAQ,
            "Не удалось сформировать блок FAQ: отсутствуют подготовленные данные.",
        )

    def _build_fail_safe_article(self) -> str:
        topic = self.topic or "SEO-статья"
        intro = (
            "Эта резервная версия подготовлена автоматически. Она кратко передаёт ключевые "
            "смыслы темы и даёт базовые рекомендации по дальнейшему самостоятельному изучению."
        )
        main_sections = [
            (
                "Быстрый запуск",
                (
                    "Сформулируйте цель и определите метрики, по которым будете отслеживать прогресс. "
                    "Подготовьте рабочую таблицу с ответственными лицами, сроками и ожидаемыми результатами."
                ),
            ),
            (
                "Практическая проработка",
                (
                    "Разбейте внедрение на последовательные этапы, начиная с самых простых действий. "
                    "Используйте доступные инструменты аналитики и фиксируйте промежуточные выводы после каждой итерации."
                ),
            ),
            (
                "Контроль и корректировки",
                (
                    "Каждую неделю сопоставляйте ожидания с фактическими результатами, чтобы вовремя увидеть отклонения. "
                    "Фиксируйте инсайты и формируйте короткие резюме, которые помогут защитить инициативу перед командой."
                ),
            ),
        ]
        base_faq_entries = [
            (
                "С чего начать?",
                "Определите ключевую задачу и запишите стартовые показатели, от которых будете отталкиваться в анализе.",
            ),
            (
                "Как распределить ответственность?",
                "Назначьте владельца процесса и закрепите за ним контрольные точки. Остальным участникам оставьте конкретные действия.",
            ),
            (
                "Какие инструменты использовать?",
                "Выберите аналитические сервисы, с которыми команда уже знакома, чтобы не тратить время на внедрение сложных решений.",
            ),
            (
                "Как оценить первые результаты?",
                "Сравните фактические метрики с планом через неделю и месяц, отметьте, какие корректировки потребуются.",
            ),
            (
                "Что делать при задержках?",
                "Зафиксируйте причину, подготовьте три варианта компенсации и обсудите их с ключевыми заинтересованными сторонами.",
            ),
        ]
        if self.faq_target > 0:
            faq_entries = list(base_faq_entries)
            if len(faq_entries) < self.faq_target and faq_entries:
                seed = faq_entries[-1][1]
                for idx in range(len(faq_entries) + 1, self.faq_target + 1):
                    faq_entries.append(
                        (
                            f"Дополнительный вопрос {idx}?",
                            f"Раскройте нюансы подхода, опираясь на резервный сценарий: {seed}",
                        )
                    )
            faq_entries = faq_entries[: self.faq_target]
        else:
            faq_entries = []
        conclusion = (
            "Зафиксируйте выводы, выберите одну метрику оперативного контроля и договоритесь о дате следующей оценки результатов. "
            "Запишите идеи для последующих улучшений, чтобы постепенно расширять эффект от инициативы."
        )

        lines: List[str] = [f"# {topic}", "", "## Введение", intro, ""]
        for heading, paragraph in main_sections:
            lines.extend([f"## {heading}", paragraph, ""])
        if faq_entries:
            lines.extend(["## FAQ", "<!--FAQ_START-->"])
            for index, (question, answer) in enumerate(faq_entries, start=1):
                lines.append(f"**Вопрос {index}.** {question}")
                lines.append(f"**Ответ.** {answer}")
                lines.append("")
            if lines and lines[-1] == "":
                lines.pop()
            lines.extend(["<!--FAQ_END-->", ""])
        lines.extend(["## Заключение", conclusion, ""])
        article = "\n".join(lines).strip() + "\n"
        controller = ensure_article_length(
            article,
            min_chars=self.min_chars,
            max_chars=self.max_chars,
            protected_blocks=self.locked_terms,
            faq_expected=self.faq_target,
            exact_chars=self.min_chars if self.min_chars == self.max_chars else None,
        )
        return controller.text if controller.text else article

    def _run_trim(self, text: str) -> TrimResult:
        self._log(PipelineStep.TRIM, "running")
        reserve = self.jsonld_reserve if self.jsonld else 0
        hard_cap = ARTICLE_HARD_CHAR_CAP if ARTICLE_HARD_CHAR_CAP > 0 else None
        effective_upper = self.max_chars
        if hard_cap is not None:
            effective_upper = min(effective_upper, hard_cap)
        target_max = max(self.min_chars, effective_upper - reserve)
        try:
            result = trim_text(
                text,
                min_chars=self.min_chars,
                max_chars=target_max,
                protected_blocks=self.locked_terms,
                faq_expected=self.faq_target,
                required_terms=self.required_keywords,
                preferred_terms=self.preferred_keywords,
            )
        except TrimValidationError as exc:
            raise PipelineStepError(PipelineStep.TRIM, str(exc)) from exc
        current_length = length_no_spaces(result.text)

        soft_min, soft_max, tolerance_below, tolerance_above = compute_soft_length_bounds(
            self.min_chars, self.max_chars
        )
        effective_max = self.max_chars
        length_notes: Dict[str, object] = {}
        if getattr(result, "length_relaxed", False):
            effective_max = max(effective_max, int(result.relaxed_limit or self.max_chars))
            length_notes["length_relaxed"] = True
            length_notes["length_relaxed_limit"] = effective_max
            length_notes["length_relaxed_status"] = "accepted"
        strict_violation = current_length < self.min_chars or current_length > effective_max
        if strict_violation:
            controller = ensure_article_length(
                result.text,
                min_chars=self.min_chars,
                max_chars=effective_max,
                protected_blocks=self.locked_terms,
                faq_expected=self.faq_target,
                exact_chars=self.min_chars if self.min_chars == self.max_chars else None,
            )
            if controller.adjusted:
                length_notes["length_controller_adjusted"] = True
                length_notes["length_controller_iterations"] = controller.iterations
                length_notes["length_controller_history"] = list(controller.history)
                length_notes["length_controller_success"] = controller.success
                result = TrimResult(
                    text=controller.text,
                    removed_paragraphs=result.removed_paragraphs,
                    length_relaxed=getattr(result, "length_relaxed", False),
                    relaxed_limit=getattr(result, "relaxed_limit", None),
                )
                current_length = controller.length
                strict_violation = current_length < self.min_chars or current_length > effective_max
            if strict_violation:
                LOGGER.warning(
                    "TRIM_LEN_RELAXED length=%d range=%d-%d soft_range=%d-%d",
                    current_length,
                    self.min_chars,
                    effective_max,
                    soft_min,
                    soft_max,
                )
                length_notes.setdefault("length_relaxed", True)
                length_notes["length_soft_min"] = soft_min
                length_notes["length_soft_max"] = soft_max
                length_notes["length_tolerance_below"] = tolerance_below
                length_notes["length_tolerance_above"] = tolerance_above
                length_notes.setdefault("length_relaxed_status", "accepted")
                if current_length < soft_min or current_length > soft_max:
                    controller_success = controller.success if controller.adjusted else False
                    length_notes["length_controller_success"] = controller_success
                    length_notes["length_controller_reason"] = controller.failure_reason
                    fail_safe_article = self._build_fail_safe_article()
                    result = TrimResult(
                        text=fail_safe_article,
                        removed_paragraphs=result.removed_paragraphs,
                        length_relaxed=getattr(result, "length_relaxed", False),
                        relaxed_limit=getattr(result, "relaxed_limit", None),
                    )
                    current_length = length_no_spaces(result.text)
                    length_notes["length_controller_fallback"] = True

        missing_locks = [
            term
            for term in self.required_keywords
            if LOCK_START_TEMPLATE.format(term=term) not in result.text
        ]
        if missing_locks:
            raise PipelineStepError(
                PipelineStep.TRIM,
                "После тримминга потеряны ключевые фразы: " + ", ".join(sorted(missing_locks)),
            )

        coverage_post = evaluate_keyword_coverage(
            result.text,
            self.required_keywords,
            preferred=self.preferred_keywords,
        )
        if coverage_post.missing_required:
            raise PipelineStepError(
                PipelineStep.TRIM,
                "После тримминга отсутствуют обязательные ключевые слова: "
                + ", ".join(sorted(coverage_post.missing_required)),
            )

        reinforcement_applied = False
        if (
            coverage_post.missing_preferred
            and coverage_post.overall_percent < 100.0
        ):
            reinforced_text = self._reinforce_keywords(
                result.text, coverage_post.missing_preferred
            )
            if reinforced_text != result.text:
                reinforcement_applied = True
                result = TrimResult(
                    text=reinforced_text,
                    removed_paragraphs=result.removed_paragraphs,
                    length_relaxed=getattr(result, "length_relaxed", False),
                    relaxed_limit=getattr(result, "relaxed_limit", None),
                )
                current_length = length_no_spaces(result.text)
                coverage_post = evaluate_keyword_coverage(
                    result.text,
                    self.required_keywords,
                    preferred=self.preferred_keywords,
                )
                if coverage_post.missing_required:
                    raise PipelineStepError(
                        PipelineStep.TRIM,
                        "После дозаряда отсутствуют обязательные ключевые слова: "
                        + ", ".join(sorted(coverage_post.missing_required)),
                    )
                if coverage_post.overall_percent < 100.0:
                    raise PipelineStepError(
                        PipelineStep.TRIM,
                        "Не удалось восстановить покрытие ключевых слов до 100%.",
                    )
                missing_locks = [
                    term
                    for term in self.required_keywords
                    if LOCK_START_TEMPLATE.format(term=term) not in result.text
                ]
                if missing_locks:
                    raise PipelineStepError(
                        PipelineStep.TRIM,
                        "После дозаряда потеряны защищённые фразы: "
                        + ", ".join(sorted(missing_locks)),
                    )
                if current_length > effective_max:
                    length_notes.setdefault("length_relaxed", True)
                    length_notes["length_relaxed_limit"] = effective_max
                    length_notes.setdefault("length_relaxed_status", "accepted")

        self.keywords_required_coverage_percent = coverage_post.required_percent
        self.keywords_coverage_percent = coverage_post.overall_percent
        if reinforcement_applied:
            length_notes["keyword_reinforced"] = True

        faq_block = ""
        if self.faq_target > 0:
            if FAQ_START in result.text and FAQ_END in result.text:
                faq_block = result.text.split(FAQ_START, 1)[1].split(FAQ_END, 1)[0]
            faq_pairs = re.findall(r"\*\*Вопрос\s+\d+\.\*\*", faq_block)
            if len(faq_pairs) != self.faq_target:
                raise PipelineStepError(
                    PipelineStep.TRIM,
                    f"FAQ должен содержать ровно {self.faq_target} вопросов после тримминга.",
                )
        LOGGER.info(
            "TRIM_OK chars_no_spaces=%d removed_paragraphs=%d",
            current_length,
            len(result.removed_paragraphs),
        )
        self._update_log(
            PipelineStep.TRIM,
            "ok",
            removed=len(result.removed_paragraphs),
            **self._metrics(result.text),
            **length_notes,
            KEYWORDS_COVERAGE_REQUIRED=self.keywords_required_coverage_percent,
            KEYWORDS_COVERAGE_OVERALL=self.keywords_coverage_percent,
        )
        self.checkpoints[PipelineStep.TRIM] = result.text
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> PipelineState:
        text = self._run_skeleton()
        keyword_result = self._run_keywords(text)
        faq_text = self._run_faq(keyword_result.text)
        trim_result = self._run_trim(faq_text)
        combined_text = trim_result.text
        if self.jsonld and self.jsonld_requested:
            combined_text = f"{combined_text.rstrip()}\n\n{self.jsonld}\n"
        try:
            validation = validate_article(
                combined_text,
                required_keywords=self.required_keywords,
                preferred_keywords=self.preferred_keywords,
                min_chars=self.min_chars,
                max_chars=self.max_chars,
                skeleton_payload=self.skeleton_payload,
                keyword_required_coverage_percent=self.keywords_required_coverage_percent,
                keyword_coverage_percent=self.keywords_coverage_percent,
                faq_expected=self.faq_target,
            )
        except ValidationError as exc:
            raise PipelineStepError(PipelineStep.TRIM, str(exc), status_code=400) from exc
        LOGGER.info(
            "VALIDATION_OK length=%s keywords=%.0f%%",
            validation.stats.get("length_no_spaces"),
            float(validation.stats.get("keywords_coverage_percent") or 0.0),
        )
        return PipelineState(
            text=combined_text,
            jsonld=self.jsonld,
            validation=validation,
            logs=self.logs,
            checkpoints=self.checkpoints,
            model_used=self._model_used or self.model,
            fallback_used=self._fallback_used,
            fallback_reason=self._fallback_reason,
            api_route=self._api_route,
            token_usage=self._token_usage,
            skeleton_payload=self.skeleton_payload,
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
            raise PipelineStepError(from_step, "Чекпоинты отсутствуют; требуется полный перезапуск.")

        base_step = order[fallback_index]
        base_text = self.checkpoints[base_step]
        self._sync_locked_terms(base_text)

        text = base_text
        for step in order[fallback_index + 1 :]:
            if step == PipelineStep.KEYWORDS:
                text = self._run_keywords(text).text
            elif step == PipelineStep.FAQ:
                text = self._run_faq(text)
            elif step == PipelineStep.TRIM:
                text = self._run_trim(text).text

        combined_text = text
        if self.jsonld and self.jsonld_requested:
            combined_text = f"{combined_text.rstrip()}\n\n{self.jsonld}\n"
        try:
            validation = validate_article(
                combined_text,
                required_keywords=self.required_keywords,
                preferred_keywords=self.preferred_keywords,
                min_chars=self.min_chars,
                max_chars=self.max_chars,
                skeleton_payload=self.skeleton_payload,
                keyword_required_coverage_percent=self.keywords_required_coverage_percent,
                keyword_coverage_percent=self.keywords_coverage_percent,
            )
        except ValidationError as exc:
            raise PipelineStepError(step, str(exc), status_code=400) from exc
        LOGGER.info(
            "VALIDATION_OK length=%s keywords=%.0f%%",
            validation.stats.get("length_no_spaces"),
            float(validation.stats.get("keywords_coverage_percent") or 0.0),
        )
        return PipelineState(
            text=combined_text,
            jsonld=self.jsonld,
            validation=validation,
            logs=self.logs,
            checkpoints=self.checkpoints,
            model_used=self._model_used or self.model,
            fallback_used=self._fallback_used,
            fallback_reason=self._fallback_reason,
            api_route=self._api_route,
            token_usage=self._token_usage,
            skeleton_payload=self.skeleton_payload,
        )
