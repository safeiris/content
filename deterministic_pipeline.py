"""LLM-driven content pipeline with explicit step-level guarantees."""

from __future__ import annotations

import json
import logging
import re
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config import (
    G5_MAX_OUTPUT_TOKENS_MAX,
    SKELETON_BATCH_SIZE_MAIN,
    SKELETON_FAQ_BATCH,
    TAIL_FILL_MAX_TOKENS,
)
from llm_client import FALLBACK_MODEL, GenerationResult, generate as llm_generate
from faq_builder import _normalize_entry
from keyword_injector import (
    KeywordInjectionResult,
    LOCK_END,
    LOCK_START_TEMPLATE,
    build_term_pattern,
    inject_keywords,
)
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
        adjusted = list(cleaned[:current_len])
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
        keywords: Iterable[str],
        min_chars: int,
        max_chars: int,
        messages: Sequence[Dict[str, object]],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
        backoff_schedule: Optional[List[float]] = None,
        provided_faq: Optional[List[Dict[str, str]]] = None,
        jsonld_requested: bool = True,
    ) -> None:
        if not model or not str(model).strip():
            raise PipelineStepError(PipelineStep.SKELETON, "Не указана модель для генерации.")

        self.topic = topic.strip() or "Тема"
        self.base_outline = list(base_outline) if base_outline else ["Введение", "Основная часть", "Вывод"]
        self.keywords = [str(term).strip() for term in keywords if str(term).strip()]
        self.normalized_keywords = [term for term in self.keywords if term]
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.messages = [dict(message) for message in messages]
        self.model = str(model).strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens) if max_tokens else 0
        self.timeout_s = int(timeout_s)
        self.backoff_schedule = list(backoff_schedule) if backoff_schedule else None
        self.provided_faq = provided_faq or []
        self.jsonld_requested = bool(jsonld_requested)

        self.logs: List[PipelineLogEntry] = []
        self.checkpoints: Dict[PipelineStep, str] = {}
        self.jsonld: Optional[str] = None
        self.locked_terms: List[str] = []
        self.jsonld_reserve: int = 0
        self.skeleton_payload: Optional[Dict[str, object]] = None
        self._skeleton_faq_entries: List[Dict[str, str]] = []
        self.keywords_coverage_percent: float = 0.0

        self._model_used: Optional[str] = None
        self._fallback_used: Optional[str] = None
        self._fallback_reason: Optional[str] = None
        self._api_route: Optional[str] = None
        self._token_usage: Optional[float] = None

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

    def _prompt_length(self, messages: Sequence[Dict[str, object]]) -> int:
        length = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                length += len(content)
        return length

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
                    temperature=self.temperature,
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
        min_chars = max(3200, int(self.min_chars) if self.min_chars > 0 else 3200)
        max_chars = max(min_chars + 400, int(self.max_chars) if self.max_chars > 0 else min_chars + 1200)
        avg_chars = max(min_chars, int((min_chars + max_chars) / 2))
        approx_tokens = max(1100, int(avg_chars / 3.2))
        main_count = max(1, len(outline.main_headings))
        faq_count = 5 if outline.has_faq else 0
        intro_tokens = max(160, int(approx_tokens * 0.12))
        conclusion_tokens = max(140, int(approx_tokens * 0.1))
        faq_pool = max(0, int(approx_tokens * 0.2)) if faq_count else 0
        per_faq_tokens = max(70, int(faq_pool / faq_count)) if faq_count else 0
        allocated_faq = per_faq_tokens * faq_count
        remaining_for_main = max(
            approx_tokens - intro_tokens - conclusion_tokens - allocated_faq,
            220 * main_count,
        )
        per_main_tokens = max(220, int(remaining_for_main / main_count)) if main_count else 0
        predicted = intro_tokens + conclusion_tokens + per_main_tokens * main_count + per_faq_tokens * faq_count
        start_max = int(predicted * 1.2)
        if cap is not None and cap > 0:
            start_max = min(start_max, cap)
        start_max = max(600, start_max)
        requires_chunking = bool(cap is not None and predicted > cap)
        LOGGER.info(
            "SKELETON_ESTIMATE predicted=%d start_max=%d cap=%s",
            predicted,
            start_max,
            cap if cap is not None else "-",
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

    def _build_skeleton_batches(self, outline: SkeletonOutline) -> List[SkeletonBatchPlan]:
        batches: List[SkeletonBatchPlan] = [SkeletonBatchPlan(kind=SkeletonBatchKind.INTRO, label="intro")]
        main_count = len(outline.main_headings)
        if main_count > 0:
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
        if outline.has_faq:
            total_faq = 5
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

    def _batch_schema(
        self,
        batch: SkeletonBatchPlan,
        *,
        outline: SkeletonOutline,
        item_count: int,
    ) -> Dict[str, object]:
        if batch.kind == SkeletonBatchKind.INTRO:
            schema = {
                "type": "object",
                "properties": {
                    "intro": {"type": "string"},
                    "main_headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": len(outline.main_headings),
                        "maxItems": len(outline.main_headings),
                    },
                    "conclusion_heading": {"type": "string"},
                },
                "required": ["intro", "main_headers", "conclusion_heading"],
                "additionalProperties": False,
            }
        elif batch.kind == SkeletonBatchKind.MAIN:
            min_items = 1 if item_count > 0 else 0
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
                        "maxItems": max(item_count, 1),
                    }
                },
                "required": ["sections"],
                "additionalProperties": False,
            }
        elif batch.kind == SkeletonBatchKind.FAQ:
            min_items = 1 if item_count > 0 else 0
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
                        "maxItems": max(item_count, 1),
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
        return {
            "type": "json_schema",
            "name": f"seo_article_{batch.kind.value}",
            "schema": schema,
            "strict": True,
        }

    def _extract_response_json(self, raw_text: str) -> Optional[object]:
        candidate = (raw_text or "").strip()
        if not candidate:
            return None
        if "<response_json>" in candidate and "</response_json>" in candidate:
            try:
                candidate = candidate.split("<response_json>", 1)[1].split("</response_json>", 1)[0]
            except Exception:  # pragma: no cover - defensive
                candidate = candidate
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
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
        if len(headers) < len(outline.main_headings):
            missing.append("main_headers")
        normalized["intro"] = intro_text
        normalized["main_headers"] = headers[: len(outline.main_headings)]
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
            lines.append(
                "Верни JSON {\"faq\": [{\"q\": str, \"a\": str}, ...]} в количестве, равном запросу."
            )
            lines.append(
                "Продолжай нумерацию вопросов, начиная с пункта №%d." % start_number
            )
            if tail_fill:
                lines.append("Добавь только недостающие вопросы и ответы.")
        else:
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
        previous_id = str(
            metadata.get("response_id")
            or metadata.get("previous_response_id")
            or metadata.get("id")
            or ""
        ).strip()
        if not previous_id:
            return
        tail_plan = SkeletonBatchPlan(
            kind=batch.kind,
            indices=list(pending),
            label=batch.label + "#tail",
            tail_fill=True,
        )
        messages, format_block = self._build_batch_messages(
            tail_plan,
            outline=outline,
            assembly=assembly,
            target_indices=list(pending),
            tail_fill=True,
        )
        budget = self._batch_token_budget(batch, estimate, len(pending))
        max_tokens = min(budget, TAIL_FILL_MAX_TOKENS)
        LOGGER.info(
            "TAIL_FILL missing_items=%s max_tokens=%d",
            ",".join(str(item + 1) for item in pending),
            max_tokens,
        )
        result = self._call_llm(
            step=PipelineStep.SKELETON,
            messages=messages,
            max_tokens=max_tokens,
            previous_response_id=previous_id,
            responses_format=format_block,
        )
        tail_metadata = result.metadata or {}
        payload = self._extract_response_json(result.text)
        if batch.kind == SkeletonBatchKind.MAIN:
            normalized, missing = self._normalize_main_batch(payload, list(pending), outline)
            for index, heading, body in normalized:
                assembly.apply_main(index, body, heading=heading)
            if missing:
                raise PipelineStepError(
                    PipelineStep.SKELETON,
                    "Не удалось достроить все разделы основной части.",
                )
        elif batch.kind == SkeletonBatchKind.FAQ:
            normalized, missing = self._normalize_faq_batch(payload, list(pending))
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
        for key, value in tail_metadata.items():
            if value:
                metadata[key] = value

    def _render_skeleton_markdown(self, payload: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
        if not isinstance(payload, dict):
            raise ValueError("Структура скелета не является объектом")

        intro = str(payload.get("intro") or "").strip()
        main = payload.get("main")
        conclusion = str(payload.get("conclusion") or "").strip()
        faq = payload.get("faq")
        if not intro or not conclusion or not isinstance(main, list) or len(main) == 0:
            raise ValueError("Скелет не содержит обязательных полей intro/main/conclusion")

        if not 3 <= len(main) <= 6:
            raise ValueError("Скелет основной части должен содержать 3–6 блоков")

        normalized_main: List[str] = []
        for idx, item in enumerate(main):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Элемент основной части №{idx + 1} пуст")
            normalized_main.append(item.strip())

        if not isinstance(faq, list) or len(faq) != 5:
            raise ValueError("Скелет FAQ должен содержать ровно 5 элементов")

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
        if len(sanitized) != 5:
            raise PipelineStepError(PipelineStep.FAQ, "FAQ должно содержать ровно 5 пар вопросов и ответов.")
        return sanitized

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
        self.locked_terms = pattern.findall(text)
        if self.normalized_keywords:
            article = strip_jsonld(text)
            found = 0
            for term in self.normalized_keywords:
                lock_token = LOCK_START_TEMPLATE.format(term=term)
                lock_pattern = re.compile(rf"{re.escape(lock_token)}.*?{re.escape(LOCK_END)}", re.DOTALL)
                if lock_pattern.search(text) and build_term_pattern(term).search(article):
                    found += 1
            self.keywords_coverage_percent = round(found / len(self.normalized_keywords) * 100, 2)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _run_skeleton(self) -> str:
        self._log(PipelineStep.SKELETON, "running")
        outline = self._prepare_outline()
        estimate = self._predict_skeleton_volume(outline)
        batches = self._build_skeleton_batches(outline)
        assembly = SkeletonAssembly(outline=outline)
        metadata_snapshot: Dict[str, object] = {}
        last_result: Optional[GenerationResult] = None

        for batch in batches:
            target_indices = list(batch.indices)
            messages, format_block = self._build_batch_messages(
                batch,
                outline=outline,
                assembly=assembly,
                target_indices=target_indices,
                tail_fill=batch.tail_fill,
            )
            max_tokens = self._batch_token_budget(batch, estimate, len(target_indices))
            try:
                result = self._call_llm(
                    step=PipelineStep.SKELETON,
                    messages=messages,
                    max_tokens=max_tokens,
                    responses_format=format_block,
                )
            except PipelineStepError:
                raise

            last_result = result
            metadata_snapshot = result.metadata or {}
            payload_obj = self._extract_response_json(result.text)

            if batch.kind == SkeletonBatchKind.INTRO:
                normalized, missing_fields = self._normalize_intro_batch(payload_obj, outline)
                intro_text = normalized.get("intro", "")
                headers = normalized.get("main_headers") or []
                if len(headers) < len(outline.main_headings):
                    headers = headers + outline.main_headings[len(headers) :]
                assembly.apply_intro(intro_text, headers, normalized.get("conclusion_heading"))
                if missing_fields:
                    self._tail_fill_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        estimate=estimate,
                        missing_items=[0],
                        metadata=metadata_snapshot,
                    )
            elif batch.kind == SkeletonBatchKind.MAIN:
                normalized_sections, missing_indices = self._normalize_main_batch(
                    payload_obj, target_indices, outline
                )
                for index, heading, body in normalized_sections:
                    assembly.apply_main(index, body, heading=heading)
                if missing_indices:
                    self._tail_fill_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        estimate=estimate,
                        missing_items=missing_indices,
                        metadata=metadata_snapshot,
                    )
            elif batch.kind == SkeletonBatchKind.FAQ:
                normalized_entries, missing_faq = self._normalize_faq_batch(payload_obj, target_indices)
                for _, question, answer in normalized_entries:
                    assembly.apply_faq(question, answer)
                if missing_faq:
                    self._tail_fill_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        estimate=estimate,
                        missing_items=missing_faq,
                        metadata=metadata_snapshot,
                    )
            else:
                conclusion_text, missing_flag = self._normalize_conclusion_batch(payload_obj)
                assembly.apply_conclusion(conclusion_text)
                if missing_flag:
                    self._tail_fill_batch(
                        batch,
                        outline=outline,
                        assembly=assembly,
                        estimate=estimate,
                        missing_items=[0],
                        metadata=metadata_snapshot,
                    )

            LOGGER.info("SKELETON_BATCH_ACCEPT kind=%s label=%s", batch.kind.value, batch.label)

        if not assembly.intro:
            raise PipelineStepError(PipelineStep.SKELETON, "Не удалось получить вводный блок скелета.")
        if not assembly.conclusion:
            raise PipelineStepError(PipelineStep.SKELETON, "Не удалось получить вывод скелета.")
        missing_main = assembly.missing_main_indices()
        if missing_main:
            raise PipelineStepError(
                PipelineStep.SKELETON,
                "Не удалось заполнить все разделы основной части.",
            )
        if outline.has_faq and assembly.missing_faq_count(5):
            raise PipelineStepError(
                PipelineStep.SKELETON,
                "Не удалось собрать полный FAQ на этапе скелета.",
            )

        payload = assembly.build_payload()
        if outline.has_faq and len(payload.get("faq", [])) > 5:
            payload["faq"] = payload["faq"][:5]

        normalized_payload = normalize_skeleton_payload(payload)
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
        result = inject_keywords(text, self.keywords)
        self.locked_terms = list(result.locked_terms)
        self.keywords_coverage_percent = result.coverage_percent
        total = result.total_terms
        found = result.found_terms
        missing = sorted(result.missing_terms)
        LOGGER.info(
            "KEYWORDS_COVERAGE=%.0f%% missing=%s",
            result.coverage_percent,
            ",".join(missing) if missing else "-",
        )
        if total and found < total:
            raise PipelineStepError(
                PipelineStep.KEYWORDS,
                "Не удалось обеспечить 100% покрытие ключей: " + ", ".join(missing),
            )
        LOGGER.info("KEYWORDS_OK coverage=%.2f%%", result.coverage_percent)
        self._update_log(
            PipelineStep.KEYWORDS,
            "ok",
            KEYWORDS_COVERAGE=result.coverage_report,
            KEYWORDS_COVERAGE_PERCENT=result.coverage_percent,
            KEYWORDS_MISSING=missing,
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

        if entries_source:
            sanitized = self._sanitize_entries(entries_source)
            if len(sanitized) != 5:
                raise PipelineStepError(
                    PipelineStep.FAQ,
                    "FAQ должно содержать ровно 5 пар вопросов и ответов.",
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

        raise PipelineStepError(
            PipelineStep.FAQ,
            "Не удалось сформировать блок FAQ: отсутствуют подготовленные данные.",
        )

    def _run_trim(self, text: str) -> TrimResult:
        self._log(PipelineStep.TRIM, "running")
        reserve = self.jsonld_reserve if self.jsonld else 0
        target_max = max(self.min_chars, self.max_chars - reserve)
        try:
            result = trim_text(
                text,
                min_chars=self.min_chars,
                max_chars=target_max,
                protected_blocks=self.locked_terms,
            )
        except TrimValidationError as exc:
            raise PipelineStepError(PipelineStep.TRIM, str(exc)) from exc
        current_length = length_no_spaces(result.text)
        if current_length < self.min_chars or current_length > self.max_chars:
            raise PipelineStepError(
                PipelineStep.TRIM,
                f"Объём после трима вне диапазона {self.min_chars}–{self.max_chars} (без пробелов).",
            )

        missing_locks = [
            term
            for term in self.normalized_keywords
            if LOCK_START_TEMPLATE.format(term=term) not in result.text
        ]
        if missing_locks:
            raise PipelineStepError(
                PipelineStep.TRIM,
                "После тримминга потеряны ключевые фразы: " + ", ".join(sorted(missing_locks)),
            )

        faq_block = ""
        if FAQ_START in result.text and FAQ_END in result.text:
            faq_block = result.text.split(FAQ_START, 1)[1].split(FAQ_END, 1)[0]
        faq_pairs = re.findall(r"\*\*Вопрос\s+\d+\.\*\*", faq_block)
        if len(faq_pairs) != 5:
            raise PipelineStepError(
                PipelineStep.TRIM,
                "FAQ должен содержать ровно 5 вопросов после тримминга.",
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
                keywords=self.keywords,
                min_chars=self.min_chars,
                max_chars=self.max_chars,
                skeleton_payload=self.skeleton_payload,
                keyword_coverage_percent=self.keywords_coverage_percent,
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
                keywords=self.keywords,
                min_chars=self.min_chars,
                max_chars=self.max_chars,
                skeleton_payload=self.skeleton_payload,
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
