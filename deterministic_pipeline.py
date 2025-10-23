"""LLM-driven content pipeline with explicit step-level guarantees."""

from __future__ import annotations

import json
import json
import logging
import re
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from llm_client import GenerationResult, generate as llm_generate
from keyword_injector import (
    KeywordInjectionResult,
    LOCK_END,
    LOCK_START_TEMPLATE,
    build_term_pattern,
    inject_keywords,
)
from length_trimmer import TrimResult, TrimValidationError, trim_text
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
    ) -> GenerationResult:
        prompt_len = self._prompt_length(messages)
        limit = max_tokens if max_tokens and max_tokens > 0 else self.max_tokens
        if not limit or limit <= 0:
            limit = 700
        attempt = 0
        while attempt < 3:
            attempt += 1
            LOGGER.info(
                "LOG:LLM_REQUEST step=%s model=%s prompt_len=%d attempt=%d max_tokens=%d",
                step.value,
                self.model,
                prompt_len,
                attempt,
                limit,
            )
            try:
                result = llm_generate(
                    list(messages),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=limit,
                    timeout_s=self.timeout_s,
                    backoff_schedule=self.backoff_schedule,
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

    def _resolve_skeleton_tokens(self) -> int:
        baseline = max(self.max_tokens, self.max_chars + 400)
        if baseline <= 0:
            baseline = self.max_chars + 400
        return min(1500, max(600, baseline))

    def _skeleton_contract(self) -> Dict[str, object]:
        outline = [segment.strip() for segment in self.base_outline if segment.strip()]
        intro = outline[0] if outline else "Введение"
        outro = outline[-1] if len(outline) > 1 else "Вывод"
        core_sections = [
            item
            for item in outline[1:-1]
            if item.lower() not in {"faq", "f.a.q.", "вопросы и ответы"}
        ]
        if not core_sections:
            core_sections = ["Основная часть"]
        contract = {
            "intro": f"2-3 плотных абзаца для раздела '{intro}'",
            "main": [
                f"2-3 абзаца раскрывают тему '{heading}' на практических примерах"
                for heading in core_sections
            ],
            "outro": f"1-2 абзаца с выводами и призывом к действию для блока '{outro}'",
        }
        return contract

    def _build_skeleton_messages(self) -> List[Dict[str, object]]:
        outline = [segment.strip() for segment in self.base_outline if segment.strip()]
        contract_payload = self._skeleton_contract()
        contract = json.dumps(contract_payload, ensure_ascii=False, indent=2)
        main_expected = max(1, len(contract_payload.get("main") or []))
        user_payload = textwrap.dedent(
            f"""
            Сформируй структуру статьи в строгом JSON-формате.
            Требования:
            1. Соблюдай порядок разделов: {', '.join(outline) if outline else 'Введение, Основная часть, Вывод'}.
            2. Верни JSON вида {{"intro": str, "main": [str, ...], "outro": str}} без дополнительных ключей.
            3. main должен содержать {main_expected} элемента — по одному на каждый раздел основной части.
            4. Каждый элемент intro/main/outro содержит 2-3 осмысленных абзаца по 3-4 предложения, без Markdown и приветствий.
            5. Не добавляй FAQ, разметку, комментарии и служебные подписи.
            Образец структуры:
            {contract}
            """
        ).strip()
        messages = list(self.messages)
        messages.append({"role": "user", "content": user_payload})
        return messages

    def _render_skeleton_markdown(self, payload: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
        if not isinstance(payload, dict):
            raise ValueError("Структура скелета не является объектом")

        intro = str(payload.get("intro") or "").strip()
        main = payload.get("main")
        outro = str(payload.get("outro") or "").strip()
        if not intro or not outro or not isinstance(main, list) or not main:
            raise ValueError("Скелет не содержит обязательных полей")

        for idx, item in enumerate(main):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Элемент основной части №{idx + 1} пуст")

        outline = [segment.strip() for segment in self.base_outline if segment.strip()]
        outline = [
            entry
            for entry in outline
            if entry.lower() not in {"faq", "f.a.q.", "вопросы и ответы"}
        ]
        if len(outline) < 3:
            outline = ["Введение", "Основная часть", "Вывод"]

        intro_heading = outline[0]
        outro_heading = outline[-1]
        main_headings = outline[1:-1]
        if not main_headings:
            main_headings = ["Основная часть"]
        if len(main_headings) != len(main):
            raise ValueError(
                "Количество блоков в основной части не совпадает с ожиданиями по структуре"
            )

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
        for heading, body in zip(main_headings, main):
            _append_section(heading, body)
        _append_section(outro_heading, outro)

        lines.append("## FAQ")
        lines.append(FAQ_START)
        lines.append(FAQ_END)
        markdown = "\n".join(lines).strip()
        outline_summary = [intro_heading, *main_headings, outro_heading]
        return markdown, {"outline": outline_summary}

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
        for entry in entries:
            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()
            if not question or not answer:
                continue
            lowered = (question + " " + answer).lower()
            if "дополнительно рассматривается" in lowered:
                raise PipelineStepError(PipelineStep.FAQ, "FAQ содержит шаблонную фразу 'Дополнительно рассматривается'.")
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
        messages = self._build_skeleton_messages()
        skeleton_tokens = self._resolve_skeleton_tokens()
        attempt = 0
        last_error: Optional[Exception] = None
        payload: Optional[Dict[str, object]] = None
        markdown: Optional[str] = None
        metadata_snapshot: Dict[str, object] = {}
        while attempt < 3 and markdown is None:
            attempt += 1
            try:
                result = self._call_llm(
                    step=PipelineStep.SKELETON,
                    messages=messages,
                    max_tokens=skeleton_tokens,
                )
            except PipelineStepError:
                raise
            metadata_snapshot = result.metadata or {}
            status = str(metadata_snapshot.get("status") or "ok").lower()
            if status == "incomplete" or metadata_snapshot.get("incomplete_reason"):
                LOGGER.warning(
                    "SKELETON_RETRY_incomplete attempt=%d status=%s reason=%s",
                    attempt,
                    status,
                    metadata_snapshot.get("incomplete_reason") or "",
                )
                skeleton_tokens = max(600, int(skeleton_tokens * 0.9))
                continue
            raw_text = result.text.strip()
            if not raw_text:
                last_error = PipelineStepError(PipelineStep.SKELETON, "Модель вернула пустой ответ.")
                LOGGER.warning("SKELETON_RETRY_json_error attempt=%d error=empty", attempt)
                skeleton_tokens = max(600, int(skeleton_tokens * 0.9))
                continue
            try:
                payload = json.loads(raw_text)
                LOGGER.info("SKELETON_JSON_OK attempt=%d", attempt)
            except json.JSONDecodeError as exc:
                LOGGER.warning("SKELETON_JSON_INVALID attempt=%d error=%s", attempt, exc)
                LOGGER.warning("SKELETON_RETRY_json_error attempt=%d", attempt)
                skeleton_tokens = max(600, int(skeleton_tokens * 0.9))
                last_error = PipelineStepError(PipelineStep.SKELETON, "Ответ модели не является корректным JSON.")
                continue
            try:
                markdown, summary = self._render_skeleton_markdown(payload)
                snapshot = dict(payload)
                snapshot["outline"] = summary.get("outline", [])
                self.skeleton_payload = snapshot
                LOGGER.info("SKELETON_RENDERED_WITH_MARKERS outline=%s", ",".join(summary.get("outline", [])))
            except Exception as exc:  # noqa: BLE001
                last_error = PipelineStepError(PipelineStep.SKELETON, str(exc))
                LOGGER.warning("SKELETON_RETRY_json_error attempt=%d error=%s", attempt, exc)
                payload = None
                skeleton_tokens = max(600, int(skeleton_tokens * 0.9))
                markdown = None

        if markdown is None:
            if last_error:
                raise last_error
            raise PipelineStepError(
                PipelineStep.SKELETON,
                "Не удалось получить корректный скелет статьи после нескольких попыток.",
            )

        if FAQ_START not in markdown or FAQ_END not in markdown:
            raise PipelineStepError(PipelineStep.SKELETON, "Не удалось вставить маркеры FAQ на этапе скелета.")

        self._check_template_text(markdown, PipelineStep.SKELETON)
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
        messages = self._build_faq_messages(text)
        faq_tokens = 700
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < 3:
            attempt += 1
            try:
                result = self._call_llm(step=PipelineStep.FAQ, messages=messages, max_tokens=faq_tokens)
            except PipelineStepError:
                raise
            metadata = result.metadata or {}
            status = str(metadata.get("status") or "").lower()
            if status == "incomplete" or metadata.get("incomplete_reason"):
                LOGGER.warning(
                    "FAQ_RETRY_incomplete attempt=%d status=%s reason=%s",
                    attempt,
                    status or "incomplete",
                    metadata.get("incomplete_reason") or "",
                )
                faq_tokens = max(300, int(faq_tokens * 0.9))
                last_error = PipelineStepError(PipelineStep.FAQ, "Модель не завершила формирование FAQ.")
                continue
            try:
                entries = self._parse_faq_entries(result.text)
            except PipelineStepError as exc:
                last_error = exc
                LOGGER.warning("FAQ_RETRY_parse_error attempt=%d error=%s", attempt, exc)
                faq_tokens = max(300, int(faq_tokens * 0.9))
                continue
            faq_block = self._render_faq_markdown(entries)
            merged_text = self._merge_faq(text, faq_block)
            self.jsonld = self._build_jsonld(entries)
            self.jsonld_reserve = len(self.jsonld.replace(" ", "")) if self.jsonld else 0
            LOGGER.info("FAQ_OK entries=%s", ",".join(entry["question"] for entry in entries))
            self._update_log(
                PipelineStep.FAQ,
                "ok",
                entries=[entry["question"] for entry in entries],
                **self._metrics(merged_text),
            )
            self.checkpoints[PipelineStep.FAQ] = merged_text
            return merged_text

        if last_error:
            raise last_error
        raise PipelineStepError(PipelineStep.FAQ, "Не удалось сформировать блок FAQ после нескольких попыток.")

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
