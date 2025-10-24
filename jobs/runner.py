"""Background execution engine for generation jobs with soft degradation."""
from __future__ import annotations

import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config import JOB_MAX_RETRIES_PER_STEP, JOB_SOFT_TIMEOUT_S
from observability.logger import get_logger, log_step
from observability.metrics import get_registry
from orchestrate import generate_article_from_payload
from services.guardrails import GuardrailResult, parse_and_repair_jsonld

from .models import Job, JobStep, JobStepStatus
from .store import JobStore

LOGGER = get_logger("content_factory.jobs.runner")
REGISTRY = get_registry()
QUEUE_GAUGE = REGISTRY.gauge("jobs.queue_length")
JOB_COUNTER = REGISTRY.counter("jobs.processed_total")

PROGRESS_STAGE_WEIGHTS = {
    "draft": (0.0, 0.82),
    "trim": (0.82, 0.1),
    "validate": (0.92, 0.06),
    "done": (1.0, 0.0),
}

PROGRESS_STAGE_MESSAGES = {
    "draft": "Генерируем черновик",
    "trim": "Нормализуем объём",
    "validate": "Проверяем результат",
    "done": "Готово",
}


@dataclass
class RunnerTask:
    job_id: str
    payload: Dict[str, Any]
    trace_id: Optional[str] = None


@dataclass
class PipelineContext:
    markdown: str = ""
    meta_json: Dict[str, Any] = field(default_factory=dict)
    faq_entries: List[Dict[str, str]] = field(default_factory=list)
    degradation_flags: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None

    def ensure_markdown(self, fallback: str) -> None:
        if not self.markdown.strip():
            self.markdown = fallback


@dataclass
class StepResult:
    status: JobStepStatus
    payload: Dict[str, Any] = field(default_factory=dict)
    degradation_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    continue_pipeline: bool = True


class JobRunner:
    """Serial job runner executing pipeline tasks in a background thread."""

    def __init__(self, store: JobStore, *, soft_timeout_s: int = JOB_SOFT_TIMEOUT_S) -> None:
        self._store = store
        self._soft_timeout_s = soft_timeout_s
        self._tasks: "queue.Queue[RunnerTask]" = queue.Queue()
        self._events: Dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="job-runner", daemon=True)
        self._started = False
        self._shutdown = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        self._shutdown = True
        self._tasks.put(RunnerTask(job_id="__shutdown__", payload={}))
        if self._started:
            self._thread.join(timeout=1.0)

    def submit(self, payload: Dict[str, Any], *, trace_id: Optional[str] = None) -> Job:
        job_id = uuid.uuid4().hex
        steps = [
            JobStep(name="draft"),
            JobStep(name="refine"),
            JobStep(name="jsonld"),
            JobStep(name="post_analysis"),
        ]
        job = Job(id=job_id, steps=steps, trace_id=trace_id)
        self._store.create(job)
        event = threading.Event()
        with self._events_lock:
            self._events[job_id] = event
        self._tasks.put(RunnerTask(job_id=job_id, payload=payload, trace_id=trace_id))
        QUEUE_GAUGE.set(float(self._tasks.qsize()))
        self.start()
        LOGGER.info("job_enqueued", extra={"job_id": job_id})
        return job

    def wait(self, job_id: str, timeout: Optional[float] = None) -> bool:
        with self._events_lock:
            event = self._events.get(job_id)
        if not event:
            snapshot = self._store.snapshot(job_id)
            if snapshot and snapshot.get("status") in {"succeeded", "failed"}:
                return True
            return False
        return event.wait(timeout)

    def get_job(self, job_id: str) -> Optional[dict]:
        snapshot = self._store.snapshot(job_id)
        if not snapshot:
            return None
        return snapshot

    def soft_timeout(self) -> int:
        return self._soft_timeout_s

    def _worker(self) -> None:
        while not self._shutdown:
            task = self._tasks.get()
            QUEUE_GAUGE.set(float(self._tasks.qsize()))
            if task.job_id == "__shutdown__":
                break
            try:
                self._run_job(task)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("job_failed", extra={"job_id": task.job_id, "error": str(exc)})
            finally:
                with self._events_lock:
                    event = self._events.pop(task.job_id, None)
                if event:
                    event.set()

    def _run_job(self, task: RunnerTask) -> None:
        job = self._store.get(task.job_id)
        if not job:
            LOGGER.warning("job_missing", extra={"job_id": task.job_id})
            return

        job.trace_id = job.trace_id or task.trace_id
        job.mark_running()
        self._store.touch(job.id)

        ctx = PipelineContext(trace_id=job.trace_id)
        start_time = time.monotonic()
        deadline = start_time + self._soft_timeout_s
        refine_extension = max(5.0, self._soft_timeout_s * 0.35)
        refine_extension_applied = False

        for step in job.steps:
            if step.name == "refine" and not refine_extension_applied:
                deadline += refine_extension
                refine_extension_applied = True
                LOGGER.info(
                    "job_soft_timeout_extend",
                    extra={"step": step.name, "extra_seconds": round(refine_extension, 2)},
                )
            if time.monotonic() >= deadline:
                ctx.degradation_flags.append("soft_timeout")
                step.mark_degraded("soft_timeout")
                log_step(
                    LOGGER,
                    job_id=job.id,
                    step=step.name,
                    status=step.status.value,
                    reason="soft_timeout",
                )
                break

            step.mark_running()
            self._store.touch(job.id)
            result = self._execute_step(step.name, task.payload, ctx, job)
            if result.status == JobStepStatus.SUCCEEDED:
                step.mark_succeeded(**result.payload)
            elif result.status == JobStepStatus.DEGRADED:
                step.mark_degraded(result.error, **result.payload)
            else:
                step.mark_failed(result.error, **result.payload)
            log_step(
                LOGGER,
                job_id=job.id,
                step=step.name,
                status=step.status.value,
                error=result.error,
                payload=result.payload or None,
            )
            ctx.degradation_flags.extend(result.degradation_flags)
            self._store.touch(job.id)
            if not result.continue_pipeline:
                break

        ctx.ensure_markdown(_build_fallback_text(task.payload))
        if ctx.degradation_flags:
            ctx.degradation_flags = list(dict.fromkeys(ctx.degradation_flags))
        result_payload = {
            "markdown": ctx.markdown,
            "meta_json": ctx.meta_json,
            "faq_entries": ctx.faq_entries,
            "errors": ctx.errors or None,
        }
        job.mark_succeeded(result_payload, degradation_flags=ctx.degradation_flags)
        self._record_progress(job, "done", 1.0, message=PROGRESS_STAGE_MESSAGES.get("done"))
        self._store.touch(job.id)
        JOB_COUNTER.inc()

    def _execute_step(
        self,
        step_name: str,
        payload: Dict[str, Any],
        ctx: PipelineContext,
        job: Optional[Job],
    ) -> StepResult:
        if step_name == "draft":
            return self._run_draft_step(payload, ctx, job)
        if step_name == "refine":
            return self._run_refine_step(ctx, job)
        if step_name == "jsonld":
            return self._run_jsonld_step(ctx, job)
        if step_name == "post_analysis":
            return self._run_post_analysis_step(ctx, job)
        return StepResult(JobStepStatus.SUCCEEDED, payload={"skipped": True})

    def _record_progress(
        self,
        job: Optional[Job],
        stage: str,
        ratio: float,
        *,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if job is None:
            return
        stage_key = str(stage or "").strip().lower() or "draft"
        base, span = PROGRESS_STAGE_WEIGHTS.get(stage_key, (0.0, 0.0))
        try:
            normalized = float(ratio)
        except (TypeError, ValueError):
            normalized = 0.0
        normalized = max(0.0, min(1.0, normalized))
        progress_value = base + normalized * span
        if stage_key == "done":
            progress_value = 1.0
        if job.progress_value is not None:
            progress_value = max(float(job.progress_value), progress_value)
        effective_message = message or PROGRESS_STAGE_MESSAGES.get(stage_key) or "Обработка задания"
        job.update_progress(
            stage=stage_key,
            progress=min(1.0, progress_value),
            message=effective_message,
            payload=payload,
        )
        self._store.touch(job.id)

    def _run_draft_step(
        self,
        payload: Dict[str, Any],
        ctx: PipelineContext,
        job: Optional[Job],
    ) -> StepResult:
        self._record_progress(job, "draft", 0.0)

        def _progress_event(
            stage: str,
            *,
            progress: float = 0.0,
            message: Optional[str] = None,
            payload: Optional[Dict[str, Any]] = None,
        ) -> None:
            self._record_progress(job, stage, progress, message=message, payload=payload)

        attempt = 0
        last_error: Optional[str] = None
        while attempt <= JOB_MAX_RETRIES_PER_STEP:
            attempt += 1
            try:
                result = generate_article_from_payload(
                    **payload,
                    progress_callback=_progress_event,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                ctx.errors.append(last_error)
                continue

            markdown = str(result.get("text") or result.get("markdown") or "").strip()
            metadata = result.get("metadata")
            if isinstance(metadata, dict):
                ctx.meta_json = metadata
            if markdown:
                ctx.markdown = markdown
                self._record_progress(job, "draft", 1.0)
                return StepResult(JobStepStatus.SUCCEEDED, payload={"attempts": attempt})
            last_error = "empty_response"
            ctx.errors.append(last_error)

        ctx.ensure_markdown(_build_fallback_text(payload, error=last_error))
        flags = ["draft_failed"]
        self._record_progress(job, "draft", 1.0, message="Черновик по запасному сценарию")
        return StepResult(
            JobStepStatus.DEGRADED,
            payload={"attempts": attempt, "error": last_error},
            degradation_flags=flags,
            error=last_error,
        )

    def _run_refine_step(self, ctx: PipelineContext, job: Optional[Job]) -> StepResult:
        if not ctx.markdown.strip():
            ctx.ensure_markdown("Черновик пока пустой.")
            return StepResult(
                JobStepStatus.DEGRADED,
                payload={"action": "fallback_text"},
                degradation_flags=["refine_skipped"],
                error="empty_markdown",
            )
        self._record_progress(job, "trim", 0.0)
        refined = ctx.markdown.strip()
        passes = []

        # Pass 1: stylistic cleanup
        cleaned = "\n".join(line.rstrip() for line in refined.splitlines())
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        passes.append("style")

        # Pass 2: light SEO polish (placeholder for future LLM call)
        polished = re.sub(r"\s{2,}", " ", cleaned)
        passes.append("seo")

        final_text = polished.strip()
        ctx.markdown = final_text
        self._record_progress(job, "trim", 1.0, payload={"chars": len(final_text)})
        return StepResult(
            JobStepStatus.SUCCEEDED,
            payload={"chars": len(final_text), "passes": passes},
        )

    def _run_jsonld_step(self, ctx: PipelineContext, job: Optional[Job]) -> StepResult:
        raw_jsonld = None
        if isinstance(ctx.meta_json, dict):
            raw_jsonld = ctx.meta_json.get("jsonld")
        guardrail_result: GuardrailResult = parse_and_repair_jsonld(raw_jsonld, trace_id=ctx.trace_id)
        ctx.degradation_flags.extend(guardrail_result.degradation_flags)
        if guardrail_result.faq_entries:
            ctx.faq_entries = guardrail_result.faq_entries
        if guardrail_result.repaired_json is not None:
            ctx.meta_json["jsonld"] = guardrail_result.repaired_json
        status = JobStepStatus.SUCCEEDED if guardrail_result.ok else JobStepStatus.DEGRADED
        if guardrail_result.ok:
            self._record_progress(job, "validate", 0.6, payload={"faq_preview": guardrail_result.faq_entries})
        return StepResult(
            status,
            payload={
                "attempts": guardrail_result.attempts,
                "faq_preview": guardrail_result.faq_entries[:2] if guardrail_result.faq_entries else None,
            },
            degradation_flags=guardrail_result.degradation_flags,
            error=guardrail_result.error,
        )

    def _run_post_analysis_step(self, ctx: PipelineContext, job: Optional[Job]) -> StepResult:
        if not ctx.markdown.strip():
            return StepResult(
                JobStepStatus.DEGRADED,
                payload={"reason": "empty_markdown"},
                degradation_flags=["post_analysis_skipped"],
                error="no_markdown",
            )
        length = len(ctx.markdown.replace(" ", ""))
        payload = {"chars_no_spaces": length, "faq_count": len(ctx.faq_entries)}
        self._record_progress(job, "validate", 1.0, payload=payload)
        return StepResult(JobStepStatus.SUCCEEDED, payload=payload)


def _build_fallback_text(payload: Dict[str, Any], *, error: Optional[str] = None) -> str:
    theme = str(payload.get("theme") or payload.get("title") or "Материал").strip()
    if not theme:
        theme = "Материал"
    message = [f"Предварительный черновик для темы: {theme}."]
    if error:
        message.append(f"Ошибка: {error}")
    message.append("Контент недоступен, используйте черновик для доработки.")
    return "\n\n".join(message)


__all__ = ["JobRunner", "RunnerTask", "PipelineContext", "StepResult"]
