"""Background execution engine for generation jobs with soft degradation."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

        for step in job.steps:
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

            result = self._execute_step(step.name, task.payload, ctx)
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
        self._store.touch(job.id)
        JOB_COUNTER.inc()

    def _execute_step(
        self,
        step_name: str,
        payload: Dict[str, Any],
        ctx: PipelineContext,
    ) -> StepResult:
        if step_name == "draft":
            return self._run_draft_step(payload, ctx)
        if step_name == "refine":
            return self._run_refine_step(ctx)
        if step_name == "jsonld":
            return self._run_jsonld_step(ctx)
        if step_name == "post_analysis":
            return self._run_post_analysis_step(ctx)
        return StepResult(JobStepStatus.SUCCEEDED, payload={"skipped": True})

    def _run_draft_step(self, payload: Dict[str, Any], ctx: PipelineContext) -> StepResult:
        attempt = 0
        last_error: Optional[str] = None
        while attempt <= JOB_MAX_RETRIES_PER_STEP:
            attempt += 1
            try:
                result = generate_article_from_payload(**payload)
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
                return StepResult(JobStepStatus.SUCCEEDED, payload={"attempts": attempt})
            last_error = "empty_response"
            ctx.errors.append(last_error)

        ctx.ensure_markdown(_build_fallback_text(payload, error=last_error))
        flags = ["draft_failed"]
        return StepResult(
            JobStepStatus.DEGRADED,
            payload={"attempts": attempt, "error": last_error},
            degradation_flags=flags,
            error=last_error,
        )

    def _run_refine_step(self, ctx: PipelineContext) -> StepResult:
        if not ctx.markdown.strip():
            ctx.ensure_markdown("Черновик пока пустой.")
            return StepResult(
                JobStepStatus.DEGRADED,
                payload={"action": "fallback_text"},
                degradation_flags=["refine_skipped"],
                error="empty_markdown",
            )
        refined = ctx.markdown.strip()
        if refined != ctx.markdown:
            ctx.markdown = refined
        return StepResult(JobStepStatus.SUCCEEDED, payload={"chars": len(refined)})

    def _run_jsonld_step(self, ctx: PipelineContext) -> StepResult:
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
        return StepResult(
            status,
            payload={
                "attempts": guardrail_result.attempts,
                "faq_preview": guardrail_result.faq_entries[:2] if guardrail_result.faq_entries else None,
            },
            degradation_flags=guardrail_result.degradation_flags,
            error=guardrail_result.error,
        )

    def _run_post_analysis_step(self, ctx: PipelineContext) -> StepResult:
        if not ctx.markdown.strip():
            return StepResult(
                JobStepStatus.DEGRADED,
                payload={"reason": "empty_markdown"},
                degradation_flags=["post_analysis_skipped"],
                error="no_markdown",
            )
        length = len(ctx.markdown.replace(" ", ""))
        payload = {"chars_no_spaces": length, "faq_count": len(ctx.faq_entries)}
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
