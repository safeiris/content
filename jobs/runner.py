"""Background runner executing generation jobs."""
from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import JOB_SOFT_TIMEOUT_S
from observability.logger import get_logger, log_step
from orchestrate import generate_article_from_payload
from services.guardrails import parse_jsonld_or_repair

from .models import Job, JobStep
from .store import JobStore

LOGGER = get_logger("content_factory.jobs")


@dataclass
class RunnerTask:
    job_id: str
    payload: Dict[str, Any]


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

    def submit(self, payload: Dict[str, Any]) -> Job:
        job_id = uuid.uuid4().hex
        steps = [
            JobStep(name="draft"),
            JobStep(name="refine"),
            JobStep(name="jsonld"),
            JobStep(name="post-analysis"),
        ]
        job = Job(id=job_id, steps=steps)
        self._store.create(job)
        event = threading.Event()
        with self._events_lock:
            self._events[job_id] = event
        self._tasks.put(RunnerTask(job_id=job_id, payload=payload))
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
        return self._store.snapshot(job_id)

    def _worker(self) -> None:
        while not self._shutdown:
            task = self._tasks.get()
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

        job.mark_running()
        self._store.touch(job.id)

        draft_step, refine_step, jsonld_step, post_step = job.steps

        # Draft
        draft_step.mark_running()
        log_step(LOGGER, job_id=job.id, step=draft_step.name, status=draft_step.status.value)
        try:
            result = generate_article_from_payload(**task.payload)
            draft_step.mark_succeeded()
        except Exception as exc:  # noqa: BLE001
            draft_step.mark_failed(error=str(exc))
            job.mark_failed(str(exc))
            self._store.touch(job.id)
            log_step(
                LOGGER,
                job_id=job.id,
                step=draft_step.name,
                status=draft_step.status.value,
                error=str(exc),
            )
            return

        log_step(LOGGER, job_id=job.id, step=draft_step.name, status=draft_step.status.value)

        # Refine (placeholder for future incremental refinements)
        refine_step.mark_running()
        log_step(LOGGER, job_id=job.id, step=refine_step.name, status=refine_step.status.value)
        refine_step.mark_succeeded()
        log_step(LOGGER, job_id=job.id, step=refine_step.name, status=refine_step.status.value)

        metadata = result.get("metadata") if isinstance(result, dict) else None
        raw_jsonld = ""
        if isinstance(metadata, dict):
            raw_jsonld = str(metadata.get("jsonld") or "")

        # JSON-LD repair
        jsonld_step.mark_running()
        log_step(LOGGER, job_id=job.id, step=jsonld_step.name, status=jsonld_step.status.value)
        faq_entries, repair_attempts, degraded = parse_jsonld_or_repair(raw_jsonld)
        jsonld_payload: Dict[str, Any] = {"repair_attempts": repair_attempts, "faq_entries": faq_entries}
        if degraded:
            job.degradation_flags.append("jsonld_repaired")
            jsonld_payload["degraded"] = True
        jsonld_step.mark_succeeded(**jsonld_payload)
        log_step(LOGGER, job_id=job.id, step=jsonld_step.name, status=jsonld_step.status.value)

        # Post analysis step
        post_step.mark_running()
        log_step(LOGGER, job_id=job.id, step=post_step.name, status=post_step.status.value)
        post_payload = {"jsonld_repair_attempts": repair_attempts}
        if faq_entries:
            post_payload["faq_preview"] = faq_entries[:2]
        post_step.mark_succeeded(**post_payload)
        log_step(LOGGER, job_id=job.id, step=post_step.name, status=post_step.status.value)

        job.mark_succeeded(result)
        self._store.touch(job.id)

    def soft_timeout(self) -> int:
        return self._soft_timeout_s
