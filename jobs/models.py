"""Data models describing asynchronous generation jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    """Lifecycle states for a background job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobStepStatus(str, Enum):
    """Lifecycle states for an individual pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"
    SKIPPED = "skipped"


@dataclass
class JobStep:
    """Progress information for a single pipeline step."""

    name: str
    status: JobStepStatus = JobStepStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def mark_running(self) -> None:
        self.status = JobStepStatus.RUNNING
        self.started_at = self.started_at or utcnow()

    def mark_succeeded(self, **payload: Any) -> None:
        self.status = JobStepStatus.SUCCEEDED
        if payload:
            self.payload.update(payload)
        self.finished_at = utcnow()
        self.error = None

    def mark_degraded(self, reason: str | None = None, **payload: Any) -> None:
        self.status = JobStepStatus.DEGRADED
        if payload:
            self.payload.update(payload)
        self.error = reason
        self.finished_at = utcnow()

    def mark_failed(self, reason: str | None = None, **payload: Any) -> None:
        self.status = JobStepStatus.FAILED
        if payload:
            self.payload.update(payload)
        self.error = reason
        self.finished_at = utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.strftime(ISO_FORMAT) if self.started_at else None,
            "finished_at": self.finished_at.strftime(ISO_FORMAT) if self.finished_at else None,
            "payload": self.payload or None,
            "error": self.error,
        }


@dataclass
class Job:
    """Representation of a long-running generation request."""

    id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    steps: List[JobStep] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    degradation_flags: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    last_event_at: datetime = field(default_factory=utcnow)
    progress_stage: Optional[str] = None
    progress_value: Optional[float] = None
    progress_message: Optional[str] = None
    progress_payload: Dict[str, Any] = field(default_factory=dict)

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = self.started_at or utcnow()
        self.last_event_at = utcnow()

    def mark_succeeded(self, result: Dict[str, Any], *, degradation_flags: Optional[List[str]] = None) -> None:
        self.status = JobStatus.SUCCEEDED
        self.result = result
        if degradation_flags:
            self.degradation_flags.extend(flag for flag in degradation_flags if flag)
        self.finished_at = utcnow()
        self.last_event_at = utcnow()

    def mark_failed(self, error: str | Dict[str, Any], *, degradation_flags: Optional[List[str]] = None) -> None:
        self.status = JobStatus.FAILED
        self.error = {"message": error} if isinstance(error, str) else error
        self.finished_at = utcnow()
        if degradation_flags:
            self.degradation_flags.extend(flag for flag in degradation_flags if flag)
        self.last_event_at = utcnow()

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at.strftime(ISO_FORMAT),
            "started_at": self.started_at.strftime(ISO_FORMAT) if self.started_at else None,
            "finished_at": self.finished_at.strftime(ISO_FORMAT) if self.finished_at else None,
            "steps": [step.to_dict() for step in self.steps],
            "result": self.result,
            "error": self.error,
            "degradation_flags": list(self.degradation_flags) or None,
            "trace_id": self.trace_id,
            "progress_stage": self.progress_stage,
            "progress_message": self.progress_message,
            "progress_payload": self.progress_payload or None,
        }
        payload.update(summarize_job(self))
        return payload

    def update_progress(
        self,
        *,
        stage: str,
        progress: float,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized_stage = stage.strip().lower() if isinstance(stage, str) else ""
        self.progress_stage = normalized_stage or self.progress_stage or "draft"
        if progress is not None:
            try:
                value = float(progress)
            except (TypeError, ValueError):
                value = self.progress_value or 0.0
            self.progress_value = max(0.0, min(1.0, value))
        if message is not None:
            self.progress_message = message
        if payload is not None:
            try:
                self.progress_payload = dict(payload)
            except Exception:  # pragma: no cover - defensive
                self.progress_payload = {}
        self.last_event_at = utcnow()


def summarize_job(job: "Job") -> Dict[str, Any]:
    status_map = {
        JobStatus.PENDING: "queued",
        JobStatus.RUNNING: "running",
        JobStatus.SUCCEEDED: "succeeded",
        JobStatus.FAILED: "failed",
    }
    step_alias = {
        "jsonld": "finalize",
        "post_analysis": "finalize",
    }
    step_labels = {
        "draft": "Черновик",
        "refine": "Полировка",
        "finalize": "Финализация",
        "jsonld": "JSON-LD",
        "post_analysis": "Пост-анализ",
        "done": "Готово",
    }

    status = status_map.get(job.status, job.status.value)

    total_steps = len(job.steps)
    completed = sum(
        1
        for step in job.steps
        if step.status in {JobStepStatus.SUCCEEDED, JobStepStatus.DEGRADED, JobStepStatus.SKIPPED}
    )
    running_step = next((step for step in job.steps if step.status == JobStepStatus.RUNNING), None)
    pending_step = next((step for step in job.steps if step.status == JobStepStatus.PENDING), None)

    if status in {"succeeded", "failed"}:
        step_name = "done"
        progress = 1.0
    else:
        if running_step:
            step_name = step_alias.get(running_step.name, running_step.name)
        elif pending_step:
            step_name = step_alias.get(pending_step.name, pending_step.name)
        elif job.steps:
            step_name = step_alias.get(job.steps[-1].name, job.steps[-1].name)
        else:
            step_name = "draft"
        if total_steps:
            progress = completed / total_steps
            if running_step:
                progress += 0.5 / total_steps
            progress = min(1.0, max(0.0, progress))
        else:
            progress = 0.0

    if job.progress_value is not None:
        progress = max(0.0, min(1.0, float(job.progress_value)))
    if job.progress_stage:
        step_name = job.progress_stage

    if status == "queued":
        message = "Задание в очереди"
    elif status == "running":
        message = job.progress_message or f"Шаг: {step_labels.get(step_name, step_name)}"
    elif status == "succeeded":
        message = job.progress_message or "Готово"
    else:
        error_message = ""
        if isinstance(job.error, dict):
            error_message = str(job.error.get("message") or "").strip()
        message = error_message or "Завершено с ошибкой"

    if job.progress_message and status in {"running", "succeeded"}:
        message = job.progress_message

    timestamps = [job.created_at, job.started_at, job.finished_at, job.last_event_at]
    for step in job.steps:
        timestamps.extend([step.started_at, step.finished_at])
    last_event_candidates = [ts for ts in timestamps if ts]
    last_event = max(last_event_candidates) if last_event_candidates else utcnow()

    return {
        "status": status,
        "step": step_name,
        "progress": round(progress, 4),
        "last_event_at": last_event.strftime(ISO_FORMAT),
        "message": message,
    }
