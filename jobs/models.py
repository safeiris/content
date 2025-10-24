"""Data models describing asynchronous generation jobs."""
from __future__ import annotations

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

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = self.started_at or utcnow()

    def mark_succeeded(self, result: Dict[str, Any], *, degradation_flags: Optional[List[str]] = None) -> None:
        self.status = JobStatus.SUCCEEDED
        self.result = result
        if degradation_flags:
            self.degradation_flags.extend(flag for flag in degradation_flags if flag)
        self.finished_at = utcnow()

    def mark_failed(self, error: str | Dict[str, Any], *, degradation_flags: Optional[List[str]] = None) -> None:
        self.status = JobStatus.FAILED
        self.error = {"message": error} if isinstance(error, str) else error
        self.finished_at = utcnow()
        if degradation_flags:
            self.degradation_flags.extend(flag for flag in degradation_flags if flag)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "created_at": self.created_at.strftime(ISO_FORMAT),
            "started_at": self.started_at.strftime(ISO_FORMAT) if self.started_at else None,
            "finished_at": self.finished_at.strftime(ISO_FORMAT) if self.finished_at else None,
            "steps": [step.to_dict() for step in self.steps],
            "result": self.result,
            "error": self.error,
            "degradation_flags": list(self.degradation_flags) or None,
            "trace_id": self.trace_id,
        }
