"""In-memory job store with TTL semantics."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from .models import Job, JobStep


class JobStore:
    """Thread-safe in-memory storage for jobs."""

    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._jobs: Dict[str, Job] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()

    def create(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._expiry[job.id] = time.time() + self._ttl_seconds
            self._purge_expired_locked()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._purge_expired_locked()
            return self._jobs.get(job_id)

    def update_step(self, job_id: str, step_name: str, mutator: Callable[[JobStep], None]) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for step in job.steps:
                if step.name == step_name:
                    mutator(step)
                    break
            self._expiry[job_id] = time.time() + self._ttl_seconds
            return job

    def set_result(self, job_id: str, result: dict, *, degradation_flags: Optional[list[str]] = None) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.mark_succeeded(result, degradation_flags=degradation_flags)
            self._expiry[job_id] = time.time() + self._ttl_seconds
            return job

    def set_failed(
        self,
        job_id: str,
        error: dict | str,
        *,
        degradation_flags: Optional[list[str]] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.mark_failed(error, degradation_flags=degradation_flags)
            self._expiry[job_id] = time.time() + self._ttl_seconds
            return job

    def touch(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._expiry[job_id] = time.time() + self._ttl_seconds

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._expiry.pop(job_id, None)

    def snapshot(self, job_id: str) -> Optional[dict]:
        job = self.get(job_id)
        return job.to_dict() if job else None

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [job_id for job_id, deadline in self._expiry.items() if deadline <= now]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._expiry.pop(job_id, None)

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            self._purge_expired_locked()
            return len(self._jobs)
