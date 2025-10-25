from __future__ import annotations

import pytest

from jobs.models import Job, JobStep
from jobs.runner import JobRunner
from jobs.store import JobStore


@pytest.fixture
def job_store() -> JobStore:
    return JobStore(ttl_seconds=30)


def test_job_runner_success(monkeypatch, job_store):
    def _fake_generate(**_kwargs):
        return {
            "text": "Hello world",
            "metadata": {
                "jsonld": {
                    "faq": [
                        {"question": "What?", "answer": "Answer"},
                    ]
                }
            },
        }

    monkeypatch.setattr("jobs.runner.generate_article_from_payload", _fake_generate)
    runner = JobRunner(job_store, soft_timeout_s=2)
    job = runner.submit({"theme": "demo", "data": {}, "k": 0}, trace_id="trace-1")
    assert runner.wait(job.id, timeout=5) is True
    snapshot = runner.get_job(job.id)
    assert snapshot["status"] == "succeeded"
    assert snapshot.get("step") == "done"
    assert snapshot.get("step_status") == "completed"
    assert snapshot.get("progress") == 1.0
    assert snapshot.get("message") == "Готово"
    assert snapshot.get("last_event_at")
    assert snapshot["result"]["markdown"].startswith("Hello")
    assert snapshot["degradation_flags"] in (None, [])
    assert snapshot["trace_id"] == "trace-1"


def test_job_runner_degradation(monkeypatch, job_store):
    def _raise_generate(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("jobs.runner.generate_article_from_payload", _raise_generate)
    runner = JobRunner(job_store, soft_timeout_s=1)
    job = runner.submit({"theme": "demo", "data": {}, "k": 0}, trace_id="trace-2")
    runner.wait(job.id, timeout=3)
    snapshot = runner.get_job(job.id)
    assert snapshot["status"] == "succeeded"
    assert snapshot.get("step") == "done"
    assert snapshot.get("progress") == 1.0
    assert snapshot.get("message") == "Готово"
    assert "draft_failed" in (snapshot.get("degradation_flags") or [])
    assert "markdown" in snapshot["result"]
    assert "demo" in snapshot["result"]["markdown"]


def test_job_runner_draft_degraded_on_max_tokens(monkeypatch, job_store):
    artifact_paths = {
        "markdown": "artifacts/demo.md",
        "metadata": "artifacts/demo.json",
    }

    def _fake_generate(**_kwargs):
        return {
            "text": "Частичный черновик",
            "metadata": {
                "degradation_flags": ["draft_max_tokens"],
                "completion_warning": "max_output_tokens",
            },
            "artifact_paths": artifact_paths,
        }

    monkeypatch.setattr("jobs.runner.generate_article_from_payload", _fake_generate)
    runner = JobRunner(job_store, soft_timeout_s=2)
    job = runner.submit({"theme": "demo", "data": {}, "k": 0})
    runner.wait(job.id, timeout=5)
    snapshot = runner.get_job(job.id)
    draft_step = next(step for step in snapshot["steps"] if step["name"] == "draft")
    assert draft_step["status"] == "degraded"
    assert draft_step["error"] == "max_output_tokens"
    assert draft_step["payload"]["artifact_paths"] == artifact_paths
    assert snapshot["result"]["artifact_paths"] == artifact_paths
    assert snapshot["result"].get("artifact_saved") is True
    assert "draft_max_tokens" in (snapshot.get("degradation_flags") or [])


def test_job_snapshot_includes_batch_counters(job_store):
    job = Job(id="demo-job", steps=[JobStep(name="draft")])
    job_store.create(job)

    job.update_progress(stage="draft", progress=0.25, payload={"total": 8, "completed": 2})

    snapshot = job_store.snapshot(job.id)
    assert snapshot["batches_total"] == 8
    assert snapshot["batches_completed"] == 2
