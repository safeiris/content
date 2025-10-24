from __future__ import annotations

import pytest

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
