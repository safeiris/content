from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import artifacts_store  # noqa: E402


@pytest.fixture(autouse=True)
def patch_artifacts_dir(tmp_path, monkeypatch):
    base = tmp_path / "artifacts"
    base.mkdir()
    monkeypatch.setattr(artifacts_store, "ARTIFACTS_DIR", base)
    yield


def _write_artifact(base: Path, stem: str, content: str = "# demo") -> Path:
    markdown_path = base / f"{stem}.md"
    metadata_path = base / f"{stem}.json"
    markdown_path.write_text(content, encoding="utf-8")
    metadata = {
        "id": stem,
        "theme": "finance",
        "generated_at": "2024-01-01T00:00:00",
        "status": "Ready",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    return markdown_path


def test_register_and_list_artifacts(tmp_path):
    base = artifacts_store.ARTIFACTS_DIR
    path = _write_artifact(base, "sample")
    metadata = json.loads((path.with_suffix(".json")).read_text(encoding="utf-8"))

    record = artifacts_store.register_artifact(path, metadata)

    assert record.id == "sample"
    assert record.path.endswith("sample.md")

    items = artifacts_store.list_artifacts(auto_cleanup=False)
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "sample"
    assert item["metadata"]["theme"] == "finance"
    assert item["status"] == "Ready"


def test_delete_artifact_removes_files_and_index(tmp_path):
    base = artifacts_store.ARTIFACTS_DIR
    path = _write_artifact(base, "to_delete")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    artifacts_store.register_artifact(path, metadata)

    result = artifacts_store.delete_artifact("to_delete")

    assert result["deleted"] is True
    assert result["metadata_deleted"] is True
    assert result["index_updated"] is True

    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    assert artifacts_store.list_artifacts() == []


def test_cleanup_removes_missing_entries(tmp_path):
    base = artifacts_store.ARTIFACTS_DIR
    path = _write_artifact(base, "stale")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    artifacts_store.register_artifact(path, metadata)

    path.unlink()

    result = artifacts_store.cleanup_index()

    assert result["removed"] == 1
    assert result["removed_ids"] == ["stale"]
    assert artifacts_store.list_artifacts() == []
    assert not path.with_suffix(".json").exists()
