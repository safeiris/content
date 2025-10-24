import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifacts_store import register_artifact
from server import create_app


@pytest.fixture()
def app(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    monkeypatch.setattr("artifacts_store.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("server.ARTIFACTS_DIR", artifacts_dir)

    app = create_app()
    app.config.update(TESTING=True, ARTIFACTS_DIR=str(artifacts_dir))
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def sample_artifacts(app):
    artifacts_dir = Path(app.config["ARTIFACTS_DIR"])
    markdown_path = artifacts_dir / "2025-01-02_1200_test.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("# Заголовок\n\nСодержимое", encoding="utf-8")

    metadata = {
        "id": "artifact-001",
        "name": "2025-01-02_1200_test.md",
        "status": "succeeded",
        "job_id": "job-xyz",
        "generated_at": datetime(2025, 1, 2, 12, 1, tzinfo=timezone.utc).isoformat(),
    }
    metadata_path = markdown_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    register_artifact(markdown_path, metadata)

    return {
        "markdown_path": markdown_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def test_list_artifacts_returns_file_entries(client, sample_artifacts):
    response = client.get("/api/artifacts")
    assert response.status_code == 200

    payload = response.get_json()
    assert isinstance(payload, list)
    assert len(payload) == 2

    markdown_entry = next(item for item in payload if item["type"] == "md")
    json_entry = next(item for item in payload if item["type"] == "json")

    assert markdown_entry["name"].endswith(".md")
    assert markdown_entry["url"].startswith("/api/artifacts/")
    assert markdown_entry["job_id"] == "job-xyz"
    assert markdown_entry["status"] == "succeeded"
    assert markdown_entry["created_at"].endswith("Z")

    assert json_entry["name"].endswith(".json")
    assert json_entry["url"].startswith("/api/artifacts/")
    assert json_entry["metadata"]["job_id"] == "job-xyz"


@pytest.mark.parametrize(
    "extension,content_type",
    [
        (".md", "text/markdown"),
        (".json", "application/json"),
    ],
)
def test_download_artifact_serves_files_with_expected_mimetype(
    client, sample_artifacts, extension, content_type
):
    response = client.get(f"/api/artifacts/2025-01-02_1200_test{extension}")
    assert response.status_code == 200
    assert content_type in response.headers["Content-Type"]
    content_disposition = response.headers.get("Content-Disposition", "")
    assert "attachment" in content_disposition
    assert f"2025-01-02_1200_test{extension}" in content_disposition


def test_download_artifact_unknown_path_returns_404(client):
    response = client.get("/api/artifacts/not-found.md")
    assert response.status_code == 404
    data = response.get_json()
    assert data == {"error": "file_not_found"}
