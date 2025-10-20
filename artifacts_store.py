"""Utilities for managing generated artifacts and keeping their index consistent."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

LOGGER = logging.getLogger("content_factory.artifacts")

ARTIFACTS_DIR = Path("artifacts").resolve()
INDEX_FILENAME = "index.json"


@dataclass
class ArtifactRecord:
    """Normalized representation of an artifact entry."""

    id: str
    path: str
    metadata_path: Optional[str]
    name: str
    updated_at: Optional[str]
    status: Optional[str]
    extra: Dict[str, Any]


def _index_path() -> Path:
    return ARTIFACTS_DIR / INDEX_FILENAME


def _ensure_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_artifact_path(raw_path: str | Path) -> Path:
    """Return absolute path within the artifacts directory."""

    base_dir = ARTIFACTS_DIR
    if not isinstance(raw_path, Path):
        candidate = Path(str(raw_path))
    else:
        candidate = raw_path

    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate == base_dir:
        raise ValueError("Запрошенный путь указывает на каталог artifacts")

    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:  # noqa: PERF203 - explicit error message helps debugging
        raise ValueError("Запрошенный путь вне каталога artifacts") from exc
    return candidate


def _read_index() -> List[Dict[str, Any]]:
    index_path = _index_path()
    if not index_path.exists():
        return []
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Не удалось разобрать artifacts/index.json: %s", exc)
        return []
    if not isinstance(raw, list):
        LOGGER.warning("Некорректный формат artifacts/index.json — ожидается массив")
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _write_index(entries: Sequence[Dict[str, Any]]) -> None:
    index_path = _index_path()
    _ensure_dir()
    index_path.write_text(
        json.dumps(list(entries), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(ARTIFACTS_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _read_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Некорректный JSON в %s: %s", path, exc)
        return {}


def _build_record_from_entry(entry: Dict[str, Any]) -> ArtifactRecord | None:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    try:
        artifact_path = resolve_artifact_path(raw_path)
    except ValueError:
        return None

    metadata_path_str = entry.get("metadata_path")
    metadata_path = None
    if isinstance(metadata_path_str, str) and metadata_path_str.strip():
        try:
            metadata_path = resolve_artifact_path(metadata_path_str)
        except ValueError:
            metadata_path = None

    record_id = str(entry.get("id") or artifact_path.stem)
    name = str(entry.get("name") or artifact_path.name)
    status = entry.get("status")
    updated_at = entry.get("updated_at")
    extra = {k: v for k, v in entry.items() if k not in {"id", "path", "metadata_path", "name", "updated_at", "status"}}
    return ArtifactRecord(
        id=record_id,
        path=_relative_path(artifact_path),
        metadata_path=_relative_path(metadata_path) if metadata_path else None,
        name=name,
        updated_at=str(updated_at) if updated_at else None,
        status=str(status) if status else None,
        extra=extra,
    )


def _build_record_from_file(path: Path, metadata: Optional[Dict[str, Any]] = None) -> ArtifactRecord:
    metadata_path = path.with_suffix(".json")
    payload = metadata if metadata is not None else _read_metadata(metadata_path)
    record_id = str(payload.get("id") or payload.get("artifact_id") or path.stem)
    status = payload.get("status") or ("Ready" if path.exists() else None)
    updated_at = payload.get("generated_at") or None
    name = payload.get("name") or path.name

    return ArtifactRecord(
        id=record_id,
        path=_relative_path(path),
        metadata_path=_relative_path(metadata_path) if metadata_path.exists() else None,
        name=name,
        updated_at=str(updated_at) if updated_at else None,
        status=str(status) if status else None,
        extra={},
    )


def register_artifact(markdown_path: Path, metadata: Optional[Dict[str, Any]] = None) -> ArtifactRecord:
    """Ensure that the artifact index contains an entry for the file."""

    resolved = resolve_artifact_path(markdown_path)
    payload = metadata if metadata is not None else _read_metadata(resolved.with_suffix(".json"))
    record = _build_record_from_file(resolved, payload)
    entries = _read_index()

    updated = False
    for idx, entry in enumerate(entries):
        candidate = _build_record_from_entry(entry)
        if candidate and (candidate.path == record.path or candidate.id == record.id):
            merged = dict(entry)
            merged.update(
                {
                    "id": record.id,
                    "path": record.path,
                    "metadata_path": record.metadata_path,
                    "name": record.name,
                    "status": record.status,
                    "updated_at": record.updated_at,
                }
            )
            entries[idx] = merged
            updated = True
            break
    if not updated:
        entries.append(
            {
                "id": record.id,
                "path": record.path,
                "metadata_path": record.metadata_path,
                "name": record.name,
                "status": record.status,
                "updated_at": record.updated_at,
            }
        )

    entries = _sort_entries(entries)
    _write_index(entries)
    return record


def _sort_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(entry: Dict[str, Any]) -> tuple:
        updated_at = entry.get("updated_at")
        return (str(updated_at) if updated_at else "", str(entry.get("name") or ""))

    return sorted(list(entries), key=_key, reverse=True)


def list_artifacts(theme: Optional[str] = None, *, auto_cleanup: bool = False) -> List[Dict[str, Any]]:
    """Return artifacts suitable for API output."""

    if auto_cleanup:
        cleanup_index()

    entries = _read_index()
    if entries:
        records = [rec for rec in (_build_record_from_entry(entry) for entry in entries) if rec]
    else:
        records = []
        artifacts_dir = ARTIFACTS_DIR
        if artifacts_dir.exists():
            for path in sorted(artifacts_dir.glob("*.md"), reverse=True):
                metadata = _read_metadata(path.with_suffix(".json"))
                record = _build_record_from_file(path, metadata)
                try:
                    register_artifact(path, metadata)
                except Exception as exc:  # noqa: BLE001 - keep listing even if index update fails
                    LOGGER.warning("Не удалось обновить индекс для %s: %s", path, exc)
                records.append(record)

    items: List[Dict[str, Any]] = []
    for record in records:
        artifact_path = resolve_artifact_path(record.path)
        metadata_path = None
        if record.metadata_path:
            try:
                metadata_path = resolve_artifact_path(record.metadata_path)
            except ValueError:
                metadata_path = None

        metadata = _read_metadata(metadata_path) if metadata_path else {}
        if theme and not _matches_theme(theme, metadata, artifact_path):
            continue

        stat = None
        if artifact_path.exists() and artifact_path.is_file():
            stat = artifact_path.stat()
        item = {
            "id": record.id,
            "name": record.name or artifact_path.name,
            "path": record.path,
            "metadata_path": record.metadata_path,
            "size": stat.st_size if stat else None,
            "modified_at": _format_iso(stat.st_mtime) if stat else record.updated_at,
            "metadata": metadata,
            "status": record.status or metadata.get("status") or "Ready",
        }
        items.append(item)
    items.sort(key=lambda item: item.get("modified_at") or "", reverse=True)
    return items


def _matches_theme(theme: str, metadata: Dict[str, Any], artifact_path: Path) -> bool:
    if not theme:
        return True
    normalized = theme.strip()
    if not normalized:
        return True
    meta_theme = metadata.get("theme")
    if isinstance(meta_theme, str) and meta_theme == normalized:
        return True
    if normalized and f"__{normalized}__" in artifact_path.name:
        return True
    return False


def _format_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


def delete_artifact(identifier: str) -> Dict[str, Any]:
    """Delete artifact files and remove the entry from the index."""

    entries = _read_index()
    target_entry = None
    for entry in entries:
        rec = _build_record_from_entry(entry)
        if not rec:
            continue
        if rec.id == identifier or rec.path == identifier:
            target_entry = rec
            break

    candidate_path = identifier
    if target_entry:
        candidate_path = target_entry.path

    result: Dict[str, Any] = {
        "deleted": False,
        "metadata_deleted": False,
        "not_found": False,
        "index_updated": False,
        "errors": [],
        "removed_id": target_entry.id if target_entry else None,
        "removed_path": target_entry.path if target_entry else None,
    }

    try:
        artifact_path = resolve_artifact_path(candidate_path)
    except ValueError as exc:
        if target_entry:
            # If the index entry is broken, drop it.
            new_entries = [
                entry
                for entry in entries
                if _build_record_from_entry(entry) and _build_record_from_entry(entry).path != target_entry.path
            ]
            if len(new_entries) != len(entries):
                _write_index(new_entries)
                result["index_updated"] = True
        result["errors"].append(str(exc))
        return result

    metadata_path = artifact_path.with_suffix(".json")

    try:
        artifact_path.unlink()
        result["deleted"] = True
    except FileNotFoundError:
        result["not_found"] = True
    except OSError as exc:  # noqa: BLE001 - capture failure but continue updating index
        result["errors"].append(str(exc))

    try:
        metadata_path.unlink()
        result["metadata_deleted"] = True
    except FileNotFoundError:
        pass
    except OSError as exc:  # noqa: BLE001
        result["errors"].append(str(exc))

    new_entries = []
    removed = False
    for entry in entries:
        rec = _build_record_from_entry(entry)
        if rec and (rec.path == _relative_path(artifact_path) or rec.id == result["removed_id"]):
            removed = True
            continue
        new_entries.append(entry)

    if removed:
        try:
            _write_index(new_entries)
            result["index_updated"] = True
        except OSError as exc:  # noqa: BLE001
            result["errors"].append(str(exc))
    return result


def cleanup_index() -> Dict[str, Any]:
    """Remove entries from the index if their files are missing."""

    entries = _read_index()
    if not entries:
        return {"checked": 0, "removed": 0, "errors": [], "removed_ids": []}

    kept: List[Dict[str, Any]] = []
    removed_ids: List[str] = []
    errors: List[str] = []

    for entry in entries:
        record = _build_record_from_entry(entry)
        if not record:
            removed_ids.append(str(entry.get("id") or ""))
            continue
        try:
            artifact_path = resolve_artifact_path(record.path)
        except ValueError as exc:
            errors.append(str(exc))
            removed_ids.append(record.id)
            continue
        if not artifact_path.exists():
            removed_ids.append(record.id)
            metadata_path = artifact_path.with_suffix(".json")
            try:
                metadata_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:  # noqa: BLE001
                errors.append(str(exc))
            continue
        kept.append(entry)

    removed = len(entries) - len(kept)
    if removed or errors:
        try:
            _write_index(kept)
        except OSError as exc:  # noqa: BLE001
            errors.append(str(exc))
    return {"checked": len(entries), "removed": removed, "errors": errors, "removed_ids": removed_ids}


__all__ = [
    "cleanup_index",
    "delete_artifact",
    "list_artifacts",
    "register_artifact",
    "resolve_artifact_path",
]
