"""Flask application exposing the content factory pipeline via HTTP."""
from __future__ import annotations

import json
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from assemble_messages import assemble_messages, retrieve_context
from config import DEFAULT_STRUCTURE
from orchestrate import gather_health_status, generate_article_from_payload
from retrieval import build_index

load_dotenv()

LOGGER = logging.getLogger("content_factory.api")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


class ApiError(Exception):
    """Exception translated into an HTTP error response."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):  # type: ignore[override]
        LOGGER.warning("API error: %s", exc.message)
        return jsonify({"error": exc.message}), exc.status_code

    @app.errorhandler(Exception)
    def _handle_generic_error(exc: Exception):  # type: ignore[override]
        LOGGER.exception("Unhandled error")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

    @app.get("/api/pipes")
    def list_pipes():
        pipes = _collect_pipes()
        return jsonify(pipes)

    @app.post("/api/prompt/preview")
    def prompt_preview():
        payload = _require_json(request)
        theme = str(payload.get("theme", "")).strip()
        if not theme:
            raise ApiError("Не указана тема (theme)")

        raw_data = payload.get("data") or {}
        if not isinstance(raw_data, dict):
            raise ApiError("Поле data должно быть объектом")

        k = _safe_int(payload.get("k", 3))
        if k < 0:
            k = 0

        context_bundle = retrieve_context(theme_slug=theme, query=raw_data.get("theme", ""), k=k)
        messages = assemble_messages(
            data_path="",
            theme_slug=theme,
            k=k,
            exemplars=context_bundle.items,
            data=raw_data,
        )

        system_message = next((msg["content"] for msg in messages if msg.get("role") == "system"), "")
        user_message = next((msg["content"] for msg in reversed(messages) if msg.get("role") == "user"), "")

        context_items = [
            {
                "path": item.get("path"),
                "score": item.get("score"),
                "text": item.get("text"),
                "token_estimate": item.get("token_estimate"),
            }
            for item in context_bundle.items
        ]

        return jsonify(
            {
                "system": system_message,
                "context": context_items,
                "user": user_message,
                "context_used": bool(context_bundle.context_used and not context_bundle.index_missing and k > 0),
                "context_index_missing": context_bundle.index_missing,
                "context_budget_tokens_est": context_bundle.total_tokens_est,
                "context_budget_tokens_limit": context_bundle.token_budget_limit,
                "k": k,
            }
        )

    @app.post("/api/generate")
    def generate():
        payload = _require_json(request)
        theme = str(payload.get("theme", "")).strip()
        if not theme:
            raise ApiError("Не указана тема (theme)")

        raw_data = payload.get("data") or {}
        if not isinstance(raw_data, dict):
            raise ApiError("Поле data должно быть объектом")

        k = _safe_int(payload.get("k", 3))
        if k < 0:
            k = 0

        model = payload.get("model")
        temperature = _safe_float(payload.get("temperature", 0.3), default=0.3)
        temperature = max(0.0, min(2.0, temperature))
        max_tokens = max(1, _safe_int(payload.get("max_tokens", 1400), default=1400))

        try:
            result = generate_article_from_payload(
                theme=theme,
                data=raw_data,
                k=k,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ApiError:
            raise
        except RuntimeError as exc:
            raise ApiError(str(exc), status_code=503)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Generation failed")
            raise ApiError("Не удалось завершить генерацию", status_code=500) from exc

        return jsonify(
            {
                "markdown": result["text"],
                "meta_json": result["metadata"],
                "artifact_paths": result["artifact_paths"],
            }
        )

    @app.post("/api/reindex")
    def reindex():
        payload = _require_json(request)
        theme = str(payload.get("theme", "")).strip()
        if not theme:
            raise ApiError("Не указана тема (theme)")

        try:
            stats = build_index(theme)
        except FileNotFoundError as exc:
            raise ApiError(str(exc), status_code=404) from exc
        except RuntimeError as exc:
            raise ApiError(str(exc), status_code=400) from exc

        return jsonify(stats)

    @app.get("/api/artifacts")
    def list_artifacts():
        theme = request.args.get("theme")
        items = _collect_artifacts(theme)
        return jsonify(items)

    @app.get("/api/artifacts/download")
    def download_artifact():
        raw_path = request.args.get("path")
        if not raw_path:
            raise ApiError("Не указан путь к артефакту", status_code=400)

        artifact_path = _resolve_artifact_path(raw_path)
        if not artifact_path.exists() or not artifact_path.is_file():
            raise ApiError("Файл не найден", status_code=404)

        mime_type, _ = mimetypes.guess_type(artifact_path.name)
        return send_file(
            artifact_path,
            mimetype=mime_type or "application/octet-stream",
            as_attachment=True,
            download_name=artifact_path.name,
        )

    @app.get("/api/health")
    def health():
        theme = request.args.get("theme")
        status = gather_health_status(theme)
        http_status = 200 if status.get("ok") else 503
        return jsonify(status), http_status

    return app


def _require_json(req) -> Dict[str, Any]:
    try:
        data = req.get_json(force=True)  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001
        raise ApiError("Некорректный JSON") from exc
    if not isinstance(data, dict):
        raise ApiError("Ожидается JSON-объект")
    return data


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_pipes() -> List[Dict[str, Any]]:
    base_dir = Path("profiles")
    pipes: List[Dict[str, Any]] = []

    if not base_dir.exists():
        return pipes

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        slug = entry.name
        name = slug.replace("_", " ").title()
        description, tone = _extract_style(entry / "style_guide.md")
        keywords = _extract_keywords(entry / "glossary.txt")
        pipes.append(
            {
                "id": slug,
                "name": name,
                "description": description or f"Тематика {name}",
                "tone": tone or "экспертный",
                "keywords": keywords,
                "default_structure": DEFAULT_STRUCTURE,
            }
        )
    return pipes


def _extract_style(style_path: Path) -> Tuple[str, str]:
    if not style_path.exists():
        return "", ""
    lines = [line.strip() for line in style_path.read_text(encoding="utf-8").splitlines()]
    description = next((line.lstrip("- ") for line in lines if line and not line.startswith("#")), "")

    tone = ""
    try:
        tone_index = lines.index("## Тональность")
    except ValueError:
        tone_index = -1
    if tone_index >= 0:
        for candidate in lines[tone_index + 1 :]:
            if candidate.startswith("- "):
                tone = candidate.lstrip("- ")
                break
            if candidate.startswith("## "):
                break
    return description, tone


def _extract_keywords(glossary_path: Path) -> List[str]:
    if not glossary_path.exists():
        return []
    keywords: List[str] = []
    for line in glossary_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        keyword = cleaned.split("—", 1)[0].strip()
        if keyword:
            keywords.append(keyword)
        if len(keywords) >= 6:
            break
    return keywords


def _collect_artifacts(theme: str | None) -> List[Dict[str, Any]]:
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        return []

    items: List[Dict[str, Any]] = []
    for path in sorted(artifacts_dir.glob("*.md"), reverse=True):
        metadata_path = path.with_suffix(".json")
        metadata = _read_json(metadata_path)
        theme_slug = metadata.get("theme") if isinstance(metadata, dict) else None
        if theme:
            if theme_slug and theme_slug != theme:
                continue
            if not theme_slug and f"__{theme}__" not in path.name:
                continue
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": path.as_posix(),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "metadata_path": metadata_path.as_posix(),
                "metadata": metadata,
            }
        )
    return items


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_artifact_path(raw_path: str) -> Path:
    artifacts_dir = Path("artifacts").resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (artifacts_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if artifacts_dir not in candidate.parents and candidate != artifacts_dir:
        raise ApiError("Запрошенный путь вне каталога artifacts", status_code=400)
    return candidate


app = create_app()
