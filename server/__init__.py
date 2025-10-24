"""Flask application exposing the content factory pipeline via HTTP."""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    g,
    redirect,
    render_template,
    request,
    session,
    send_file,
    send_from_directory,
    url_for,
)
from flask_cors import CORS
from werkzeug.security import check_password_hash

from assemble_messages import invalidate_style_profile_cache
from config import (
    DEFAULT_STRUCTURE,
    JOB_STORE_TTL_S,
    OPENAI_RPM,
    OPENAI_RPS,
)
from jobs import JobRunner, JobStatus, JobStore
from orchestrate import gather_health_status, make_generation_context
from llm_client import DEFAULT_MODEL
from retrieval import build_index
from artifacts_store import (
    cleanup_index as cleanup_artifact_index,
    delete_artifact as delete_artifact_entry,
    list_artifacts as list_artifact_cards,
    resolve_artifact_path,
)
from observability.logger import bind_trace_id, clear_trace_id, get_logger
from observability.metrics import get_registry

load_dotenv()

LOGGER = get_logger("content_factory.api")

PIPELINE_CONFIG_FILENAME = "pipeline.json"

JOB_STORE = JobStore(ttl_seconds=JOB_STORE_TTL_S)
JOB_RUNNER = JobRunner(JOB_STORE)

USERS: Dict[str, Dict[str, str]] = {
    "admin": {
        "display_name": "Admin",
        "password_hash": (
            "scrypt:32768:8:1$poFMhgLX1D2jug2W$724005a9a37b1f699ddda576ee89fb022c3bdcd28660826d1f9f5710c3116c6"
            "b847ea20c926c9124fbcfa9fee55967a26d488e3d04a3b58e2776f002a124d003"
        ),
    },
    "dmitriy": {
        "display_name": "Dmitriy",
        "password_hash": (
            "scrypt:32768:8:1$FRtm9J7DjkoGICbY$4f859f2fecaf592d3cffdec70a6c8ddb598a97e4851aa2f7c80d17ef5d87c02"
            "0b651cef85d9f82bf112f4ea46de4f25d17952a92c45c347000e3a413a0739af9"
        ),
    },
}


def login_required(view_func):
    """Decorator ensuring that a user is authenticated before accessing a view."""

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            next_url = request.full_path if request.query_string else request.path
            if next_url.endswith("?"):
                next_url = next_url[:-1]
            safe_next = _get_safe_redirect_target(next_url)
            if safe_next:
                return redirect(url_for("login", next=safe_next))
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


class ApiError(Exception):
    """Exception translated into an HTTP error response."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def create_app() -> Flask:
    frontend_root = (Path(__file__).resolve().parent.parent / "frontend_demo").resolve()
    template_root = frontend_root / "templates"
    app = Flask(__name__, template_folder=str(template_root))
    if hasattr(app, "json"):
        app.json.ensure_ascii = False
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    secret_key = os.environ.get("FLASK_SECRET_KEY")
    if not secret_key:
        LOGGER.warning("FLASK_SECRET_KEY is not set; generating a temporary secret key")
    app.secret_key = secret_key or secrets.token_hex(32)

    index_path = frontend_root / "index.html"

    @app.before_request
    def _bind_request_trace() -> None:
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
        g.trace_id = trace_id
        bind_trace_id(trace_id)

    @app.after_request
    def _append_trace(response):  # type: ignore[override]
        trace_id = getattr(g, "trace_id", None)
        if trace_id:
            response.headers.setdefault("X-Trace-Id", trace_id)
        clear_trace_id()
        return response

    @app.teardown_request
    def _teardown_trace(_exc):  # type: ignore[override]
        clear_trace_id()

    @app.context_processor
    def inject_current_user():
        username = session.get("user")
        return {
            "current_user": USERS.get(username),
            "current_username": username,
        }

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):  # type: ignore[override]
        LOGGER.warning("API error", extra={"message": exc.message, "code": exc.status_code})
        trace_id = getattr(g, "trace_id", None)
        return (
            jsonify(
                {
                    "error": {
                        "message": exc.message,
                        "code": exc.status_code,
                        "trace_id": trace_id,
                    }
                }
            ),
            exc.status_code,
        )

    @app.errorhandler(Exception)
    def _handle_generic_error(exc: Exception):  # type: ignore[override]
        LOGGER.exception("Unhandled error")
        trace_id = getattr(g, "trace_id", None)
        return (
            jsonify(
                {
                    "error": {
                        "message": "Внутренняя ошибка сервера",
                        "trace_id": trace_id,
                    }
                }
            ),
            500,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        next_param = _get_safe_redirect_target(
            request.args.get("next") or request.form.get("next")
        )

        if session.get("user"):
            if next_param:
                return redirect(next_param)
            return redirect(url_for("serve_frontend", path=""))

        username = ""
        if request.method == "POST":
            username = str(request.form.get("username", "")).strip()
            password = request.form.get("password", "")
            user = USERS.get(username)

            if user and check_password_hash(user["password_hash"], password):
                session.clear()
                session["user"] = username
                session["authenticated_at"] = datetime.utcnow().isoformat()
                redirect_target = next_param or url_for("serve_frontend", path="")
                return redirect(redirect_target)

            flash("Неверный логин или пароль", "error")

        return render_template("login.html", next=next_param, username=username)

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

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
        raw_data = dict(raw_data)

        explicit_keywords = payload.get("keywords")
        if explicit_keywords and "keywords" not in raw_data:
            raw_data["keywords"] = explicit_keywords

        explicit_length = payload.get("length_range")
        if (
            isinstance(explicit_length, dict)
            and "length_limits" not in raw_data
            and {"min", "max"} <= set(explicit_length.keys())
        ):
            raw_data["length_limits"] = {
                "min_chars": explicit_length.get("min"),
                "max_chars": explicit_length.get("max"),
            }

        if "faq_required" in payload and "include_faq" not in raw_data:
            raw_data["include_faq"] = bool(payload.get("faq_required"))

        if payload.get("faq_count") and "faq_questions" not in raw_data:
            raw_data["faq_questions"] = payload.get("faq_count")

        style_payload = payload.get("style")
        if isinstance(style_payload, dict):
            raw_data["style"] = style_payload

        k = _safe_int(payload.get("k"))
        if k < 0:
            k = 0

        style_profile_override = _style_profile_override_from_request(request)

        context_source, context_text, context_filename = _extract_context_settings(payload, raw_data)
        effective_k = k
        if context_source in {"off", "custom"}:
            effective_k = 0

        generation_context = make_generation_context(
            theme=theme,
            data=raw_data,
            k=effective_k,
            append_style_profile=style_profile_override,
            context_source=context_source,
            custom_context_text=context_text,
            context_filename=context_filename,
        )

        system_payload = next((msg for msg in generation_context.messages if msg.get("role") == "system"), {})
        system_message = str(system_payload.get("content", ""))
        user_message = next(
            (msg["content"] for msg in reversed(generation_context.messages) if msg.get("role") == "user"),
            "",
        )

        style_profile_applied = bool(system_payload.get("style_profile_applied"))
        style_profile_source = system_payload.get("style_profile_source") if style_profile_applied else None
        style_profile_variant = system_payload.get("style_profile_variant") if style_profile_applied else None

        context_bundle = generation_context.context_bundle
        context_items = [
            {
                "path": item.get("path"),
                "score": item.get("score"),
                "text": item.get("text"),
                "token_estimate": item.get("token_estimate"),
            }
            for item in context_bundle.items
        ]

        if generation_context.context_source == "custom":
            context_used = bool(generation_context.custom_context_text)
        else:
            context_used = bool(
                context_bundle.context_used and not context_bundle.index_missing and effective_k > 0
            )

        response_payload: Dict[str, Any] = {
            "system": system_message,
            "context": context_items,
            "user": user_message,
            "context_used": context_used,
            "context_index_missing": context_bundle.index_missing,
            "context_budget_tokens_est": context_bundle.total_tokens_est,
            "context_budget_tokens_limit": context_bundle.token_budget_limit,
            "k": effective_k,
            "context_source": generation_context.context_source,
            "style_profile_applied": style_profile_applied,
            "keywords_manual": generation_context.keywords_manual,
        }

        if generation_context.context_source == "custom":
            response_payload["context_text"] = generation_context.custom_context_text or ""
            response_payload["context_len"] = generation_context.custom_context_len
            if generation_context.custom_context_filename:
                response_payload["context_filename"] = generation_context.custom_context_filename

        if style_profile_applied and style_profile_source:
            response_payload["style_profile_source"] = style_profile_source
        if style_profile_applied and style_profile_variant:
            response_payload["style_profile_variant"] = style_profile_variant

        return jsonify(response_payload)

    @app.post("/api/generate")
    def generate():
        payload = _require_json(request)
        theme = str(payload.get("theme", "")).strip()
        if not theme:
            raise ApiError("Не указана тема (theme)")

        raw_data = payload.get("data") or {}
        if not isinstance(raw_data, dict):
            raise ApiError("Поле data должно быть объектом")
        raw_data = dict(raw_data)

        style_payload = payload.get("style")
        if isinstance(style_payload, dict):
            raw_data["style"] = style_payload

        k = _safe_int(payload.get("k"))
        if k < 0:
            k = 0

        context_source, context_text, context_filename = _extract_context_settings(payload, raw_data)
        effective_k = k
        if context_source in {"off", "custom"}:
            effective_k = 0

        style_profile_override = _style_profile_override_from_request(request)
        if payload.get("dry_run"):
            return jsonify(_make_dry_run_response(theme=theme, data=raw_data, k=effective_k))

        requested_model = str(payload.get("model", "")).strip()
        if requested_model and requested_model.lower() != DEFAULT_MODEL.lower():
            LOGGER.info(
                "IGNORED_MODEL_OVERRIDE requested=%s enforced=%s",
                requested_model,
                DEFAULT_MODEL,
            )
        model = DEFAULT_MODEL
        temperature = _safe_float(payload.get("temperature", 0.3), default=0.3)
        temperature = max(0.0, min(2.0, temperature))
        max_tokens = max(1, _safe_int(payload.get("max_tokens", 1400), default=1400))
        task_payload = {
            "theme": theme,
            "data": raw_data,
            "k": k,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "append_style_profile": style_profile_override,
            "context_source": context_source,
            "context_text": context_text,
            "context_filename": context_filename,
        }

        trace_id = getattr(g, "trace_id", None)
        job = JOB_RUNNER.submit(task_payload, trace_id=trace_id)
        sync_raw = payload.get("sync", request.args.get("sync"))
        sync_requested = str(sync_raw).lower() in {"1", "true", "yes"}

        if sync_requested:
            finished = JOB_RUNNER.wait(job.id, timeout=JOB_RUNNER.soft_timeout())
            snapshot = JOB_RUNNER.get_job(job.id)
            if snapshot:
                status = snapshot.get("status")
                if status == JobStatus.SUCCEEDED.value and snapshot.get("result"):
                    response_payload = _format_generation_success(snapshot["result"])
                    response_payload["job_id"] = job.id
                    response_payload["degradation_flags"] = snapshot.get("degradation_flags") or []
                    response_payload["trace_id"] = snapshot.get("trace_id")
                    return jsonify(response_payload)
                if status == JobStatus.FAILED.value:
                    trace_id = snapshot.get("trace_id") or getattr(g, "trace_id", None)
                    error_payload = snapshot.get("error") or {}
                    message = error_payload.get("message") if isinstance(error_payload, dict) else str(error_payload)
                    return (
                        jsonify(
                            {
                                "error": {
                                    "message": message or "Generation failed",
                                    "trace_id": trace_id,
                                },
                                "job_id": job.id,
                                "status": status,
                            }
                        ),
                        500,
                    )
            if finished:
                # No snapshot available, treat as failure
                trace_id = getattr(g, "trace_id", None)
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "Job completed without result",
                                "trace_id": trace_id,
                            },
                            "job_id": job.id,
                        }
                    ),
                    500,
                )

        snapshot = JOB_RUNNER.get_job(job.id) or job.to_dict()
        response_payload = {
            "job_id": job.id,
            "status": snapshot.get("status", JobStatus.PENDING.value),
            "steps": snapshot.get("steps"),
            "result": snapshot.get("result"),
            "degradation_flags": snapshot.get("degradation_flags"),
            "trace_id": snapshot.get("trace_id") or getattr(g, "trace_id", None),
        }
        http_status = 200 if response_payload["status"] == JobStatus.SUCCEEDED.value else 202
        return jsonify(response_payload), http_status

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        snapshot = JOB_RUNNER.get_job(job_id)
        if not snapshot:
            trace_id = getattr(g, "trace_id", None)
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Job not found",
                            "trace_id": trace_id,
                        }
                    }
                ),
                404,
            )
        return jsonify(snapshot)

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

        invalidate_style_profile_cache()
        return jsonify(stats)

    @app.get("/api/artifacts")
    def list_artifacts():
        theme = request.args.get("theme")
        items = list_artifact_cards(theme, auto_cleanup=True)
        return jsonify(items)

    @app.delete("/api/artifacts")
    def delete_artifact():
        payload = _require_json(request)
        identifier = str(payload.get("id") or payload.get("path") or "").strip()
        if not identifier:
            raise ApiError("Не указан идентификатор артефакта", status_code=400)

        result = delete_artifact_entry(identifier)
        status_code = 200
        if result.get("errors"):
            status_code = 207

        response_payload: Dict[str, Any] = {
            "deleted": bool(result.get("deleted")),
            "metadata_deleted": bool(result.get("metadata_deleted")),
            "not_found": bool(result.get("not_found")),
            "index_updated": bool(result.get("index_updated")),
            "removed_id": result.get("removed_id"),
            "removed_path": result.get("removed_path"),
        }
        if result.get("errors"):
            response_payload["errors"] = list(result["errors"])
        return jsonify(response_payload), status_code

    @app.post("/api/artifacts/cleanup")
    def cleanup_artifacts():
        result = cleanup_artifact_index()
        http_status = 200 if not result.get("errors") else 207
        return jsonify(result), http_status

    @app.get("/api/artifacts/download")
    def download_artifact():
        raw_path = request.args.get("path")
        if not raw_path:
            raise ApiError("Не указан путь к артефакту", status_code=400)

        try:
            artifact_path = resolve_artifact_path(raw_path)
        except ValueError as exc:
            raise ApiError(str(exc), status_code=400) from exc
        if not artifact_path.exists() or not artifact_path.is_file():
            return jsonify({"error": "file_not_found"}), 404

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
        checks = status.setdefault("checks", {})
        metrics_snapshot = get_registry().snapshot()
        queue_len = int(metrics_snapshot.get("jobs.queue_length", 0))
        checks["openai_rate_limits"] = {
            "ok": True,
            "message": f"Лимиты клиента активны: {OPENAI_RPS} rps / {OPENAI_RPM} rpm",
        }
        checks["job_runner"] = {
            "ok": True,
            "message": f"Очередь заданий: {len(JOB_STORE)}; soft timeout={JOB_RUNNER.soft_timeout()}s",
        }
        checks["job_queue"] = {
            "ok": queue_len < 10,
            "message": f"Размер очереди: {queue_len}",
        }
        if theme:
            profile_path = (Path("profiles") / theme / "style_profile.md").resolve()
            try:
                mtime = profile_path.stat().st_mtime
                freshness_days = max(0, (time.time() - mtime) / 86400.0)
                checks["theme_profile_freshness"] = {
                    "ok": freshness_days < 30,
                    "message": f"Профиль обновлен {freshness_days:.1f} дн. назад",
                }
            except FileNotFoundError:
                checks["theme_profile_freshness"] = {
                    "ok": False,
                    "message": f"Файл профиля не найден: {profile_path.as_posix()}",
                }
        http_status = 200 if status.get("ok") else 503
        return jsonify(status), http_status

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    @login_required
    def serve_frontend(path: str):
        if path.startswith("api/"):
            raise ApiError("Endpoint not found", status_code=404)

        candidate = (frontend_root / path).resolve()
        try:
            candidate.relative_to(frontend_root)
        except ValueError:
            abort(404)

        if candidate.is_file():
            relative_path = candidate.relative_to(frontend_root)
            return send_from_directory(frontend_root, relative_path.as_posix())

        if index_path.exists():
            return send_from_directory(frontend_root, index_path.name)

        if (template_root / "index.html").exists():
            return render_template("index.html")

        abort(404)

    return app


def _get_safe_redirect_target(target: Optional[str]) -> Optional[str]:
    if not target:
        return None
    if target.startswith("//"):
        return None

    host_url = request.host_url
    resolved_target = urljoin(host_url, target)
    parsed_host = urlparse(host_url)
    parsed_target = urlparse(resolved_target)

    if parsed_target.scheme not in {"http", "https"}:
        return None
    if parsed_host.netloc != parsed_target.netloc:
        return None

    safe_path = parsed_target.path or "/"
    if parsed_target.query:
        safe_path = f"{safe_path}?{parsed_target.query}"
    if parsed_target.fragment:
        safe_path = f"{safe_path}#{parsed_target.fragment}"
    return safe_path


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


def _style_profile_override_from_request(req) -> Optional[bool]:
    raw_value = req.args.get("style_profile") if hasattr(req, "args") else None
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if value in {"off", "false", "0"}:
        return False
    if value in {"on", "true", "1"}:
        return True
    return None


def _normalize_context_source(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"", "index", "index.json"}:
        return "index.json"
    if value == "custom":
        return "custom"
    if value == "off":
        return "off"
    return "index.json"


def _is_allowed_context_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".txt", ".json"}


def _extract_context_settings(payload: Dict[str, Any], raw_data: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    context_source = _normalize_context_source(
        payload.get("context_source") or raw_data.get("context_source")
    )

    filename_raw = payload.get("context_filename")
    if filename_raw is None and "context_filename" in raw_data:
        filename_raw = raw_data.get("context_filename")
    context_filename = (
        str(filename_raw).strip() if isinstance(filename_raw, str) and filename_raw.strip() else None
    )

    context_text_raw = payload.get("context_text")
    if context_text_raw is None and context_source == "custom":
        context_text_raw = raw_data.get("context_text")

    if context_source == "custom":
        if not isinstance(context_text_raw, str):
            raise ApiError("Поле context_text обязательно для custom")
        context_text = context_text_raw
        if not context_text.strip():
            raise ApiError("Поле context_text обязательно для custom")
        if context_filename and not _is_allowed_context_file(context_filename):
            raise ApiError("Поддерживаются только .txt и .json")
        return context_source, context_text, context_filename

    return context_source, None, None


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
        pipeline_config = _load_pipeline_config(entry)
        pipe_payload: Dict[str, Any] = {
            "id": slug,
            "name": name,
            "description": description or f"Тематика {name}",
            "tone": tone or "экспертный",
            "keywords": keywords,
            "default_structure": DEFAULT_STRUCTURE,
        }
        default_model = str(pipeline_config.get("default_model", "")).strip()
        if default_model:
            pipe_payload["default_model"] = default_model
        pipes.append(pipe_payload)
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


def _load_pipeline_config(theme_dir: Path) -> Dict[str, Any]:
    config_path = theme_dir / PIPELINE_CONFIG_FILENAME
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Повреждён pipeline config: %s", config_path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_generation_success(result: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "markdown": result.get("text"),
        "meta_json": result.get("metadata"),
        "artifact_paths": result.get("artifact_paths"),
    }
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        if metadata.get("style_profile_applied"):
            payload["style_profile_applied"] = True
            if metadata.get("style_profile_source"):
                payload["style_profile_source"] = metadata["style_profile_source"]
            if metadata.get("style_profile_variant"):
                payload["style_profile_variant"] = metadata["style_profile_variant"]
        if metadata.get("keywords_manual") is not None:
            payload["keywords_manual"] = metadata.get("keywords_manual")
        payload["model_used"] = metadata.get("model_used")
        payload["fallback_used"] = metadata.get("fallback_used")
        payload["fallback_reason"] = metadata.get("fallback_reason")
        payload["api_route"] = metadata.get("api_route")
    return payload


def _make_dry_run_response(*, theme: str, data: Dict[str, Any], k: int) -> Dict[str, Any]:
    topic = str(data.get("theme") or data.get("goal") or theme).strip() or "Тема не указана"
    markdown = (
        f"# Черновик (dry run)\n\n"
        f"Тематика: {theme}\n\n"
        f"Запрошенная тема: {topic}\n\n"
        "Этот ответ сформирован без обращения к модели."
    )
    generated_at = datetime.utcnow().isoformat()
    metadata: Dict[str, Any] = {
        "model_used": "dry-run",
        "characters": len(markdown),
        "generated_at": generated_at,
        "theme": theme,
        "retrieval_k": k,
        "input_data": data,
        "clips": [],
        "context_used": False,
        "context_index_missing": False,
        "context_budget_tokens_est": 0,
        "context_budget_tokens_limit": 0,
        "system_prompt_preview": "",
        "user_prompt_preview": "",
        "retry_used": False,
        "plagiarism_detected": False,
        "postfix_appended": False,
        "disclaimer_appended": False,
        "length_adjustment": None,
        "status": "dry-run",
        "fallback_used": None,
        "fallback_reason": None,
    }
    return {
        "markdown": markdown,
        "meta_json": metadata,
        "artifact_paths": {
            "markdown": None,
            "metadata": None,
        },
    }


app = create_app()
