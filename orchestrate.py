# -*- coding: utf-8 -*-
"""End-to-end pipeline: assemble prompt → call LLM → store artefacts."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

from assemble_messages import ContextBundle, assemble_messages, retrieve_context
from llm_client import DEFAULT_MODEL, generate as llm_generate
from plagiarism_guard import is_too_similar


BELGRADE_TZ = ZoneInfo("Europe/Belgrade")
DEFAULT_CTA_TEXT = (
    "Семейная ипотека помогает молодым семьям купить жильё на понятных условиях. "
    "Сравните программы банков и сделайте первый шаг к дому своей мечты уже сегодня."
)
LENGTH_EXTEND_THRESHOLD = 1800
LENGTH_SHRINK_THRESHOLD = 4500
TARGET_LENGTH_RANGE: Tuple[int, int] = (2000, 4000)


@dataclass
class GenerationContext:
    data: Dict[str, Any]
    context_bundle: ContextBundle
    messages: List[Dict[str, str]]
    clip_texts: List[str]


def _get_cta_text() -> str:
    cta = os.getenv("DEFAULT_CTA", DEFAULT_CTA_TEXT).strip()
    return cta or DEFAULT_CTA_TEXT


def _is_truncated(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith("…") or stripped.endswith("...") or stripped.endswith(","):
        return True
    paragraphs = [para.strip() for para in stripped.splitlines() if para.strip()]
    if not paragraphs:
        return False
    last_paragraph = paragraphs[-1]
    return not last_paragraph.endswith((".", "!", "?"))


def _append_cta_if_needed(text: str) -> Tuple[str, bool]:
    if not _is_truncated(text):
        return text, False
    cta = _get_cta_text()
    if text.strip():
        return text.rstrip() + "\n\n" + cta, True
    return cta, True


def _choose_section_for_extension(data: Dict[str, Any]) -> str:
    structure = data.get("structure")
    if isinstance(structure, Iterable):
        structure_list = [str(item).strip() for item in structure if str(item).strip()]
        if len(structure_list) >= 2:
            return structure_list[1]
        if structure_list:
            return structure_list[0]
    return "основную часть"


def _build_extend_prompt(section_name: str) -> str:
    return f"Раскрой раздел «{section_name}» примерами и чек-листом, сохрани структуру материала."


def _build_shrink_prompt() -> str:
    return "Сократи повторы, оставь ключевые тезисы и сохрани исходную структуру текста."


def _ensure_length(
    text: str,
    messages: List[Dict[str, str]],
    *,
    data: Dict[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Tuple[str, Optional[str], List[Dict[str, str]]]:
    length = len(text)
    if length < LENGTH_EXTEND_THRESHOLD:
        section = _choose_section_for_extension(data)
        prompt = _build_extend_prompt(section)
        adjusted_messages = list(messages)
        adjusted_messages.append({"role": "user", "content": prompt})
        new_text = llm_generate(
            adjusted_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout,
        )
        return new_text, "extend", adjusted_messages
    if length > LENGTH_SHRINK_THRESHOLD:
        prompt = _build_shrink_prompt()
        adjusted_messages = list(messages)
        adjusted_messages.append({"role": "user", "content": prompt})
        new_text = llm_generate(
            adjusted_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout,
        )
        return new_text, "shrink", adjusted_messages
    return text, None, messages


def _local_now() -> datetime:
    return datetime.now(BELGRADE_TZ)


def _make_generation_context(
    *,
    theme: str,
    data: Dict[str, Any],
    k: int,
) -> GenerationContext:
    if k <= 0:
        bundle = ContextBundle(items=[], total_tokens_est=0, index_missing=False, context_used=False)
    else:
        bundle = retrieve_context(theme_slug=theme, query=data.get("theme", ""), k=k)
        if bundle.index_missing:
            print("[orchestrate] CONTEXT: none (index missing)")
    messages = assemble_messages(data_path="", theme_slug=theme, k=k, exemplars=bundle.items, data=data)
    clip_texts = [str(item.get("text", "")) for item in bundle.items if item.get("text")]
    return GenerationContext(data=data, context_bundle=bundle, messages=messages, clip_texts=clip_texts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an article using the configured LLM.")
    parser.add_argument("--theme", help="Theme slug (matches profiles/<theme>/...)")
    parser.add_argument("--data", help="Path to the JSON brief with generation parameters.")
    parser.add_argument(
        "--outfile",
        help="Optional path for the resulting markdown. Defaults to artifacts/<timestamp>__<theme>__article.md",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of exemplar clips to attach to CONTEXT (default: 3).")
    parser.add_argument("--model", help="Override model name (otherwise uses LLM_MODEL env or default).")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3).")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1400,
        dest="max_tokens",
        help="Max tokens for generation (default: 1400).",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per request in seconds (default: 60).")
    parser.add_argument("--mode", choices=["draft", "final"], default="final", help="Execution mode for metadata tags.")
    parser.add_argument("--ab", choices=["compare"], help="Run A/B comparison (compare: without vs with context).")
    parser.add_argument("--batch", help="Path to a JSON/YAML file describing batch generation payloads.")
    parser.add_argument("--check", action="store_true", help="Validate environment prerequisites and exit.")
    return parser.parse_args()


def _load_input(path: str) -> Dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Не найден файл входных данных: {payload_path}")
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {payload_path}: {exc}") from exc


def _resolve_model(cli_model: str | None) -> str:
    candidate = (cli_model or os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()
    return candidate or DEFAULT_MODEL


def _make_output_path(theme: str, outfile: str | None) -> Path:
    if outfile:
        return Path(outfile)
    timestamp = _local_now().strftime("%Y-%m-%d_%H%M")
    filename = f"{timestamp}__{theme}__article.md"
    return Path("artifacts") / filename


def _write_outputs(markdown_path: Path, text: str, metadata: Dict[str, Any]) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text, encoding="utf-8")
    metadata_path = markdown_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_variant(
    *,
    theme: str,
    data: Dict[str, Any],
    data_path: str,
    k: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    mode: str,
    output_path: Path,
    variant_label: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    generation_context = _make_generation_context(theme=theme, data=data, k=k)
    active_messages = list(generation_context.messages)

    article_text = llm_generate(
        active_messages,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout,
    )

    retry_used = False
    if generation_context.clip_texts and is_too_similar(article_text, generation_context.clip_texts):
        retry_used = True
        active_messages = list(active_messages)
        active_messages.append(
            {
                "role": "user",
                "content": "Перефразируй разделы, добавь списки и FAQ, избегай совпадений с примерами.",
            }
        )
        print("[orchestrate] Обнаружено совпадение с примерами, выполняю перегенерацию...", file=sys.stderr)
        article_text = llm_generate(
            active_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout,
        )

    article_text, length_adjustment, active_messages = _ensure_length(
        article_text,
        active_messages,
        data=data,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    article_text, postfix_appended = _append_cta_if_needed(article_text)

    duration = time.time() - start_time
    context_bundle = generation_context.context_bundle
    context_used = bool(context_bundle.context_used and not context_bundle.index_missing and k > 0)

    metadata: Dict[str, Any] = {
        "theme": theme,
        "data_path": data_path,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_s": timeout,
        "retrieval_k": k,
        "context_applied_k": len(context_bundle.items),
        "clips": [
            {
                "path": item.get("path"),
                "score": item.get("score"),
                "token_estimate": item.get("token_estimate"),
            }
            for item in context_bundle.items
        ],
        "plagiarism_flagged": retry_used,
        "retry_used": retry_used,
        "generated_at": _local_now().isoformat(),
        "duration_seconds": round(duration, 3),
        "characters": len(article_text),
        "words": len(article_text.split()) if article_text.strip() else 0,
        "messages_count": len(active_messages),
        "context_used": context_used,
        "context_index_missing": context_bundle.index_missing,
        "context_budget_tokens_est": context_bundle.total_tokens_est,
        "postfix_appended": postfix_appended,
        "length_adjustment": length_adjustment,
        "length_range_target": {"min": TARGET_LENGTH_RANGE[0], "max": TARGET_LENGTH_RANGE[1]},
        "mode": mode,
        "model_used": model_name,
        "temperature_used": temperature,
        "max_tokens_used": max_tokens,
    }
    if variant_label:
        metadata["ab_variant"] = variant_label

    _write_outputs(output_path, article_text, metadata)
    _summarise(theme, k, model_name, article_text, variant=variant_label)

    return {
        "text": article_text,
        "metadata": metadata,
        "output_path": output_path,
        "duration": duration,
        "variant": variant_label,
    }


def _summarise(theme: str, k: int, model: str, text: str, *, variant: str | None = None) -> None:
    chars = len(text)
    words = len(text.split()) if text.strip() else 0
    suffix = f" variant={variant}" if variant else ""
    print(f"[orchestrate] theme={theme}{suffix} k={k} model={model} length={chars} chars / {words} words")


def _suffix_output_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")


def _run_ab_compare(
    *,
    theme: str,
    data: Dict[str, Any],
    data_path: str,
    model_name: str,
    args: argparse.Namespace,
    base_output_path: Path,
) -> None:
    path_a = _suffix_output_path(base_output_path, "__A")
    path_b = _suffix_output_path(base_output_path, "__B")

    result_a = _generate_variant(
        theme=theme,
        data=data,
        data_path=data_path,
        k=0,
        model_name=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=path_a,
        variant_label="A",
    )

    result_b = _generate_variant(
        theme=theme,
        data=data,
        data_path=data_path,
        k=max(args.k, 0),
        model_name=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=path_b,
        variant_label="B",
    )

    len_a = len(result_a["text"])
    len_b = len(result_b["text"])
    duration_a = result_a["duration"]
    duration_b = result_b["duration"]
    print(
        "[orchestrate][A/B] len_A=%d len_B=%d Δlen=%+d duration_A=%.2fs duration_B=%.2fs Δt=%.2fs"
        % (len_a, len_b, len_b - len_a, duration_a, duration_b, duration_b - duration_a)
    )


def _load_batch_config(path: str) -> List[Dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл батча не найден: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Не удалось разобрать YAML. Установите PyYAML или используйте JSON."
            ) from exc
        data = yaml.safe_load(raw)
    if not isinstance(data, list):
        raise ValueError("Батч-файл должен содержать массив заданий.")
    return data


def _resolve_batch_entry(
    entry: Dict[str, Any],
    *,
    default_theme: Optional[str],
    default_mode: str,
    default_k: int,
    default_temperature: float,
    default_max_tokens: int,
    default_timeout: int,
    default_model: Optional[str],
) -> Tuple[str, Dict[str, Any], str, int, Optional[str], str, float, int, int, str]:
    theme = entry.get("theme") or default_theme
    if not theme:
        raise ValueError("Для задания в батче требуется указать theme либо задать его на уровне CLI.")

    data_field = entry.get("data")
    payload_field = entry.get("payload")
    if isinstance(data_field, dict):
        payload = data_field
        data_path = entry.get("data_path") or "<inline>"
    elif isinstance(payload_field, dict):
        payload = payload_field
        data_path = entry.get("data_path") or "<inline>"
    elif isinstance(data_field, str):
        payload = _load_input(data_field)
        data_path = str(Path(data_field).resolve())
    else:
        raise ValueError("Поле data должно быть путем к JSON или объектом с параметрами.")

    outfile = entry.get("outfile")
    mode = entry.get("mode", default_mode)
    k = int(entry.get("k", default_k))
    temperature = float(entry.get("temperature", default_temperature))
    max_tokens = int(entry.get("max_tokens", default_max_tokens))
    timeout = int(entry.get("timeout", default_timeout))
    model_name = _resolve_model(entry.get("model") or default_model)

    return theme, payload, data_path, k, outfile, mode, temperature, max_tokens, timeout, model_name


def _run_batch(args: argparse.Namespace) -> None:
    batch_items = _load_batch_config(args.batch)
    start = time.time()
    report_rows: List[Dict[str, Any]] = []
    successes = 0

    for idx, entry in enumerate(batch_items, start=1):
        try:
            (
                theme,
                payload,
                data_path,
                k,
                outfile,
                mode,
                temperature,
                max_tokens,
                timeout,
                model_name,
            ) = _resolve_batch_entry(
                entry,
                default_theme=args.theme,
                default_mode=args.mode,
                default_k=args.k,
                default_temperature=args.temperature,
                default_max_tokens=args.max_tokens,
                default_timeout=args.timeout,
                default_model=args.model,
            )

            if outfile:
                output_path = Path(outfile)
            else:
                base_path = _make_output_path(theme, None)
                output_path = base_path.with_name(f"{base_path.stem}__{idx:02d}{base_path.suffix}")

            result = _generate_variant(
                theme=theme,
                data=payload,
                data_path=data_path,
                k=k,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                mode=mode,
                output_path=output_path,
            )

            report_rows.append(
                {
                    "index": idx,
                    "theme": theme,
                    "output_path": str(result["output_path"]),
                    "metadata_path": str(result["output_path"].with_suffix(".json")),
                    "characters": len(result["text"]),
                    "duration_seconds": round(result["duration"], 3),
                    "status": "ok",
                }
            )
            successes += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[batch] Ошибка в задании #{idx}: {exc}", file=sys.stderr)
            report_rows.append(
                {
                    "index": idx,
                    "theme": entry.get("theme"),
                    "status": "error",
                    "error": str(exc),
                }
            )

    total_duration = time.time() - start
    print(
        f"[batch] Completed {successes}/{len(batch_items)} items in {total_duration:.2f}s"
    )

    report = {
        "generated_at": _local_now().isoformat(),
        "total": len(batch_items),
        "success": successes,
        "duration_seconds": round(total_duration, 3),
        "results": report_rows,
    }
    report_path = Path("artifacts") / "batch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_checks() -> None:
    ok = True

    python_version = sys.version.split()[0]
    print(f"✅ Python version: {python_version}")

    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY установлен")
    else:
        print("❌ OPENAI_API_KEY не найден")
        ok = False

    artifacts_dir = Path("artifacts")
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        probe = artifacts_dir / ".write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print("✅ Права на запись в artifacts/ подтверждены")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Нет доступа к artifacts/: {exc}")
        ok = False

    index_paths = list(Path("profiles").glob("*/index.json"))
    if index_paths:
        print(f"✅ Найдено {len(index_paths)} индекс(ов) для CONTEXT")
    else:
        print("❌ Индексы CONTEXT не найдены")
        ok = False

    sys.exit(0 if ok else 1)


def main() -> None:
    args = _parse_args()

    if args.check:
        _run_checks()
        return

    if args.batch:
        _run_batch(args)
        return

    if not args.theme or not args.data:
        raise ValueError("Параметры --theme и --data обязательны для одиночного запуска.")

    data = _load_input(args.data)
    data_path = str(Path(args.data).resolve())
    model_name = _resolve_model(args.model)
    base_output_path = _make_output_path(args.theme, args.outfile)

    if args.ab == "compare":
        _run_ab_compare(
            theme=args.theme,
            data=data,
            data_path=data_path,
            model_name=model_name,
            args=args,
            base_output_path=base_output_path,
        )
        return

    result = _generate_variant(
        theme=args.theme,
        data=data,
        data_path=data_path,
        k=args.k,
        model_name=model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        mode=args.mode,
        output_path=base_output_path,
    )
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
