"""Unified OpenAI client facade with rate limiting and idempotency."""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from config import (
    G5_MAX_OUTPUT_TOKENS_MAX,
    OPENAI_CACHE_TTL_S,
    OPENAI_CLIENT_MAX_QUEUE,
    OPENAI_MAX_RETRIES,
    OPENAI_RPM,
    OPENAI_RPS,
    OPENAI_TIMEOUT_S,
)
from llm_client import GenerationResult, generate as _legacy_generate
from observability.logger import get_logger


@dataclass
class RetryPolicy:
    max_retries: int = OPENAI_MAX_RETRIES
    base_delay: float = 0.4
    jitter: float = 0.3


@dataclass
class _CacheEntry:
    result: GenerationResult
    expires_at: float


class _RateLimiter:
    def __init__(self, *, rps: int, rpm: int) -> None:
        self._rps = max(1, rps)
        self._rpm = max(self._rps, rpm)
        self._per_second: deque[float] = deque()
        self._per_minute: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._trim(now)
                if len(self._per_second) < self._rps and len(self._per_minute) < self._rpm:
                    self._per_second.append(now)
                    self._per_minute.append(now)
                    return
                wait_options: List[float] = []
                if self._per_second:
                    wait_options.append(1.0 - (now - self._per_second[0]))
                if self._per_minute:
                    wait_options.append(60.0 - (now - self._per_minute[0]))
            delay = max(0.05, min(wait_options) if wait_options else 0.05)
            time.sleep(delay)

    def _trim(self, now: float) -> None:
        while self._per_second and now - self._per_second[0] >= 1.0:
            self._per_second.popleft()
        while self._per_minute and now - self._per_minute[0] >= 60.0:
            self._per_minute.popleft()


class OpenAIClient:
    """High-level facade delegating to the legacy generator with safeguards."""

    def __init__(self) -> None:
        self._limiter = _RateLimiter(rps=OPENAI_RPS, rpm=OPENAI_RPM)
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._pending_lock = threading.Condition()
        self._pending_requests = 0

    def _make_key(
        self,
        *,
        system: str,
        messages: Sequence[Dict[str, str]],
        seed: Optional[str],
        structure: Optional[Iterable[str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: str,
    ) -> str:
        payload = {
            "system": system or "",
            "messages": list(messages),
            "seed": seed or "",
            "structure": list(structure) if structure else [],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[GenerationResult]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if entry.expires_at < time.time():
                self._cache.pop(key, None)
                return None
            return entry.result

    def _set_cached(self, key: str, result: GenerationResult) -> None:
        expires_at = time.time() + OPENAI_CACHE_TTL_S
        with self._lock:
            self._cache[key] = _CacheEntry(result=result, expires_at=expires_at)

    def _acquire_slot(self) -> None:
        with self._pending_lock:
            while self._pending_requests >= OPENAI_CLIENT_MAX_QUEUE:
                self._pending_lock.wait()
            self._pending_requests += 1

    def _release_slot(self) -> None:
        with self._pending_lock:
            self._pending_requests = max(0, self._pending_requests - 1)
            self._pending_lock.notify()

    def generate(
        self,
        *,
        system: str,
        messages: Sequence[Dict[str, str]],
        model: str = "gpt-5",
        response_format: str = "text",
        timeout: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        seed: Optional[str] = None,
        structure: Optional[Iterable[str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1400,
    ) -> GenerationResult:
        policy = retry_policy or RetryPolicy()
        key = self._make_key(
            system=system,
            messages=messages,
            seed=seed,
            structure=structure,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        cached = self._get_cached(key)
        if cached:
            LOGGER.info("llm_cache_hit", extra={"model": model, "max_tokens": max_tokens})
            return cached

        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(dict(message) for message in messages)

        resolved_max_tokens = min(max_tokens, G5_MAX_OUTPUT_TOKENS_MAX)
        if resolved_max_tokens != max_tokens:
            LOGGER.warning(
                "llm_max_tokens_capped",
                extra={
                    "requested": max_tokens,
                    "cap": G5_MAX_OUTPUT_TOKENS_MAX,
                    "resolved": resolved_max_tokens,
                },
            )

        attempts = 0
        last_error: Optional[Exception] = None
        self._acquire_slot()
        try:
            while attempts <= policy.max_retries:
                attempts += 1
                self._limiter.acquire()
                started_at = time.perf_counter()
                try:
                    result = _legacy_generate(
                        full_messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=resolved_max_tokens,
                        timeout_s=timeout or OPENAI_TIMEOUT_S,
                    )
                    duration_ms = int((time.perf_counter() - started_at) * 1000)
                    LOGGER.info(
                        "llm_request_succeeded",
                        extra={
                            "model": model,
                            "attempt": attempts,
                            "duration_ms": duration_ms,
                            "max_tokens": resolved_max_tokens,
                        },
                    )
                    self._set_cached(key, result)
                    return result
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    LOGGER.warning(
                        "llm_request_failed",
                        extra={
                            "model": model,
                            "attempt": attempts,
                            "error": str(exc),
                        },
                    )
                    if attempts > policy.max_retries:
                        break
                    delay = policy.base_delay * (2 ** (attempts - 1))
                    jitter = random.random() * policy.jitter
                    time.sleep(delay + jitter)
        finally:
            self._release_slot()
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI client failed without raising an error")


_default_client = OpenAIClient()


def get_default_client() -> OpenAIClient:
    return _default_client


LOGGER = get_logger("content_factory.services.llm_client")


__all__ = ["OpenAIClient", "RetryPolicy", "get_default_client"]
