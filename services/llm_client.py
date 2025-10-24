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

from config import OPENAI_MAX_RETRIES, OPENAI_RPM, OPENAI_RPS, OPENAI_TIMEOUT_S
from llm_client import GenerationResult, generate as _legacy_generate


@dataclass
class RetryPolicy:
    max_retries: int = OPENAI_MAX_RETRIES
    base_delay: float = 0.4
    jitter: float = 0.3


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
        self._cache: Dict[str, GenerationResult] = {}
        self._lock = threading.Lock()

    def _make_key(
        self,
        *,
        system: str,
        messages: Sequence[Dict[str, str]],
        seed: Optional[str],
        structure: Optional[Iterable[str]],
    ) -> str:
        payload = {
            "system": system or "",
            "messages": list(messages),
            "seed": seed or "",
            "structure": list(structure) if structure else [],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

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
        key = self._make_key(system=system, messages=messages, seed=seed, structure=structure)
        with self._lock:
            cached = self._cache.get(key)
        if cached:
            return cached

        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(dict(message) for message in messages)

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts <= policy.max_retries:
            attempts += 1
            self._limiter.acquire()
            try:
                result = _legacy_generate(
                    full_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout or OPENAI_TIMEOUT_S,
                )
                with self._lock:
                    self._cache[key] = result
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempts > policy.max_retries:
                    break
                delay = policy.base_delay * (2 ** (attempts - 1))
                jitter = random.random() * policy.jitter
                time.sleep(delay + jitter)
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI client failed without raising an error")


_default_client = OpenAIClient()


def get_default_client() -> OpenAIClient:
    return _default_client


__all__ = ["OpenAIClient", "RetryPolicy", "get_default_client"]
