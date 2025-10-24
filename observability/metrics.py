"""Lightweight in-process metrics helpers."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class _BaseMetric:
    name: str
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> float:
        with self._lock:
            return float(self._value)


class Counter(_BaseMetric):
    """Simple monotonically increasing counter."""

    def inc(self, amount: float = 1.0) -> None:
        if amount == 0:
            return
        with self._lock:
            self._value += amount


class Gauge(_BaseMetric):
    """Gauge metric supporting set/add operations."""

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def add(self, amount: float) -> None:
        if amount == 0:
            return
        with self._lock:
            self._value += amount


class MetricsRegistry:
    """Thread-safe registry storing metrics by name."""

    def __init__(self) -> None:
        self._metrics: Dict[str, _BaseMetric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str) -> Counter:
        with self._lock:
            metric = self._metrics.get(name)
            if isinstance(metric, Counter):
                return metric
            counter = Counter(name=name)
            self._metrics[name] = counter
            return counter

    def gauge(self, name: str) -> Gauge:
        with self._lock:
            metric = self._metrics.get(name)
            if isinstance(metric, Gauge):
                return metric
            gauge = Gauge(name=name)
            self._metrics[name] = gauge
            return gauge

    def get(self, name: str) -> Optional[_BaseMetric]:
        return self._metrics.get(name)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {name: metric.snapshot() for name, metric in self._metrics.items()}


_DEFAULT_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _DEFAULT_REGISTRY


__all__ = [
    "Counter",
    "Gauge",
    "MetricsRegistry",
    "get_registry",
]
