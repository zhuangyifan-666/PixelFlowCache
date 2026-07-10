from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch


def compact_history_tensor(
    value: torch.Tensor,
    clone_history: bool = False,
) -> torch.Tensor:
    """Detach history and avoid retaining a larger backing storage through a view."""

    detached = value.detach()
    if (
        clone_history
        or detached.storage_offset() != 0
        or not detached.is_contiguous()
        or getattr(detached, "_base", None) is not None
    ):
        return detached.clone()
    return detached


def detached_history_tensor(value: torch.Tensor, *, clone: bool) -> torch.Tensor:
    """Backward-compatible spelling for callers outside the DiCache runtime."""

    return compact_history_tensor(value, clone_history=clone)


@dataclass
class RunningStats:
    """Exact running moments plus bounded samples for approximate quantiles."""

    max_samples: int = 4096
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    samples: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.samples = deque(maxlen=self.max_samples)

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("RunningStats only accepts finite values")
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.samples.append(value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 0.0
        variance = max(self.total_sq / self.count - self.mean * self.mean, 0.0)
        return math.sqrt(variance)

    def quantile(self, probability: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        position = (len(ordered) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def summary(self) -> dict[str, Any]:
        bounded_samples = list(self.samples)
        return {
            "count": self.count,
            "sum": self.total,
            "sum_sq": self.total_sq,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "p50": self.quantile(0.50),
            "p90": self.quantile(0.90),
            "p95": self.quantile(0.95),
            "sample_count": len(self.samples),
            "max_samples": self.max_samples,
            "quantiles_approximate": True,
            "bounded_samples": bounded_samples,
            "samples_bounded": bounded_samples,
        }


@dataclass
class DiCacheBranchHistory:
    previous_probe_feature: torch.Tensor | None = None
    full_residual_history: deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=2))
    probe_residual_history: deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=2))
    refresh_step_history: deque[int] = field(default_factory=lambda: deque(maxlen=2))

    def clear(self) -> None:
        self.previous_probe_feature = None
        self.full_residual_history.clear()
        self.probe_residual_history.clear()
        self.refresh_step_history.clear()


@dataclass
class DiCacheRuntimeState:
    enabled: bool
    total_steps: int
    current_step_idx: int | None = None
    accumulated_error: float = 0.0
    previous_input_feature: torch.Tensor | None = None
    branch_histories: dict[str, DiCacheBranchHistory] = field(
        default_factory=lambda: {
            "cond": DiCacheBranchHistory(),
            "uncond": DiCacheBranchHistory(),
        }
    )

    def clear_batch(self) -> None:
        self.current_step_idx = None
        self.accumulated_error = 0.0
        self.previous_input_feature = None
        for history in self.branch_histories.values():
            history.clear()
