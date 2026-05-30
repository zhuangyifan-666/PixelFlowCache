from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def measure_latency_ms(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    repeat: int = 10,
    use_cuda_events: bool = True,
    **kwargs: Any,
) -> dict[str, float | int | bool]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    for _ in range(max(0, warmup)):
        fn(*args, **kwargs)

    cuda_timing = bool(use_cuda_events and torch.cuda.is_available())
    times_ms: list[float] = []

    if cuda_timing:
        torch.cuda.synchronize()
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(repeat):
            start = time.perf_counter()
            fn(*args, **kwargs)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return {
        "mean_ms": float(statistics.fmean(times_ms)),
        "std_ms": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "repeat": repeat,
        "warmup": warmup,
        "cuda_events": cuda_timing,
    }

