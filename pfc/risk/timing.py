from __future__ import annotations

import time
from typing import Any


class DiagnosticActionTimer:
    latency_semantics = "isolated_counterfactual_diagnostic"
    not_comparable_to_end_to_end_generation = True

    def __init__(self, device: Any) -> None:
        self.device = device
        self._cpu_started: float | None = None
        self._start_event: Any | None = None
        self._end_event: Any | None = None

    def start(self) -> None:
        if getattr(self.device, "type", str(self.device).split(":", 1)[0]) == "cuda":
            import torch

            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_started = time.perf_counter()

    def stop(self) -> float:
        if self._start_event is not None:
            self._end_event.record()
            self._end_event.synchronize()
            return float(self._start_event.elapsed_time(self._end_event))
        if self._cpu_started is None:
            raise RuntimeError("diagnostic timer was not started")
        return (time.perf_counter() - self._cpu_started) * 1000.0
