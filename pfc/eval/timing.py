from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator


@dataclass
class GenerationTiming:
    """Schema-v2 generation timing with sampling isolated from I/O and setup."""

    schema_version: int = 2
    model_load_latency_sec: float = 0.0
    warmup_latency_sec: float = 0.0
    input_prepare_latency_sec: float = 0.0
    sampling_latency_sec: float = 0.0
    postprocess_latency_sec: float = 0.0
    png_save_latency_sec: float = 0.0
    npz_save_latency_sec: float = 0.0
    manifest_latency_sec: float = 0.0
    end_to_end_latency_sec: float = 0.0
    requested_images: int = 0
    generated_images_this_run: int = 0
    existing_images_skipped: int = 0
    total_images_available: int = 0
    resume: bool = False
    num_shards: int = 1
    shard_index: int = 0
    timing_scope: str = "synchronized_single_gpu_sampling"
    comparable_for_algorithm_speedup: bool = True
    peak_memory_allocated_bytes: int = 0

    def add(self, field_name: str, duration_sec: float) -> None:
        if not hasattr(self, field_name) or not field_name.endswith("_latency_sec"):
            raise ValueError(f"unsupported timing field: {field_name}")
        setattr(self, field_name, float(getattr(self, field_name)) + max(float(duration_sec), 0.0))

    @contextmanager
    def measure(
        self,
        field_name: str,
        *,
        device: Any | None = None,
        synchronize: bool = False,
    ) -> Iterator[None]:
        if synchronize:
            synchronize_device(device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                synchronize_device(device)
            self.add(field_name, time.perf_counter() - started)

    def finalize_comparability(self) -> None:
        if self.resume or self.existing_images_skipped > 0 or self.num_shards > 1:
            self.comparable_for_algorithm_speedup = False

    def to_dict(self) -> dict[str, Any]:
        self.finalize_comparability()
        payload = asdict(self)
        payload["timing_schema_version"] = payload.pop("schema_version")
        payload["sampling_images_per_sec"] = _rate(
            self.generated_images_this_run,
            self.sampling_latency_sec,
        )
        payload["end_to_end_images_per_sec"] = _rate(
            self.generated_images_this_run,
            self.end_to_end_latency_sec,
        )
        payload["legacy_timing"] = False
        return payload


def synchronize_device(device: Any | None) -> None:
    """Synchronize only an actual CUDA device; CPU tests remain CUDA-free."""

    if device is None or getattr(device, "type", str(device).split(":", 1)[0]) != "cuda":
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def normalize_timing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Read schema-v2 timing or conservatively expose a legacy record."""

    if int(payload.get("timing_schema_version", 0) or 0) >= 2:
        normalized = dict(payload)
        normalized.setdefault("legacy_timing", False)
        return normalized
    legacy = dict(payload)
    legacy_latency = legacy.get("latency_sec")
    legacy.update(
        {
            "timing_schema_version": 1,
            "sampling_latency_sec": None,
            "end_to_end_latency_sec": legacy_latency,
            "timing_scope": "legacy_unspecified",
            "legacy_timing": True,
            "comparable_for_algorithm_speedup": False,
        }
    )
    return legacy


def _rate(count: int, duration_sec: float) -> float | None:
    return float(count) / float(duration_sec) if duration_sec > 0.0 else None
