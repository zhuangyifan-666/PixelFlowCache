from __future__ import annotations

import time

import pytest
import torch

from pfc.eval.timing import GenerationTiming, normalize_timing_payload, synchronize_device


def test_generation_timing_separates_rates_and_is_cpu_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: pytest.fail("CUDA sync on CPU"))
    timing = GenerationTiming(requested_images=8, generated_images_this_run=8)
    with timing.measure("sampling_latency_sec", device=torch.device("cpu"), synchronize=True):
        time.sleep(0.001)
    timing.end_to_end_latency_sec = timing.sampling_latency_sec + 1.0
    payload = timing.to_dict()
    assert payload["timing_schema_version"] == 2
    assert payload["sampling_images_per_sec"] > payload["end_to_end_images_per_sec"]
    assert payload["comparable_for_algorithm_speedup"] is True


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"resume": True}, False),
        ({"existing_images_skipped": 1}, False),
        ({"num_shards": 4}, False),
        ({}, True),
    ],
)
def test_generation_timing_comparability(kwargs: dict[str, object], expected: bool) -> None:
    assert GenerationTiming(**kwargs).to_dict()["comparable_for_algorithm_speedup"] is expected


def test_legacy_timing_never_becomes_sampling_timing() -> None:
    payload = normalize_timing_payload({"latency_sec": 12.0, "images_per_sec": 3.0})
    assert payload["legacy_timing"] is True
    assert payload["sampling_latency_sec"] is None
    assert payload["end_to_end_latency_sec"] == 12.0
    assert payload["comparable_for_algorithm_speedup"] is False


def test_synchronize_device_ignores_none_and_cpu() -> None:
    synchronize_device(None)
    synchronize_device(torch.device("cpu"))
