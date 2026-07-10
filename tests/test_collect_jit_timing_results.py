from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.collect_jit_timing_results as collector


def _meta() -> dict:
    return {
        "model_name": "JiT",
        "checkpoint_path": "ckpts/JiT/checkpoint-last.pth",
        "eval_steps": 50,
        "reference_steps": 50,
        "batch_size": 8,
        "num_images": 64,
        "dtype": "float32",
        "amp_enabled": False,
        "autocast_enabled": False,
        "compile_enabled": False,
        "device_type": "cuda",
        "warmup_batches": 2,
        "seed": 0,
        "cfg": 3.0,
        "cfg_interval": [0.1, 1.0],
        "sampler": "euler",
        "solver": "euler",
        "img_size": 256,
        "num_shards": 1,
        "resume": False,
        "save_png": False,
        "save_npz": False,
        "provenance": {
            "gpu_count": 1,
            "gpu_names": ["Test GPU"],
            "checkpoint": {
                "path": "ckpts/JiT/checkpoint-last.pth",
                "size": 1234,
                "sha256": "abc",
            },
        },
    }


def _timing(latency: float) -> dict:
    return {
        "timing_schema_version": 2,
        "timing_scope": "synchronized_single_gpu_sampling",
        "comparable_for_algorithm_speedup": True,
        "sampling_latency_sec": latency,
        "sampling_images_per_sec": 64 / latency,
        "peak_memory_allocated_bytes": 100,
        "requested_images": 64,
        "num_shards": 1,
        "resume": False,
    }


def _write_repeat(
    root: Path,
    run_id: str,
    repeat: int,
    method: str,
    latency: float,
    *,
    meta: dict | None = None,
    timing: dict | None = None,
) -> None:
    run_dir = root / "jit" / f"{run_id}_r{repeat:02d}" / method
    run_dir.mkdir(parents=True)
    (run_dir / "generation_meta.json").write_text(
        json.dumps(meta or _meta()), encoding="utf-8"
    )
    (run_dir / "latency.json").write_text(
        json.dumps(timing or _timing(latency)), encoding="utf-8"
    )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repeats: int = 1) -> dict:
    out = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_jit_timing_results.py",
            "--run-id", "timing",
            "--output-root", str(tmp_path / "runs"),
            "--out-dir", str(out),
            "--methods", "no_cache_50,seacache_style",
            "--repeats", str(repeats),
        ],
    )
    assert collector.main() == 0
    return json.loads((out / "jit_timing_summary.json").read_text(encoding="utf-8"))


def test_matched_comparison_reports_mean_std_median_and_speedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    for repeat, baseline_latency, method_latency in (
        (1, 4.0, 2.0),
        (2, 6.0, 3.0),
        (3, 5.0, 2.5),
    ):
        _write_repeat(root, "timing", repeat, "no_cache_50", baseline_latency)
        _write_repeat(root, "timing", repeat, "seacache_style", method_latency)

    payload = _run(tmp_path, monkeypatch, repeats=3)
    baseline, method = payload["rows"]
    assert baseline["mean_sampling_latency"] == pytest.approx(5.0)
    assert baseline["std_sampling_latency"] == pytest.approx(0.8164965809)
    assert baseline["median_sampling_latency"] == pytest.approx(5.0)
    assert method["speedup_vs_no_cache"] == pytest.approx(2.0)
    assert method["comparable"] is True


@pytest.mark.parametrize(
    "mismatch",
    ["gpu_name", "dtype", "compile", "steps", "checkpoint", "resume", "sharded"],
)
def test_signature_mismatch_disables_speedup_with_detailed_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    root = tmp_path / "runs"
    _write_repeat(root, "timing", 1, "no_cache_50", 4.0)
    meta = _meta()
    timing = _timing(2.0)
    expected_field = mismatch
    if mismatch == "gpu_name":
        meta["provenance"]["gpu_names"] = ["Other GPU"]
    elif mismatch == "dtype":
        meta["dtype"] = "float16"
    elif mismatch == "compile":
        meta["compile_enabled"] = True
        expected_field = "compile_enabled"
    elif mismatch == "steps":
        meta["eval_steps"] = 35
        expected_field = "eval_steps"
    elif mismatch == "checkpoint":
        meta["checkpoint_path"] = "other/checkpoint.pth"
        expected_field = "checkpoint_path"
    elif mismatch == "resume":
        meta["resume"] = True
        timing["resume"] = True
        timing["comparable_for_algorithm_speedup"] = False
    elif mismatch == "sharded":
        meta["num_shards"] = 4
        timing["num_shards"] = 4
        timing["comparable_for_algorithm_speedup"] = False
        expected_field = "num_shards"
    _write_repeat(root, "timing", 1, "seacache_style", 2.0, meta=meta, timing=timing)

    payload = _run(tmp_path, monkeypatch)
    method = payload["rows"][1]
    assert method["speedup_vs_no_cache"] is None
    assert method["comparable"] is False
    serialized = json.dumps(method["signature_diff"], sort_keys=True)
    assert expected_field in serialized
