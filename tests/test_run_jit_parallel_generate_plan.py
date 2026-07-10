from __future__ import annotations

import subprocess
import sys
import shlex
from pathlib import Path

from scripts.run_jit_parallel_generate import make_shard_debug_path


ROOT = Path(__file__).resolve().parents[1]


def test_run_jit_parallel_generate_print_only_outputs_workers_and_merge(tmp_path: Path) -> None:
    safe_map = tmp_path / "safe.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_parallel_generate.py",
            "--method",
            "safe_bfc_speed",
            "--run-id",
            "unit",
            "--safe-map",
            str(safe_map),
            "--gpus",
            "0,1,2,3",
            "--num-shards",
            "4",
            "--dynamic-cache-threshold",
            "0.06",
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    stdout = result.stdout
    for idx in range(4):
        assert f"CUDA_VISIBLE_DEVICES={idx}" in stdout
        assert f"--shard-index {idx}" in stdout
    assert "--num-shards 4" in stdout
    assert f"--safe-map {shlex.quote(str(safe_map))}" in stdout
    assert "--dynamic-cache-threshold 0.06" in stdout
    assert "merge_jit_parallel_shards.py" in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout



def test_run_jit_parallel_generate_print_only_passes_taylorseer_args(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_parallel_generate.py",
            "--method",
            "taylorseer_style",
            "--run-id",
            "unit_taylor",
            "--gpus",
            "0,1,2,3",
            "--num-shards",
            "4",
            "--taylorseer-interval",
            "4",
            "--taylorseer-max-order",
            "4",
            "--taylorseer-debug-jsonl",
            str(tmp_path / "debug.jsonl"),
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    stdout = result.stdout
    assert "--method taylorseer_style" in stdout
    assert "--taylorseer-interval 4" in stdout
    assert "--taylorseer-max-order 4" in stdout
    assert "--taylorseer-debug-jsonl" in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout
    assert "--speca-clone-forecast" not in stdout


def test_run_jit_parallel_generate_print_only_passes_speca_args(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_parallel_generate.py",
            "--method",
            "speca_style",
            "--run-id",
            "unit_speca",
            "--gpus",
            "0,1,2,3",
            "--num-shards",
            "4",
            "--speca-max-order",
            "4",
            "--speca-first-full-steps",
            "3",
            "--speca-base-threshold",
            "0.1",
            "--speca-decay-rate",
            "0.01",
            "--speca-min-threshold",
            "0.01",
            "--speca-min-forecast-steps",
            "2",
            "--speca-max-forecast-steps",
            "5",
            "--speca-error-metric",
            "relative_l1",
            "--speca-branch-aggregation",
            "mean",
            "--speca-verifier-module",
            "auto",
            "--speca-min-history",
            "2",
            "--speca-debug-jsonl",
            str(tmp_path / "speca.jsonl"),
            "--taylorseer-debug-jsonl",
            str(tmp_path / "taylor.jsonl"),
            "--safe-debug-jsonl",
            str(tmp_path / "safe.jsonl"),
            "--dynamic-cache-debug-jsonl",
            str(tmp_path / "dynamic.jsonl"),
            "--speca-clone-forecast",
            "--speca-eps",
            "1e-9",
            "--speca-max-error-samples",
            "128",
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    stdout = result.stdout
    for expected in (
        "--method speca_style",
        "--speca-max-order 4",
        "--speca-first-full-steps 3",
        "--speca-base-threshold 0.1",
        "--speca-decay-rate 0.01",
        "--speca-min-threshold 0.01",
        "--speca-min-forecast-steps 2",
        "--speca-max-forecast-steps 5",
        "--speca-error-metric relative_l1",
        "--speca-branch-aggregation mean",
        "--speca-verifier-module auto",
        "--speca-min-history 2",
        "--speca-debug-jsonl",
        "--speca-eps 1e-09",
        "--speca-max-error-samples 128",
        "--speca-clone-forecast",
    ):
        assert expected in stdout
    assert "--resume" not in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout
    worker_lines = [line for line in stdout.splitlines() if "run_jit_stage4a_generate.py" in line]
    assert len(worker_lines) == 4
    for idx, line in enumerate(worker_lines):
        assert f"speca_shard{idx}.jsonl" in line
        assert f"taylor_shard{idx}.jsonl" in line
        assert f"safe_shard{idx}.jsonl" in line
        assert f"dynamic_shard{idx}.jsonl" in line


def test_make_shard_debug_path_handles_none_suffix_and_single_shard(tmp_path: Path) -> None:
    path = tmp_path / "debug.jsonl"
    assert make_shard_debug_path(None, 0, 4) is None
    assert make_shard_debug_path(path, 0, 1) == path
    assert make_shard_debug_path(path, 3, 4) == tmp_path / "debug_shard3.jsonl"


def test_speca_parallel_defaults_pass_bounded_stats_without_clone_flag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_parallel_generate.py",
            "--method",
            "speca_style",
            "--run-id",
            "unit_speca_defaults",
            "--gpus",
            "0",
            "--num-shards",
            "1",
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "--speca-eps 1e-10" in result.stdout
    assert "--speca-max-error-samples 4096" in result.stdout
    assert "--speca-clone-forecast" not in result.stdout


def test_run_jit_parallel_generate_passes_dicache_args_and_shards_debug(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_parallel_generate.py",
            "--method",
            "dicache_style",
            "--run-id",
            "unit_dicache",
            "--gpus",
            "0,1,2,3",
            "--num-shards",
            "4",
            "--dicache-probe-depth",
            "1",
            "--dicache-reuse-threshold",
            "0.4",
            "--dicache-error-choice",
            "delta_y",
            "--dicache-branch-aggregation",
            "mean",
            "--dicache-ret-ratio",
            "0.2",
            "--dicache-force-last-step-full",
            "--dicache-dcta",
            "--dicache-gamma-min",
            "1.0",
            "--dicache-gamma-max",
            "1.5",
            "--dicache-eps",
            "1e-10",
            "--dicache-max-stat-samples",
            "4096",
            "--no-dicache-share-cfg-prefix",
            "--dicache-schedule-variant",
            "released_flux_compat",
            "--dicache-debug-jsonl",
            str(tmp_path / "dicache.jsonl"),
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    worker_lines = [line for line in result.stdout.splitlines() if "run_jit_stage4a_generate.py" in line]
    assert len(worker_lines) == 4
    for idx, line in enumerate(worker_lines):
        assert f"dicache_shard{idx}.jsonl" in line
        for expected in (
            "--dicache-probe-depth 1",
            "--dicache-reuse-threshold 0.4",
            "--dicache-error-choice delta_y",
            "--dicache-branch-aggregation mean",
            "--dicache-ret-ratio 0.2",
            "--dicache-force-last-step-full",
            "--dicache-dcta",
            "--dicache-gamma-min 1.0",
            "--dicache-gamma-max 1.5",
            "--dicache-eps 1e-10",
            "--dicache-max-stat-samples 4096",
            "--no-dicache-share-cfg-prefix",
            "--dicache-schedule-variant released_flux_compat",
        ):
            assert expected in line
    assert "--resume" not in result.stdout
    assert "torchrun" not in result.stdout
    assert "accelerate" not in result.stdout
    assert "nohup" not in result.stdout
    assert "released-" + "code-" + "compat" not in result.stdout
