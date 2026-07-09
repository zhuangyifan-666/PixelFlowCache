from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    )
    stdout = result.stdout
    for idx in range(4):
        assert f"CUDA_VISIBLE_DEVICES={idx}" in stdout
        assert f"--shard-index {idx}" in stdout
    assert "--num-shards 4" in stdout
    assert f"--safe-map {safe_map}" in stdout
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
    )
    stdout = result.stdout
    assert "--method taylorseer_style" in stdout
    assert "--taylorseer-interval 4" in stdout
    assert "--taylorseer-max-order 4" in stdout
    assert "--taylorseer-debug-jsonl" in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout
