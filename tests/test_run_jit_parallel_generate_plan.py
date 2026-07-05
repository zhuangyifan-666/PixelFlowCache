from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_jit_parallel_generate_print_only_outputs_workers_and_merge() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_parallel_generate.py",
            "--method",
            "safe_bfc_speed",
            "--run-id",
            "unit",
            "--safe-map",
            "/tmp/safe.json",
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
    assert "--safe-map /tmp/safe.json" in stdout
    assert "--dynamic-cache-threshold 0.06" in stdout
    assert "merge_jit_parallel_shards.py" in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout
