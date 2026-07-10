from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_safe_1000_plan_contains_explicit_dicache_generation_and_metrics() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_jit_safe_1000_eval_plan.py", "--print-only"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    stdout = result.stdout
    for option in (
        "--method dicache_style",
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
        assert option in stdout
    assert "/dicache_style/fid_results.json" in stdout
    assert "/dicache_style/pair_metrics.json" in stdout
    assert "--resume" not in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout
    assert "released-" + "code-" + "compat" not in stdout
