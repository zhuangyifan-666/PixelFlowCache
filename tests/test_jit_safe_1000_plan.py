from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_jit_safe_1000_plan_prints_commands_only() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_jit_safe_1000_eval_plan.py", "--print-only", "--gpus", "0,1,2,3"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    stdout = result.stdout
    assert "run_jit_safe_calibration.py" in stdout
    assert "check_safe_map_density.py" in stdout
    assert "make_forced_safe_map.py" in stdout
    assert "run_jit_parallel_generate.py" in stdout
    assert "--execute" in stdout
    assert "--gpus 0,1,2,3" in stdout
    for method in (
        "no_cache_50",
        "safe_bfc_quality",
        "safe_bfc_speed",
        "seacache_style",
        "taylorseer_style",
        "speca_style",
        "dicache_style",
        "reduced_steps_35",
        "reduced_steps_30",
    ):
        assert f"--method {method}" in stdout or f"/{method}/" in stdout
    assert "--taylorseer-interval 4" in stdout
    assert "--taylorseer-max-order 4" in stdout
    assert "--speca-max-order 4" in stdout
    assert "--speca-first-full-steps 3" in stdout
    assert "--speca-base-threshold 0.1" in stdout
    assert "--speca-decay-rate 0.01" in stdout
    assert "--speca-min-threshold 0.01" in stdout
    assert "--speca-min-forecast-steps 2" in stdout
    assert "--speca-max-forecast-steps 5" in stdout
    assert "--speca-error-metric relative_l1" in stdout
    assert "--speca-branch-aggregation mean" in stdout
    assert "--speca-verifier-module auto" in stdout
    assert "--speca-min-history 2" in stdout
    assert "--speca-eps 1e-10" in stdout
    assert "--speca-max-error-samples 4096" in stdout
    assert "--dicache-probe-depth 1" in stdout
    assert "--dicache-reuse-threshold 0.4" in stdout
    assert "--dicache-error-choice delta_y" in stdout
    assert "--dicache-branch-aggregation mean" in stdout
    assert "--dicache-ret-ratio 0.2" in stdout
    assert "--dicache-force-last-step-full" in stdout
    assert "--dicache-dcta" in stdout
    assert "--dicache-gamma-min 1.0" in stdout
    assert "--dicache-gamma-max 1.5" in stdout
    assert "--dicache-eps 1e-10" in stdout
    assert "--dicache-max-stat-samples 4096" in stdout
    assert "--no-dicache-share-cfg-prefix" in stdout
    assert "--dicache-schedule-variant released_flux_compat" in stdout
    assert "released-" + "code-" + "compat" not in stdout
    assert "evaluate_stage4a_fid.py" in stdout
    assert "evaluate_stage4b_pair_metrics.py" in stdout
    assert "collect_jit_safe_1000_results.py" in stdout
    assert "--resume" not in stdout
    assert "torchrun" not in stdout
    assert "accelerate" not in stdout
    assert "nohup" not in stdout

def test_jit_safe_1000_plan_resume_is_opt_in() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_safe_1000_eval_plan.py",
            "--print-only",
            "--gpus",
            "0,1,2,3",
            "--resume",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "--resume" in result.stdout
