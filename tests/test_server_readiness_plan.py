from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readiness_planner_prints_all_gates_without_resume_or_execution(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_server_readiness_plan.py",
            "--models", "jit",
            "--methods", "no_cache_50,seacache_style,dicache_style",
            "--run-id", "test_readiness",
            "--gpus", "0,1,2,3",
            "--skip-safe-calibration",
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    for gate in range(8):
        assert f"Gate {gate}" in result.stdout
    assert "--expected-images 1000" in result.stdout
    assert "--proxy-result" in result.stdout
    assert "--resume" not in result.stdout


def test_single_gpu_timing_plan_defaults_are_print_only() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_jit_single_gpu_timing_plan.py", "--print-only"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "--num-images 64" in result.stdout
    assert "--batch-size 8" in result.stdout
    assert "--warmup-batches 2" in result.stdout
    assert result.stdout.count("--method no_cache_50") == 3
    assert "--no-save-png" in result.stdout
    assert "--no-save-npz" in result.stdout
    assert "--resume" not in result.stdout


def test_readiness_plan_routes_env_batches_safe_maps_and_all_jit_pairs(tmp_path: Path) -> None:
    methods = (
        "no_cache_50,safe_bfc_quality,safe_bfc_speed,seacache_style,"
        "taylorseer_style,speca_style,dicache_style,reduced_steps_35,reduced_steps_30"
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_server_readiness_plan.py",
            "--models", "jit,deco,pixelgen",
            "--methods", methods,
            "--run-id", "ready",
            "--output-root", "outputs/ready",
            "--gpus", "0,1,2,3",
            "--env-jit", "jit-custom",
            "--env-deco", "deco-custom",
            "--env-pixelgen", "pixelgen-custom",
            "--safe-map-quality", "quality.json",
            "--safe-map-speed", "speed.json",
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    output = result.stdout
    assert "conda run -n jit-custom python scripts/run_jit_stage4a_generate.py" in output
    assert "conda run -n deco-custom python scripts/run_deco_stage4a_generate.py" in output
    assert "conda run -n pixelgen-custom python scripts/run_pixelgen_stage4a_generate.py" in output
    pixelgen_smoke = next(
        line for line in output.splitlines()
        if "run_pixelgen_stage4a_generate.py" in line and "--method no_cache_50" in line
    )
    assert "--batch-size 4" in pixelgen_smoke
    assert "--safe-map quality.json" in next(
        line for line in output.splitlines() if "--method safe_bfc_quality" in line
    )
    assert "--safe-map speed.json" in next(
        line for line in output.splitlines() if "--method safe_bfc_speed" in line
    )
    assert "Expected equivalence: PSNR inf; SSIM 1; LPIPS 0; rel_l2 0" in output
    assert "--safe-map-quality quality.json" in output
    assert "--safe-map-speed speed.json" in output
    assert "conda run -n jit-custom python scripts/run_parallel_generate.py" not in output
    assert "python scripts/run_parallel_generate.py" in output
    assert "--expected-images 1000" in output
    assert "--proxy-result" in output
    for method in (
        "safe_bfc_quality",
        "safe_bfc_speed",
        "seacache_style",
        "taylorseer_style",
        "speca_style",
        "dicache_style",
        "reduced_steps_35",
        "reduced_steps_30",
    ):
        assert f"pair_metrics/jit/{method}/pair_metrics.json" in output
    assert "deco: no generic paired-metric collector" in output
    assert "pixelgen: no generic paired-metric collector" in output
    assert "--resume" not in output


def test_readiness_plan_requires_selected_safe_map() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_server_readiness_plan.py",
            "--models", "jit",
            "--methods", "no_cache_50,safe_bfc_quality",
            "--print-only",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "require safe maps" in result.stderr
