from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_jit_stage4a_generate import load_jit_runtime_helpers


ROOT = Path(__file__).resolve().parents[1]


def test_stage4a_plan_prints_generation_and_fid_commands(tmp_path: Path) -> None:
    out_script = tmp_path / "stage4a_plan.sh"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage4a_full_eval_plan.py",
            "--models",
            "jit",
            "--num-images",
            "100",
            "--out-script",
            str(out_script),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "run_jit_stage4a_generate.py" in result.stdout
    assert "evaluate_stage4a_fid.py" in result.stdout
    assert "conda run -n jit" in result.stdout
    assert out_script.exists()
    assert "Review and run commands manually" in out_script.read_text(encoding="utf-8")


def test_stage4a_jit_runtime_helper_imports_sampler_from_stage2_cache() -> None:
    _config_cls, _load_model, sample_jit = load_jit_runtime_helpers()
    assert sample_jit.__module__ == "scripts.run_jit_stage2_cache"


def test_stage4a_jit_dry_run_serializes_paths(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "no_cache_50",
            "--num-images",
            "100",
            "--output-root",
            str(tmp_path / "outputs"),
            "--jit-ckpt-dir",
            str(tmp_path / "missing_ckpt"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "checkpoint_exists" in result.stdout
    assert "Missing JiT checkpoint" in result.stdout
    assert "TypeError" not in result.stderr
