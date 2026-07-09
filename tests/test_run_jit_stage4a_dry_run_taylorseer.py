from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_jit_stage4a_taylorseer_dry_run_prints_config_without_sampling(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "taylorseer_style",
            "--num-images",
            "8",
            "--batch-size",
            "2",
            "--run-id",
            "dryrun_jit_taylorseer",
            "--output-root",
            str(tmp_path / "outputs"),
            "--jit-ckpt-dir",
            str(tmp_path / "missing_ckpt"),
            "--taylorseer-interval",
            "4",
            "--taylorseer-max-order",
            "4",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "taylorseer_style" in result.stdout
    assert '"method_type": "forecast_cache"' in result.stdout
    assert '"taylorseer_interval": 4' in result.stdout
    assert '"taylorseer_max_order": 4' in result.stdout
    assert "TaylorSeer-style" in result.stdout
    assert "Missing JiT checkpoint" in result.stdout
    assert "Traceback" not in result.stderr
