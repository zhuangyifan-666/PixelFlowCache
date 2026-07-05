from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_jit_safe_calibration_dry_run_does_not_require_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_safe_calibration.py",
            "--num-calibration-images",
            "8",
            "--batch-size",
            "2",
            "--run-id",
            "dryrun_jit_safe_calib",
            "--out-dir",
            str(tmp_path / "calib"),
            "--jit-ckpt-dir",
            str(tmp_path / "missing_ckpt"),
            "--max-age",
            "3",
            "--quantile",
            "0.95",
            "--quality-lambda",
            "0.5",
            "--speed-lambda",
            "1.0",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "No checkpoint is loaded" in result.stdout
    assert "safe_map_quality.json" in result.stdout
    assert '"num_calibration_images": 8' in result.stdout
    assert '"max_age": 3' in result.stdout
    assert '"quantile": 0.95' in result.stdout
    assert '"quality_lambda": 0.5' in result.stdout
    assert '"speed_lambda": 1.0' in result.stdout
