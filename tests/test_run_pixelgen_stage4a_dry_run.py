from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pixelgen_stage4a_dry_run_does_not_require_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pixelgen_stage4a_generate.py",
            "--method",
            "bfc_quality_t02_08",
            "--num-images",
            "8",
            "--batch-size",
            "2",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "dryrun_pixelgen",
            "--pixelgen-ckpt",
            str(tmp_path / "missing.ckpt"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert '"checkpoint_exists": false' in result.stdout
    assert '"model_name": "PixelGen"' in result.stdout
    assert "pixelgen_runtime" not in result.stderr
    assert not (tmp_path / "outputs").exists()


def test_pixelgen_seacache_dry_run_reports_dynamic_meta_without_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pixelgen_stage4a_generate.py",
            "--method",
            "seacache_style",
            "--num-images",
            "8",
            "--batch-size",
            "2",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "dryrun_pixelgen_seacache",
            "--pixelgen-ckpt",
            str(tmp_path / "missing.ckpt"),
            "--amp-dtype",
            "bf16",
            "--dynamic-cache-threshold",
            "0.06",
            "--sea-beta",
            "2.0",
            "--sea-proxy-downsample",
            "64",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert '"checkpoint_exists": false' in result.stdout
    assert '"method_name": "seacache_style"' in result.stdout
    assert '"dynamic_cache_type": "sea"' in result.stdout
    assert '"resolved_dynamic_cache_threshold": 0.06' in result.stdout
    assert '"sea_beta": 2.0' in result.stdout
    assert '"sea_proxy_downsample": 64' in result.stdout
    assert "pixelgen_runtime" not in result.stderr
    assert not (tmp_path / "outputs").exists()
