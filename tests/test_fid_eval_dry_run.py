from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fid_eval_dry_run_fake_image_dir(tmp_path: Path) -> None:
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    (fake_dir / "000000.png").write_bytes(b"not decoded during dry run")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_stage4a_fid.py",
            "--fake-dir",
            str(fake_dir),
            "--out",
            str(tmp_path / "fid_results.json"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "dry_run" in result.stdout
    assert "num_fake_images" in result.stdout
