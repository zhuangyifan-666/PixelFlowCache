from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_jit_dry_run_includes_pixbfc_adapter_metadata(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "jit_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint-last.pth").write_bytes(b"dry-run placeholder")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "bfc_speed_t02_10",
            "--num-images",
            "10",
            "--jit-ckpt-dir",
            str(ckpt_dir),
            "--output-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    meta = json.loads(result.stdout)["meta"]
    assert meta["prediction_type"] == "xpred"
    assert meta["output_to_velocity"] == "xpred_to_velocity"
    assert meta["pixbfc_adapter"]["model_name"] == "jit"
    assert meta["boundary_set"]["name"] == "jit_whole_backbone"


def test_deco_dry_run_includes_pixbfc_adapter_metadata(tmp_path: Path) -> None:
    ckpt = tmp_path / "deco.ckpt"
    config = tmp_path / "deco.yaml"
    ckpt.write_bytes(b"dry-run placeholder")
    config.write_text("model: {}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_deco_stage4a_generate.py",
            "--method",
            "bfc_all_candidates_t02_10",
            "--num-images",
            "10",
            "--deco-ckpt",
            str(ckpt),
            "--deco-config",
            str(config),
            "--output-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    meta = json.loads(result.stdout)["meta"]
    assert meta["prediction_type"] == "vpred"
    assert meta["output_to_velocity"] == "identity"
    assert meta["pixbfc_adapter"]["model_name"] == "deco"
    assert meta["boundary_set"]["name"] == "deco_all_candidates"
