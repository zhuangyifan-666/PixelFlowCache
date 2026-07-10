from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_jit_speca_dry_run_resolves_parameters_without_loading_runtime(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "speca_style",
            "--num-images",
            "8",
            "--batch-size",
            "2",
            "--run-id",
            "dryrun_jit_speca",
            "--output-root",
            str(output_root),
            "--jit-ckpt-dir",
            str(tmp_path / "missing_checkpoint"),
            "--speca-base-threshold",
            "0.08",
            "--speca-verifier-module",
            "auto",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    json_text, _separator, warning = result.stdout.partition("\nMissing JiT checkpoint:")
    payload = json.loads(json_text)
    meta = payload["meta"]
    assert meta["method_type"] == "speculative_cache"
    assert meta["baseline_name"] == "adapted SpeCa-style"
    assert meta["official_reproduction"] is False
    assert meta["speca_base_threshold"] == 0.08
    assert meta["speca_verifier_module_requested"] == "auto"
    assert meta["speca_verifier_module_resolved"] == "blocks.11"
    assert meta["speca_max_error_samples"] == 4096
    assert meta["selected_modules"][-1] == "blocks.11"
    assert warning
    assert not output_root.exists()
    assert "Traceback" not in result.stderr


def test_jit_speca_dry_run_preserves_explicit_verifier_request(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "speca_style",
            "--num-images",
            "2",
            "--output-root",
            str(tmp_path / "outputs"),
            "--speca-verifier-module",
            "blocks.10",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    json_text, _separator, _warning = result.stdout.partition("\nMissing JiT checkpoint:")
    meta = json.loads(json_text)["meta"]
    assert meta["speca_verifier_module_requested"] == "blocks.10"
    assert meta["speca_verifier_module_resolved"] == "blocks.10"


def test_jit_speca_dry_run_rejects_invalid_verifier_module(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "speca_style",
            "--num-images",
            "2",
            "--output-root",
            str(tmp_path / "outputs"),
            "--speca-verifier-module",
            "blocks.99",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "not in selected modules" in result.stderr
