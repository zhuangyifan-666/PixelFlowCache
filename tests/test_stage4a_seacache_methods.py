from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pfc.eval.method_presets import get_deco_stage4a_methods, get_jit_stage4a_methods


ROOT = Path(__file__).resolve().parents[1]


def test_seacache_methods_are_registered() -> None:
    assert {"teacache_style", "seacache_style"}.issubset(get_jit_stage4a_methods())
    assert {"teacache_style", "seacache_style"}.issubset(get_deco_stage4a_methods())
    assert get_jit_stage4a_methods()["seacache_style"].dynamic_cache_type == "sea"
    assert get_deco_stage4a_methods()["teacache_style"].dynamic_cache_type == "tea"


def test_jit_seacache_dry_run_with_fake_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "jit_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint-last.pth").write_bytes(b"dry-run placeholder")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "seacache_style",
            "--dynamic-cache-threshold",
            "0.06",
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
        timeout=60,
    )
    payload = json.loads(result.stdout)
    meta = payload["meta"]
    assert meta["method_name"] == "seacache_style"
    assert meta["dynamic_cache"]["threshold"] == 0.06
    assert meta["resolved_dynamic_cache_threshold"] == 0.06
    assert meta["method"]["resolved_dynamic_cache_threshold"] == 0.06


def test_deco_seacache_dry_run_with_fake_checkpoint_and_config(tmp_path: Path) -> None:
    ckpt = tmp_path / "deco.ckpt"
    config = tmp_path / "deco.yaml"
    ckpt.write_bytes(b"dry-run placeholder")
    config.write_text("model: {}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_deco_stage4a_generate.py",
            "--method",
            "seacache_style",
            "--dynamic-cache-threshold",
            "0.06",
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
        timeout=60,
    )
    payload = json.loads(result.stdout)
    meta = payload["meta"]
    assert meta["method_name"] == "seacache_style"
    assert meta["dynamic_cache"]["threshold"] == 0.06
    assert meta["resolved_dynamic_cache_threshold"] == 0.06
    assert meta["method"]["resolved_dynamic_cache_threshold"] == 0.06
