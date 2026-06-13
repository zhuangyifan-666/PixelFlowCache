from __future__ import annotations

from pathlib import Path

from scripts import run_deco_stage4a_generate, run_jit_stage4a_generate


def test_jit_dynamic_metadata_uses_runtime_threshold(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "jit_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint-last.pth").write_bytes(b"placeholder")
    parser = run_jit_stage4a_generate.build_parser()
    args = parser.parse_args(
        [
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
        ]
    )
    resolved = run_jit_stage4a_generate.resolve_config(args)
    meta = resolved["meta"]

    assert meta["model_name"] == "JiT"
    assert meta["method_name"] == "seacache_style"
    assert meta["dynamic_cache_type"] == "sea"
    assert meta["dynamic_cache_threshold"] == 0.06
    assert meta["resolved_dynamic_cache_threshold"] == 0.06
    assert meta["method"]["dynamic_cache_threshold"] == 0.06
    assert meta["method"]["resolved_dynamic_cache_threshold"] == 0.06
    assert meta["eval_steps"] == 50
    assert meta["reference_steps"] == 50
    assert meta["cache_units"] == "jit_blocks"
    assert meta["selected_modules"] == "all"


def test_deco_dynamic_metadata_uses_runtime_threshold(tmp_path: Path) -> None:
    ckpt = tmp_path / "deco.ckpt"
    config = tmp_path / "deco.yaml"
    ckpt.write_bytes(b"placeholder")
    config.write_text("model: {}\n", encoding="utf-8")
    parser = run_deco_stage4a_generate.build_parser()
    args = parser.parse_args(
        [
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
        ]
    )
    resolved = run_deco_stage4a_generate.resolve_config(args)
    meta = resolved["meta"]

    assert meta["model_name"] == "DeCo"
    assert meta["method_name"] == "seacache_style"
    assert meta["dynamic_cache_type"] == "sea"
    assert meta["dynamic_cache_threshold"] == 0.06
    assert meta["resolved_dynamic_cache_threshold"] == 0.06
    assert meta["method"]["dynamic_cache_threshold"] == 0.06
    assert meta["method"]["resolved_dynamic_cache_threshold"] == 0.06
    assert meta["eval_steps"] == 50
    assert meta["reference_steps"] == 50
    assert meta["cache_units"] == "all_candidates"
    assert meta["selected_modules"] == "all_candidates"
