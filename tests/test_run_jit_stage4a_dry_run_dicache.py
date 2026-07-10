from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_jit_dicache_dry_run_resolves_without_loading_checkpoint(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_stage4a_generate.py",
            "--method",
            "dicache_style",
            "--num-images",
            "8",
            "--batch-size",
            "2",
            "--run-id",
            "dryrun_jit_dicache",
            "--output-root",
            str(output_root),
            "--jit-ckpt-dir",
            str(tmp_path / "missing_checkpoint"),
            "--dicache-probe-depth",
            "1",
            "--dicache-reuse-threshold",
            "0.4",
            "--dicache-error-choice",
            "delta_y",
            "--dicache-branch-aggregation",
            "mean",
            "--dicache-ret-ratio",
            "0.2",
            "--dicache-force-last-step-full",
            "--dicache-dcta",
            "--dicache-gamma-min",
            "1.0",
            "--dicache-gamma-max",
            "1.5",
            "--dicache-eps",
            "1e-10",
            "--dicache-max-stat-samples",
            "4096",
            "--no-dicache-share-cfg-prefix",
            "--dicache-schedule-variant",
            "released_flux_compat",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    json_text, _separator, warning = result.stdout.partition("\nMissing JiT checkpoint:")
    meta = json.loads(json_text)["meta"]
    assert meta["method_type"] == "probe_cache"
    assert meta["baseline_name"] == "adapted DiCache-style"
    assert meta["official_reproduction"] is False
    assert meta["schedule_granularity"] == "batch_level_shared_cfg"
    assert meta["residual_space"] == "image_token_block_stack"
    assert meta["probe_depth"] == 1
    assert meta["total_blocks"] == 12
    assert meta["in_context_start"] == 4
    assert meta["in_context_len"] == 32
    assert meta["retention_full_last_step_idx"] == 10
    assert meta["retention_full_step_count"] == 11
    assert meta["share_cfg_prefix"] is False
    assert meta["dicache_share_cfg_prefix"] is False
    assert meta["schedule_variant"] == "released_flux_compat"
    assert meta["dicache_schedule_variant"] == "released_flux_compat"
    assert meta["cfg_prefix_fairness_mode"] == "strict_no_cache_matched"
    assert "released_" + "code_" + "compat" not in json_text
    assert meta["boundary_set"] is None
    assert warning
    assert not output_root.exists()
    assert "Traceback" not in result.stderr
