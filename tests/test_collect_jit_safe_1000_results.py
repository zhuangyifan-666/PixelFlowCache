from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_jit_safe_1000_results_writes_summary(tmp_path: Path) -> None:
    run_id = "unit_run"
    output_root = tmp_path / "outputs"
    fid_root = tmp_path / "fid"
    pair_root = tmp_path / "pair"
    out_dir = tmp_path / "summary"
    methods = ["no_cache_50", "safe_bfc_quality", "taylorseer_style", "speca_style", "dicache_style"]

    for method, ips in [
        ("no_cache_50", 5.0),
        ("safe_bfc_quality", 10.0),
        ("taylorseer_style", 8.0),
        ("speca_style", 9.0),
        ("dicache_style", 9.5),
    ]:
        run_dir = output_root / "jit" / run_id / method
        _write_json(run_dir / "generation_meta.json", {"num_images": 1000, "eval_steps": 50})
        _write_json(
            run_dir / "latency.json",
            {
                "latency_sec": 200.0 / ips,
                "images_per_sec": ips,
                "generated_images": 1000,
                "generated_images_this_run": 1000 if method == "no_cache_50" else 900,
                "existing_images_skipped": 0 if method == "no_cache_50" else 100,
            },
        )
        cache_payload = {"hit_rate": 0.0, "total_calls": 0, "hits": 0, "refreshes": 0}
        if method == "safe_bfc_quality":
            cache_payload["safe_policy"] = {
                "config": {"max_age": 2, "safe_lambda": 0.5, "quantile": 0.95},
                "stats": {"safe_reuse": 7, "unsafe_refresh": 3, "mean_age": 1.5},
            }
        if method == "taylorseer_style":
            cache_payload["taylorseer_policy"] = {
                "config": {"interval": 4, "max_order": 4},
                "stats": {
                    "forecast_decisions": 11,
                    "forecast_committed": 10,
                    "forecast_failures": 1,
                    "mean_effective_order": 3.5,
                },
            }
        if method == "speca_style":
            cache_payload["speca_policy"] = {
                "config": {
                    "base_threshold": 0.1,
                    "decay_rate": 0.01,
                    "min_threshold": 0.01,
                    "min_forecast_steps": 2,
                    "max_forecast_steps": 5,
                    "max_order": 4,
                    "error_metric": "relative_l1",
                    "verifier_module_requested": "auto",
                    "verifier_module_resolved": "blocks.11",
                    "branch_aggregation": "mean",
                    "timing_semantics": "host_dispatch_only",
                },
                "full_step_decisions": 20,
                "speculative_step_decisions": 30,
                "speculative_step_ratio": 0.6,
                "verification_steps": 18,
                "verifier_fresh_calls": 36,
                "verification_accept_decisions": 12,
                "verification_reject_decisions": 4,
                "verification_acceptance_rate": 0.75,
                "completed_speculative_runs": 8,
                "mean_speculative_run_length": 3.0,
                "max_speculative_run_length": 5,
                "verification_errors": {
                    "count": 36,
                    "mean": 0.02,
                    "std": 0.01,
                    "p50": 0.015,
                    "p90": 0.06,
                    "p95": 0.08,
                },
                "forecast_committed": 600,
                "forecast_failures": 2,
                "mean_effective_order": 3.5,
                "logical_managed_calls": 1000,
                "full_compute_calls": 400,
                "actual_original_module_forwards": 436,
                "effective_skipped_block_calls": 564,
                "raw_forecast_rate": 0.6,
                "effective_compute_saving_rate": 0.564,
                "verifier_overhead_rate": 0.036,
                "actual_compute_fraction": 0.436,
                "timing_semantics": "host_dispatch_only",
            }
        if method == "dicache_style":
            _write_json(
                run_dir / "generation_meta.json",
                {
                    "num_images": 1000,
                    "eval_steps": 50,
                    "cfg_prefix_fairness_mode": "strict_no_cache_matched",
                },
            )
            cache_payload["dicache_policy"] = {
                "config": {
                    "probe_depth": 1,
                    "reuse_threshold": 0.4,
                    "error_choice": "delta_y",
                    "branch_aggregation": "mean",
                    "ret_ratio": 0.2,
                    "dcta_enabled": True,
                    "gamma_min": 1.0,
                    "gamma_max": 1.5,
                    "share_cfg_prefix": False,
                    "schedule_variant": "released_flux_compat",
                },
                "probe_depth": 1,
                "reuse_threshold": 0.4,
                "error_choice": "delta_y",
                "branch_aggregation": "mean",
                "dcta_enabled": True,
                "full_step_decisions": 20,
                "reuse_step_decisions": 30,
                "reuse_step_ratio": 0.6,
                "probe_block_calls": 100,
                "deep_block_calls": 440,
                "reference_block_calls": 1200,
                "actual_block_calls": 540,
                "effective_skipped_block_calls": 660,
                "effective_block_compute_saving_rate": 0.55,
                "actual_block_compute_fraction": 0.45,
                "share_cfg_prefix": False,
                "schedule_variant": "released_flux_compat",
                "reference_cfg_prefix_calls": 100,
                "actual_cfg_prefix_calls": 100,
                "cfg_prefix_calls_saved": 0,
                "decision_device_to_host_syncs": 40,
                "decision_syncs_per_step": 0.8,
                "observed_branch_error_stats": {
                    "count": 80,
                    "mean": 0.02,
                    "std": 0.01,
                    "p95": 0.06,
                },
                "decision_branch_error_stats": {
                    "count": 60,
                    "mean": 0.015,
                    "std": 0.008,
                    "p95": 0.04,
                },
                "dcta_branch_calls": 50,
                "dcta_branch_fallback_calls": 10,
                "dcta_branch_degenerate_fallback_calls": 3,
                "dcta_branch_insufficient_history_fallback_calls": 7,
                "gamma_stats": {"mean": 1.25, "p50": 1.2, "p90": 1.5},
                "gamma_clip_low_count": 2,
                "gamma_clip_high_count": 4,
                "peak_history_storage_bytes": 8192,
                "timing_semantics": "host_dispatch_only",
            }
        _write_json(run_dir / "cache_stats.json", cache_payload)
        _write_json(fid_root / run_id / "jit" / method / "fid_results.json", {"fid": 12.3, "is": 45.6})

    _write_json(
        pair_root / run_id / "jit" / "safe_bfc_quality" / "pair_metrics.json",
        {
            "pair_count": 1000,
            "summary": {
                "psnr": {"mean": 30.0},
                "ssim": {"mean": 0.9},
                "lpips": {"mean": 0.1},
                "rel_l2": {"mean": 0.02},
            }
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_jit_safe_1000_results.py",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--fid-root",
            str(fid_root),
            "--pair-root",
            str(pair_root),
            "--methods",
            ",".join(methods),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    rows = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    safe = next(row for row in rows if row["method"] == "safe_bfc_quality")
    assert safe["speedup_vs_no_cache"] == ""
    assert safe["legacy_timing"] == "True"
    assert safe["comparable_for_algorithm_speedup"] == "False"
    assert safe["safe_reuse"] == "7"
    assert safe["pair_count"] == "1000"
    taylor = next(row for row in rows if row["method"] == "taylorseer_style")
    assert taylor["forecast_decisions"] == "11"
    assert taylor["forecast_committed"] == "10"
    assert taylor["forecast_failures"] == "1"
    assert taylor["mean_effective_order"] == "3.5"
    assert taylor["taylorseer_interval"] == "4"
    assert taylor["taylorseer_max_order"] == "4"
    speca = next(row for row in rows if row["method"] == "speca_style")
    assert speca["speca_full_step_decisions"] == "20"
    assert speca["speca_speculative_step_decisions"] == "30"
    assert speca["speca_speculative_step_ratio"] == "0.6"
    assert speca["speca_verification_steps"] == "18"
    assert speca["speca_verifier_fresh_calls"] == "36"
    assert speca["speca_verification_acceptance_rate"] == "0.75"
    assert speca["speca_forecast_committed"] == "600"
    assert speca["forecast_committed"] == "600"
    assert speca["speca_mean_effective_order"] == "3.5"
    assert speca["speca_effective_compute_saving_rate"] == "0.564"
    assert speca["effective_compute_saving_rate"] == "0.564"
    assert speca["speca_verification_error_p95"] == "0.08"
    assert speca["speca_verifier_module_requested"] == "auto"
    assert speca["speca_verifier_module_resolved"] == "blocks.11"
    assert speca["speca_timing_semantics"] == "host_dispatch_only"
    dicache = next(row for row in rows if row["method"] == "dicache_style")
    assert dicache["dicache_probe_depth"] == "1"
    assert dicache["dicache_reuse_threshold"] == "0.4"
    assert dicache["dicache_error_choice"] == "delta_y"
    assert dicache["dicache_branch_aggregation"] == "mean"
    assert dicache["dicache_ret_ratio"] == "0.2"
    assert dicache["dicache_full_steps"] == "20"
    assert dicache["dicache_reuse_steps"] == "30"
    assert dicache["dicache_effective_block_compute_saving_rate"] == "0.55"
    assert dicache["effective_compute_saving_rate"] == "0.55"
    assert dicache["dicache_share_cfg_prefix"] == "False"
    assert dicache["dicache_schedule_variant"] == "released_flux_compat"
    assert dicache["dicache_cfg_prefix_fairness_mode"] == "strict_no_cache_matched"
    assert dicache["dicache_reference_cfg_prefix_calls"] == "100"
    assert dicache["dicache_actual_cfg_prefix_calls"] == "100"
    assert dicache["dicache_decision_syncs"] == "40"
    assert dicache["dicache_observed_error_count"] == "80"
    assert dicache["dicache_decision_error_count"] == "60"
    assert dicache["dicache_dcta_branch_calls"] == "50"
    assert dicache["dicache_dcta_degenerate_fallback_calls"] == "3"
    assert dicache["dicache_gamma_mean"] == "1.25"
    assert dicache["dicache_peak_history_storage_bytes"] == "8192"
    assert dicache["dicache_timing_semantics"] == "host_dispatch_only"
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert any("resume skipped" in warning for warning in payload["warnings"])
    summary_md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "1000-image proxy results" in summary_md
    assert "forecast committed" in summary_md
    assert "effective compute saving" in summary_md
    assert "latency/speedup is not comparable" in summary_md
    assert "not exact FLOPs" in summary_md
    assert "batch-level shared CFG" in summary_md
