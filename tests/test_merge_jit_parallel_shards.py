from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.merge_jit_parallel_shards import _duplicate_indices, _merge_dicache_policy


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_jit_parallel_shards_merges_manifest_cache_and_latency(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True)
    for idx in range(8):
        (image_dir / f"{idx:06d}.png").write_bytes(b"png")
    for shard in range(4):
        rows = [{"index": idx, "label": idx} for idx in range(shard, 8, 4)]
        with (run_dir / f"manifest_shard{shard}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        _write_json(run_dir / f"cache_stats_shard{shard}.json", {"total_calls": 2, "hits": 1, "misses": 1, "refreshes": 1, "disabled": 0, "by_module": {"blocks.0": {"calls": 2, "hits": 1, "misses": 1, "refreshes": 1, "disabled": 0}}})
        _write_json(run_dir / f"latency_shard{shard}.json", {"latency_sec": 10 + shard, "generated_images_this_run": 2, "total_shard_images": 2})
        _write_json(run_dir / f"generation_meta_shard{shard}.json", {"method_name": "no_cache_50", "shard_index": shard})

    subprocess.run(
        [
            sys.executable,
            "scripts/merge_jit_parallel_shards.py",
            "--run-dir",
            str(run_dir),
            "--num-shards",
            "4",
            "--expected-images",
            "8",
            "--method",
            "no_cache_50",
            "--strict",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    manifest_indices = [json.loads(line)["index"] for line in (run_dir / "manifest.jsonl").read_text().splitlines()]
    assert manifest_indices == list(range(8))
    cache = json.loads((run_dir / "cache_stats.json").read_text())
    assert cache["total_calls"] == 8
    assert cache["hits"] == 4
    latency = json.loads((run_dir / "latency.json").read_text())
    assert latency["parallel_latency_sec"] == 13
    assert latency["images_per_sec"] == 8 / 13


def test_merge_jit_parallel_shards_merges_speca_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / "speca"
    (run_dir / "images").mkdir(parents=True)
    for idx in range(2):
        full_compute = 4 if idx == 0 else 6
        forecast = 6 if idx == 0 else 4
        verifier = 2 if idx == 0 else 1
        error_count = 2 if idx == 0 else 4
        error_mean = 1.0 if idx == 0 else 3.0
        completed_runs = 2 if idx == 0 else 1
        mean_run = 2.0 if idx == 0 else 5.0
        mean_order = 2.0 if idx == 0 else 4.0
        (run_dir / "images" / f"{idx:06d}.png").write_bytes(b"png")
        (run_dir / f"manifest_shard{idx}.jsonl").write_text(
            json.dumps({"index": idx, "label": idx}) + "\n",
            encoding="utf-8",
        )
        _write_json(
            run_dir / f"cache_stats_shard{idx}.json",
            {
                "enabled": True,
                "total_calls": 10,
                "hits": 6,
                "misses": 4,
                "refreshes": 4,
                "disabled": 0,
                "speca_policy": {
                    "config": {"verifier_module": "blocks.11"},
                    "total_steps_seen": 5,
                    "full_step_decisions": 2 if idx == 0 else 3,
                    "speculative_step_decisions": 3 if idx == 0 else 2,
                    "verification_steps": 2,
                    "verifier_fresh_calls": verifier,
                    "verification_accept_decisions": 3 if idx == 0 else 1,
                    "verification_reject_decisions": 1,
                    "forecast_committed": forecast,
                    "forecast_failures": 0,
                    "full_compute_calls": full_compute,
                    "logical_managed_calls": 10,
                    "actual_original_module_forwards": full_compute + verifier,
                    "effective_skipped_block_calls": forecast - verifier,
                    "mean_effective_order": mean_order,
                    "verification_errors": {
                        "count": error_count,
                        "mean": error_mean,
                        "std": 0.0,
                        "min": error_mean,
                        "max": error_mean,
                        "p50": error_mean,
                        "p90": error_mean,
                        "p95": error_mean,
                        "sample_count": error_count,
                        "max_samples": 4096,
                    },
                    "completed_speculative_runs": completed_runs,
                    "mean_speculative_run_length": mean_run,
                    "max_speculative_run_length": int(mean_run),
                    "verification_overhead_stats": {
                        "number_of_selected_blocks": 12,
                        "estimated_verifier_block_fraction": 1 / 12,
                        "verification_host_dispatch_time_sec": 0.1,
                        "forecast_host_dispatch_time_sec": 0.2,
                        "full_compute_host_dispatch_time_sec": 0.3,
                    },
                    "verifier_module": "blocks.11",
                    "max_order": 4,
                    "base_threshold": 0.1,
                    "decay_rate": 0.01,
                    "min_threshold": 0.01,
                    "min_forecast_steps": 2,
                    "max_forecast_steps": 5,
                    "first_full_steps": 3,
                    "error_metric": "relative_l1",
                    "branch_aggregation": "mean",
                },
            },
        )
        _write_json(run_dir / f"latency_shard{idx}.json", {"latency_sec": 1.0, "generated_images_this_run": 1})
        _write_json(run_dir / f"generation_meta_shard{idx}.json", {"method_name": "speca_style"})

    subprocess.run(
        [
            sys.executable,
            "scripts/merge_jit_parallel_shards.py",
            "--run-dir",
            str(run_dir),
            "--num-shards",
            "2",
            "--expected-images",
            "2",
            "--method",
            "speca_style",
            "--strict",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    merged = json.loads((run_dir / "cache_stats.json").read_text(encoding="utf-8"))
    speca = merged["speca_policy"]
    assert speca["full_step_decisions"] == 5
    assert speca["speculative_step_decisions"] == 5
    assert speca["forecast_committed"] == 10
    assert speca["full_compute_calls"] == 10
    assert speca["verifier_fresh_calls"] == 3
    assert speca["logical_managed_calls"] == 20
    assert speca["actual_original_module_forwards"] == 13
    assert speca["effective_skipped_block_calls"] == 7
    assert speca["raw_forecast_rate"] == 0.5
    assert speca["effective_compute_saving_rate"] == 0.35
    assert speca["verifier_overhead_rate"] == 0.15
    assert speca["actual_compute_fraction"] == 0.65
    assert speca["verification_acceptance_rate"] == 4 / 6
    assert speca["speculative_step_ratio"] == 0.5
    assert speca["mean_effective_order"] == 2.8
    assert speca["completed_speculative_runs"] == 3
    assert speca["mean_speculative_run_length"] == 3.0
    assert speca["verification_errors"]["count"] == 6
    assert speca["verification_errors"]["mean"] == 14 / 6
    assert speca["verification_errors"]["quantiles_approximate"] is True
    assert merged["verification_overhead_stats"]["estimated_verifier_block_fraction"] == 1 / 12


def test_merge_jit_parallel_shards_reports_manifest_and_png_index_problems(tmp_path: Path) -> None:
    run_dir = tmp_path / "invalid"
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True)
    for name in ("000000.png", "0.png", "000003.png", "bad.png"):
        (image_dir / name).write_bytes(b"png")
    (run_dir / "manifest_shard0.jsonl").write_text(
        "\n".join(
            json.dumps({"index": index, "label": 0})
            for index in (0, 0, 3)
        )
        + "\n",
        encoding="utf-8",
    )
    for stem, payload in (
        ("cache_stats_shard0.json", {}),
        ("latency_shard0.json", {"latency_sec": 1.0, "generated_images_this_run": 3}),
        ("generation_meta_shard0.json", {"method_name": "speca_style"}),
    ):
        _write_json(run_dir / stem, payload)

    command = [
        sys.executable,
        "scripts/merge_jit_parallel_shards.py",
        "--run-dir",
        str(run_dir),
        "--num-shards",
        "1",
        "--expected-images",
        "3",
        "--method",
        "speca_style",
    ]
    subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=60)
    report = json.loads((run_dir / "parallel_merge_report.json").read_text(encoding="utf-8"))
    assert report["duplicate_indices"] == [0]
    assert report["missing_indices"] == [1, 2]
    assert report["extra_indices"] == [3]
    assert report["duplicate_image_indices"] == [0]
    assert report["missing_image_indices"] == [1, 2]
    assert report["unexpected_image_indices"] == [3]
    assert report["invalid_image_filenames"] == ["bad.png"]
    assert report["warnings"]

    strict = subprocess.run(
        [*command, "--strict"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert strict.returncode != 0
    assert "strict checks failed" in strict.stderr


def test_duplicate_index_detection_scales_linearly_with_synthetic_indices() -> None:
    indices = list(range(50_000)) + [42, 49_999]
    assert _duplicate_indices(indices) == [42, 49_999]


def test_merge_dicache_counts_recomputes_ratios_and_running_moments() -> None:
    def stats(count: int, total: float, total_sq: float, samples: list[float]) -> dict:
        return {
            "count": count,
            "sum": total,
            "sum_sq": total_sq,
            "min": min(samples),
            "max": max(samples),
            "bounded_samples": samples,
        }

    policies = []
    for values in (
        {
            "total_steps_seen": 10,
            "full_step_decisions": 4,
            "reuse_step_decisions": 6,
            "reference_block_calls": 240,
            "probe_block_calls": 20,
            "deep_block_calls": 88,
            "actual_block_calls": 108,
            "effective_skipped_block_calls": 132,
            "reference_cfg_prefix_calls": 20,
            "actual_cfg_prefix_calls": 20,
            "cfg_prefix_calls_saved": 0,
            "decision_device_to_host_syncs": 8,
            "dcta_branch_calls": 12,
            "dcta_branch_fallback_calls": 2,
            "dcta_branch_degenerate_fallback_calls": 1,
            "dcta_branch_insufficient_history_fallback_calls": 1,
            "dcta_steps": 6,
            "dcta_fallback_steps": 1,
            "peak_history_storage_bytes": 100,
            "decision_branch_error_stats": stats(2, 2.0, 4.0, [0.0, 2.0]),
            "gamma_stats": stats(1, 1.2, 1.44, [1.2]),
        },
        {
            "total_steps_seen": 20,
            "full_step_decisions": 10,
            "reuse_step_decisions": 10,
            "reference_block_calls": 480,
            "probe_block_calls": 40,
            "deep_block_calls": 220,
            "actual_block_calls": 260,
            "effective_skipped_block_calls": 220,
            "reference_cfg_prefix_calls": 40,
            "actual_cfg_prefix_calls": 40,
            "cfg_prefix_calls_saved": 0,
            "decision_device_to_host_syncs": 15,
            "dcta_branch_calls": 20,
            "dcta_branch_fallback_calls": 3,
            "dcta_branch_degenerate_fallback_calls": 2,
            "dcta_branch_insufficient_history_fallback_calls": 1,
            "dcta_steps": 10,
            "dcta_fallback_steps": 2,
            "peak_history_storage_bytes": 200,
            "decision_branch_error_stats": stats(3, 12.0, 56.0, [2.0, 4.0, 6.0]),
            "gamma_stats": stats(3, 4.2, 5.9, [1.3, 1.4, 1.5]),
        },
    ):
        policies.append(
            {
                "dicache_policy": {
                    "config": {
                        "ret_ratio": 0.2,
                        "gamma_min": 1.0,
                        "gamma_max": 1.5,
                        "max_error_samples": 3,
                    },
                    "probe_depth": 1,
                    "total_blocks": 12,
                    "deep_blocks": 11,
                    "cfg_branches": 2,
                    "schedule_variant": "released_flux_compat",
                    "share_cfg_prefix": False,
                    "observed_delta_y_stats": values["decision_branch_error_stats"],
                    "observed_delta_x_stats": values["decision_branch_error_stats"],
                    "observed_branch_error_stats": values["decision_branch_error_stats"],
                    "decision_delta_y_stats": values["decision_branch_error_stats"],
                    "decision_delta_x_stats": values["decision_branch_error_stats"],
                    **values,
                }
            }
        )
    merged = _merge_dicache_policy(policies)
    assert merged is not None
    assert merged["total_steps_seen"] == 30
    assert merged["reuse_step_ratio"] == 16 / 30
    assert merged["actual_block_compute_fraction"] == 368 / 720
    assert merged["effective_block_compute_saving_rate"] == 352 / 720
    decision = merged["decision_branch_error_stats"]
    assert decision["count"] == 5
    assert decision["mean"] == 14 / 5
    assert decision["std"] == (60 / 5 - (14 / 5) ** 2) ** 0.5
    assert len(decision["bounded_samples"]) == 3
    assert decision["quantiles_approximate"] is True
    assert merged["gamma_stats"]["mean"] == 1.35
    assert merged["reference_cfg_prefix_calls"] == 60
    assert merged["actual_cfg_prefix_calls"] == 60
    assert merged["decision_device_to_host_syncs"] == 23
    assert merged["dcta_branch_calls"] == 32
    assert merged["dcta_branch_fallback_calls"] == 5
    assert merged["peak_history_storage_bytes"] == 200
