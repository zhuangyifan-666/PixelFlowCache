#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.generation_io import count_images, write_manifest_atomic  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sum_by_module(shards: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for shard in shards:
        for module, stats in (shard.get("by_module") or {}).items():
            for key, value in stats.items():
                if isinstance(value, int):
                    merged[module][key] += value
    return {module: dict(values) for module, values in sorted(merged.items())}


def _merge_nested_counts(values: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for payload in values:
        for outer, inner in (payload or {}).items():
            if isinstance(inner, dict):
                for key, value in inner.items():
                    if isinstance(value, int):
                        merged[str(outer)][str(key)] += value
    return {key: dict(value) for key, value in sorted(merged.items())}


def _merge_safe_policy(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    policies = [shard.get("safe_policy") for shard in shards if isinstance(shard.get("safe_policy"), dict)]
    if not policies:
        return None
    merged_stats: dict[str, Any] = defaultdict(int)
    reuse_weighted_age = 0.0
    reuse_count = 0
    nested_keys = ["by_reason", "by_boundary", "by_age", "by_step", "by_branch", "by_solver_stage"]
    nested_values: dict[str, list[dict[str, Any]]] = {key: [] for key in nested_keys}
    for policy in policies:
        stats = policy.get("stats") or {}
        for key, value in stats.items():
            if key in nested_keys:
                nested_values[key].append(value)
            elif isinstance(value, int):
                merged_stats[key] += value
        count = int(stats.get("safe_reuse_committed", stats.get("safe_reuse", 0)) or 0)
        reuse_count += count
        reuse_weighted_age += float(stats.get("mean_age_of_reuse", stats.get("mean_age", 0.0)) or 0.0) * count
    mean_age = reuse_weighted_age / reuse_count if reuse_count else 0.0
    merged_stats["mean_age"] = mean_age
    merged_stats["mean_age_of_reuse"] = mean_age
    merged_stats["max_age"] = max((int((policy.get("stats") or {}).get("max_age", 0) or 0) for policy in policies), default=0)
    merged_stats["max_age_of_reuse"] = max(
        (int((policy.get("stats") or {}).get("max_age_of_reuse", 0) or 0) for policy in policies),
        default=0,
    )
    for key, payloads in nested_values.items():
        if key == "by_reason":
            flat: dict[str, int] = defaultdict(int)
            for payload in payloads:
                for reason, value in payload.items():
                    if isinstance(value, int):
                        flat[str(reason)] += value
            merged_stats[key] = dict(sorted(flat.items()))
        else:
            merged_stats[key] = _merge_nested_counts(payloads)
    return {"policy": policies[0].get("policy"), "config": policies[0].get("config"), "stats": dict(merged_stats)}


def _merge_taylorseer_policy(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    policies = [shard.get("taylorseer_policy") for shard in shards if isinstance(shard.get("taylorseer_policy"), dict)]
    if not policies:
        return None
    merged_stats: dict[str, Any] = defaultdict(int)
    nested_keys = ["by_module", "by_branch", "by_step", "by_order"]
    nested_values: dict[str, list[dict[str, Any]]] = {key: [] for key in nested_keys}
    weighted_order = 0.0
    committed = 0
    for policy in policies:
        stats = policy.get("stats") or {}
        for key, value in stats.items():
            if key in nested_keys:
                nested_values[key].append(value)
            elif isinstance(value, int):
                merged_stats[key] += value
        count = int(stats.get("forecast_committed", 0) or 0)
        committed += count
        weighted_order += float(stats.get("mean_effective_order", 0.0) or 0.0) * count
    merged_stats["mean_effective_order"] = weighted_order / committed if committed else 0.0
    for key, payloads in nested_values.items():
        if key == "by_order":
            flat: dict[str, int] = defaultdict(int)
            for payload in payloads:
                for order, value in (payload or {}).items():
                    if isinstance(value, int):
                        flat[str(order)] += value
            merged_stats[key] = dict(sorted(flat.items()))
        else:
            merged_stats[key] = _merge_nested_counts(payloads)
    return {"policy": policies[0].get("policy"), "config": policies[0].get("config"), "stats": dict(merged_stats)}


def _merge_speca_policy(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    policies = [shard.get("speca_policy") for shard in shards if isinstance(shard.get("speca_policy"), dict)]
    if not policies:
        return None
    count_keys = [
        "total_steps_seen",
        "total_step_decisions",
        "full_step_decisions",
        "speculative_step_decisions",
        "initial_full_steps",
        "insufficient_history_full_steps",
        "verification_reject_full_steps",
        "max_length_full_steps",
        "missing_verification_error_full_steps",
        "verification_steps",
        "verifier_fresh_calls",
        "verification_accept_decisions",
        "verification_reject_decisions",
        "missing_branch_verification",
        "forecast_committed",
        "forecast_failures",
        "logical_managed_calls",
        "full_compute_calls",
        "actual_original_module_forwards",
        "effective_skipped_block_calls",
    ]
    merged: dict[str, Any] = {
        "policy_name": "SpeCaCachePolicy",
        "baseline_name": "adapted SpeCa-style",
        "config": policies[0].get("config") or {},
    }
    for key in count_keys:
        merged[key] = sum(int(policy.get(key, 0) or 0) for policy in policies)

    merged["total_step_decisions"] = (
        merged["full_step_decisions"] + merged["speculative_step_decisions"]
    )
    merged["logical_managed_calls"] = (
        merged["full_compute_calls"] + merged["forecast_committed"]
    )
    merged["actual_original_module_forwards"] = (
        merged["full_compute_calls"] + merged["verifier_fresh_calls"]
    )
    merged["effective_skipped_block_calls"] = max(
        merged["forecast_committed"] - merged["verifier_fresh_calls"],
        0,
    )
    merged["raw_cache_hits"] = merged["forecast_committed"]
    logical = merged["logical_managed_calls"]
    total_decisions = merged["total_step_decisions"]
    accept_total = (
        merged["verification_accept_decisions"] + merged["verification_reject_decisions"]
    )
    merged["raw_forecast_rate"] = merged["forecast_committed"] / logical if logical else 0.0
    merged["effective_compute_saving_rate"] = (
        merged["effective_skipped_block_calls"] / logical if logical else 0.0
    )
    merged["verifier_overhead_rate"] = (
        merged["verifier_fresh_calls"] / logical if logical else 0.0
    )
    merged["actual_compute_fraction"] = (
        merged["actual_original_module_forwards"] / logical if logical else 0.0
    )
    merged["verification_acceptance_rate"] = (
        merged["verification_accept_decisions"] / accept_total if accept_total else 0.0
    )
    merged["speculative_step_ratio"] = (
        merged["speculative_step_decisions"] / total_decisions if total_decisions else 0.0
    )

    error_payloads = [policy.get("verification_errors") or {} for policy in policies]
    error_count = sum(int(item.get("count", 0) or 0) for item in error_payloads)
    error_sum = sum(
        float(item.get("mean", 0.0) or 0.0) * int(item.get("count", 0) or 0)
        for item in error_payloads
    )
    error_sum_sq = sum(
        (
            float(item.get("std", 0.0) or 0.0) ** 2
            + float(item.get("mean", 0.0) or 0.0) ** 2
        )
        * int(item.get("count", 0) or 0)
        for item in error_payloads
    )
    error_mean = error_sum / error_count if error_count else None
    error_std = (
        max(error_sum_sq / error_count - float(error_mean) ** 2, 0.0) ** 0.5
        if error_count and error_mean is not None
        else None
    )

    def weighted_quantile(name: str) -> float | None:
        weighted = [
            (float(item[name]), int(item.get("sample_count", item.get("count", 0)) or 0))
            for item in error_payloads
            if item.get(name) is not None
        ]
        weight = sum(count for _value, count in weighted)
        return sum(value * count for value, count in weighted) / weight if weight else None

    error_mins = [float(item["min"]) for item in error_payloads if item.get("min") is not None]
    error_maxes = [float(item["max"]) for item in error_payloads if item.get("max") is not None]
    merged["verification_errors"] = {
        "count": error_count,
        "mean": error_mean,
        "std": error_std,
        "min": min(error_mins) if error_mins else None,
        "max": max(error_maxes) if error_maxes else None,
        "p50": weighted_quantile("p50"),
        "p90": weighted_quantile("p90"),
        "p95": weighted_quantile("p95"),
        "sample_count": sum(int(item.get("sample_count", 0) or 0) for item in error_payloads),
        "quantiles_approximate": bool(error_count),
        "max_samples": sum(int(item.get("max_samples", 0) or 0) for item in error_payloads),
    }

    completed_runs = sum(int(policy.get("completed_speculative_runs", 0) or 0) for policy in policies)
    merged["completed_speculative_runs"] = completed_runs
    merged["mean_speculative_run_length"] = (
        sum(
            float(policy.get("mean_speculative_run_length", 0.0) or 0.0)
            * int(policy.get("completed_speculative_runs", 0) or 0)
            for policy in policies
        )
        / completed_runs
        if completed_runs
        else 0.0
    )
    merged["max_speculative_run_length"] = max(
        (int(policy.get("max_speculative_run_length", 0) or 0) for policy in policies),
        default=0,
    )
    committed = merged["forecast_committed"]
    merged["mean_effective_order"] = (
        sum(float(policy.get("mean_effective_order", 0.0) or 0.0) * int(policy.get("forecast_committed", 0) or 0) for policy in policies)
        / committed
        if committed
        else 0.0
    )
    for key in (
        "verifier_module",
        "max_order",
        "base_threshold",
        "decay_rate",
        "min_threshold",
        "min_forecast_steps",
        "max_forecast_steps",
        "first_full_steps",
        "error_metric",
        "branch_aggregation",
        "verifier_module_requested",
        "verifier_module_resolved",
        "timing_semantics",
    ):
        merged[key] = policies[0].get(key)
    merged["by_module"] = _merge_nested_counts([policy.get("by_module") or {} for policy in policies])
    merged["by_branch"] = _merge_nested_counts([policy.get("by_branch") or {} for policy in policies])
    merged["by_step"] = {f"shard{idx}": policy.get("by_step") or {} for idx, policy in enumerate(policies)}

    overheads = [policy.get("verification_overhead_stats") or {} for policy in policies]
    merged["verification_overhead_stats"] = {
        "verifier_fresh_calls": merged["verifier_fresh_calls"],
        "number_of_selected_blocks": overheads[0].get("number_of_selected_blocks") if overheads else None,
        "estimated_verifier_block_fraction": overheads[0].get("estimated_verifier_block_fraction") if overheads else None,
        "timing_semantics": "host_dispatch_only",
        "cuda_event_profiling_enabled": False,
        "verification_host_dispatch_time_sec": sum(float(item.get("verification_host_dispatch_time_sec", 0.0) or 0.0) for item in overheads),
        "forecast_host_dispatch_time_sec": sum(float(item.get("forecast_host_dispatch_time_sec", 0.0) or 0.0) for item in overheads),
        "full_compute_host_dispatch_time_sec": sum(float(item.get("full_compute_host_dispatch_time_sec", 0.0) or 0.0) for item in overheads),
        "verification_cuda_time_sec": None,
        "forecast_cuda_time_sec": None,
        "full_compute_cuda_time_sec": None,
    }
    merged["stats"] = {
        key: value
        for key, value in merged.items()
        if key not in {"config", "stats", "by_step", "by_branch", "by_module"}
    }
    return merged


def _sample_quantile(samples: list[float], probability: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bounded_merged_samples(
    summaries: list[dict[str, Any]],
    maximum: int,
) -> list[float]:
    values = [
        float(value)
        for summary in summaries
        for value in (
            summary.get("bounded_samples")
            or summary.get("samples_bounded")
            or []
        )
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[-1]]
    return [values[round(index * (len(values) - 1) / (maximum - 1))] for index in range(maximum)]


def _merge_running_stats(
    summaries: list[dict[str, Any]],
    maximum: int,
) -> dict[str, Any]:
    count = sum(int(summary.get("count", 0) or 0) for summary in summaries)
    total = sum(float(summary.get("sum", 0.0) or 0.0) for summary in summaries)
    total_sq = sum(float(summary.get("sum_sq", 0.0) or 0.0) for summary in summaries)
    mean = total / count if count else 0.0
    variance = max(total_sq / count - mean * mean, 0.0) if count else 0.0
    bounded_samples = _bounded_merged_samples(summaries, maximum)
    return {
        "count": count,
        "sum": total,
        "sum_sq": total_sq,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(
            (
                float(summary["min"])
                for summary in summaries
                if summary.get("min") is not None
            ),
            default=None,
        ),
        "max": max(
            (
                float(summary["max"])
                for summary in summaries
                if summary.get("max") is not None
            ),
            default=None,
        ),
        "p50": _sample_quantile(bounded_samples, 0.50),
        "p90": _sample_quantile(bounded_samples, 0.90),
        "p95": _sample_quantile(bounded_samples, 0.95),
        "sample_count": len(bounded_samples),
        "max_samples": maximum,
        "quantiles_approximate": True,
        "bounded_samples": bounded_samples,
        "samples_bounded": bounded_samples,
    }


def _merge_dicache_policy(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    policies = [
        shard.get("dicache_policy")
        for shard in shards
        if isinstance(shard.get("dicache_policy"), dict)
    ]
    if not policies:
        return None
    count_keys = [
        "total_steps_seen",
        "full_step_decisions",
        "reuse_step_decisions",
        "first_full_steps",
        "retention_full_steps",
        "adaptive_refresh_steps",
        "last_step_full_steps",
        "insufficient_history_full_steps",
        "force_full_steps",
        "reference_block_calls",
        "probe_block_calls",
        "deep_block_calls",
        "actual_block_calls",
        "effective_skipped_block_calls",
        "reference_cfg_prefix_calls",
        "actual_cfg_prefix_calls",
        "cfg_prefix_calls_saved",
        "decision_device_to_host_syncs",
        "dcta_branch_calls",
        "dcta_branch_fallback_calls",
        "dcta_branch_degenerate_fallback_calls",
        "dcta_branch_insufficient_history_fallback_calls",
        "dcta_steps",
        "dcta_fallback_steps",
        "gamma_clip_low_count",
        "gamma_clip_high_count",
    ]
    merged: dict[str, Any] = {
        "policy_name": "DiCachePolicy",
        "baseline_name": "adapted DiCache-style",
        "official_reproduction": False,
        "config": policies[0].get("config") or {},
    }
    for key in count_keys:
        merged[key] = sum(int(policy.get(key, 0) or 0) for policy in policies)

    for key in (
        "probe_depth",
        "total_blocks",
        "deep_blocks",
        "cfg_branches",
        "branch_aggregation",
        "error_choice",
        "reuse_threshold",
        "dcta_enabled",
        "retention_full_last_step_idx",
        "retention_full_step_count",
        "timing_semantics",
        "schedule_variant",
        "share_cfg_prefix",
        "cfg_prefix_sharing_enabled",
        "debug_sync_overhead_enabled",
    ):
        merged[key] = policies[0].get(key)
    steps = merged["total_steps_seen"]
    reference_calls = merged["reference_block_calls"]
    merged["reuse_step_ratio"] = merged["reuse_step_decisions"] / steps if steps else 0.0
    merged["effective_block_compute_saving_rate"] = (
        merged["effective_skipped_block_calls"] / reference_calls if reference_calls else 0.0
    )
    merged["actual_block_compute_fraction"] = (
        merged["actual_block_calls"] / reference_calls if reference_calls else 0.0
    )
    merged["decision_syncs_per_step"] = (
        merged["decision_device_to_host_syncs"] / steps if steps else 0.0
    )
    max_samples = int(
        (merged.get("config") or {}).get("max_error_samples", 4096) or 4096
    )
    for field in (
        "observed_delta_y_stats",
        "observed_delta_x_stats",
        "observed_branch_error_stats",
        "decision_delta_y_stats",
        "decision_delta_x_stats",
        "decision_branch_error_stats",
        "gamma_stats",
    ):
        merged[field] = _merge_running_stats(
            [
                policy[field]
                for policy in policies
                if isinstance(policy.get(field), dict)
            ],
            max_samples,
        )
    for key in (
        "probe_host_dispatch_time_sec",
        "deep_compute_host_dispatch_time_sec",
        "dcta_host_dispatch_time_sec",
        "final_layer_host_dispatch_time_sec",
    ):
        merged[key] = sum(float(policy.get(key, 0.0) or 0.0) for policy in policies)
    merged["current_history_storage_bytes"] = sum(
        int(policy.get("current_history_storage_bytes", 0) or 0)
        for policy in policies
    )
    merged["peak_history_storage_bytes"] = max(
        (int(policy.get("peak_history_storage_bytes", 0) or 0) for policy in policies),
        default=0,
    )
    for key in ("history_tensor_count", "history_unique_storage_count"):
        merged[key] = sum(int(policy.get(key, 0) or 0) for policy in policies)
    merged["accumulated_error_current"] = 0.0
    return merged


def _merge_cache_stats(shards: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["total_calls", "hits", "misses", "refreshes", "disabled"]
    merged = {key: sum(int(shard.get(key, 0) or 0) for shard in shards) for key in keys}
    merged["hit_rate"] = merged["hits"] / merged["total_calls"] if merged["total_calls"] else 0.0
    merged["model_name"] = next((shard.get("model_name") for shard in shards if shard.get("model_name")), "JiT")
    merged["enabled"] = any(bool(shard.get("enabled")) for shard in shards)
    merged["num_entries"] = sum(int(shard.get("num_entries", 0) or 0) for shard in shards)
    merged["by_module"] = _sum_by_module(shards)
    safe_policy = _merge_safe_policy(shards)
    if safe_policy is not None:
        merged["safe_policy"] = safe_policy
    taylorseer_policy = _merge_taylorseer_policy(shards)
    if taylorseer_policy is not None:
        merged["taylorseer_policy"] = taylorseer_policy
    speca_policy = _merge_speca_policy(shards)
    if speca_policy is not None:
        merged["speca_policy"] = speca_policy
        merged["verification_overhead_stats"] = speca_policy["verification_overhead_stats"]
        merged["runtime_cache"] = {
            key: merged[key]
            for key in ("model_name", "enabled", "num_entries", "total_calls", "hits", "misses", "refreshes", "disabled", "hit_rate", "by_module")
        }
    dicache_policy = _merge_dicache_policy(shards)
    if dicache_policy is not None:
        merged["dicache_policy"] = dicache_policy
    dynamic = [shard.get("dynamic_cache") for shard in shards if isinstance(shard.get("dynamic_cache"), dict)]
    if dynamic:
        merged["dynamic_cache"] = dynamic[0]
    return merged


def _merge_latency(shards: list[dict[str, Any]], launcher_meta: dict[str, Any] | None, image_count: int) -> dict[str, Any]:
    worker_sampling = [
        float(shard["sampling_latency_sec"])
        for shard in shards
        if shard.get("sampling_latency_sec") is not None
    ]
    worker_end_to_end = [
        float(shard.get("end_to_end_latency_sec", shard.get("latency_sec", 0.0)) or 0.0)
        for shard in shards
    ]
    orchestration = 0.0
    if launcher_meta:
        orchestration = float(
            launcher_meta.get(
                "parallel_orchestration_wall_time_sec",
                launcher_meta.get("wall_time_sec", 0.0),
            )
            or 0.0
        )
    if orchestration <= 0.0:
        orchestration = max(worker_end_to_end, default=0.0)
    generated_this_run = sum(int(shard.get("generated_images_this_run", shard.get("generated_images", 0)) or 0) for shard in shards)
    total_shard_images = sum(int(shard.get("total_shard_images", 0) or 0) for shard in shards)
    generated_images = generated_this_run
    skipped = sum(int(shard.get("existing_images_skipped", 0) or 0) for shard in shards)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timing_schema_version": 2,
        "timing_scope": "parallel_sharded_generation",
        "comparable_for_algorithm_speedup": False,
        "legacy_timing": any(int(shard.get("timing_schema_version", 0) or 0) < 2 for shard in shards),
        "worker_sampling_latency_sec": worker_sampling,
        "max_worker_sampling_latency_sec": max(worker_sampling, default=0.0),
        "sum_worker_sampling_latency_sec": sum(worker_sampling),
        "worker_end_to_end_latency_sec": worker_end_to_end,
        "parallel_orchestration_wall_time_sec": orchestration,
        "end_to_end_latency_sec": orchestration,
        "latency_sec": orchestration,
        "parallel_latency_sec": orchestration,
        "generated_images": generated_images,
        "generated_images_this_run": generated_this_run,
        "existing_images_skipped": skipped,
        "total_images_available": image_count,
        "parallel_generation_images_per_sec": (
            generated_images / orchestration if orchestration > 0 else None
        ),
        "images_per_sec": generated_images / orchestration if orchestration > 0 else None,
        "num_shards": len(shards),
        "resume": any(bool(shard.get("resume")) for shard in shards),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge generic JiT, DeCo, or PixelGen sharded generation outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", choices=("jit", "deco", "pixelgen"), default="jit")
    parser.add_argument("--launcher-meta", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _png_indices(image_dir: Path) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    invalid_names: list[str] = []
    if not image_dir.is_dir():
        return indices, invalid_names
    for path in image_dir.glob("*.png"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            invalid_names.append(path.name)
    return indices, sorted(invalid_names)


def _duplicate_indices(indices: list[int]) -> list[int]:
    return sorted(idx for idx, count in Counter(indices).items() if count > 1)


def main() -> int:
    args = build_parser().parse_args()
    manifests = [args.run_dir / f"manifest_shard{idx}.jsonl" for idx in range(args.num_shards)]
    cache_paths = [args.run_dir / f"cache_stats_shard{idx}.json" for idx in range(args.num_shards)]
    latency_paths = [args.run_dir / f"latency_shard{idx}.json" for idx in range(args.num_shards)]
    meta_paths = [args.run_dir / f"generation_meta_shard{idx}.json" for idx in range(args.num_shards)]
    missing_files = [str(path) for path in [*manifests, *cache_paths, *latency_paths, *meta_paths] if not path.is_file()]
    rows = [row for path in manifests for row in _read_manifest(path)]
    rows.sort(key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in rows]
    image_dir = args.run_dir / "images"
    image_count = count_images(image_dir)
    image_indices, invalid_image_filenames = _png_indices(image_dir)
    expected = set(range(args.expected_images))
    observed = set(indices)
    observed_images = set(image_indices)
    duplicates = _duplicate_indices(indices)
    duplicate_image_indices = _duplicate_indices(image_indices)
    missing_indices = sorted(expected - observed)
    unexpected_indices = sorted(observed - expected)
    missing_image_indices = sorted(expected - observed_images)
    unexpected_image_indices = sorted(observed_images - expected)
    warnings_list: list[str] = []
    for condition, message in (
        (bool(missing_files), "required shard files are missing"),
        (bool(duplicates), "manifest contains duplicate indices"),
        (bool(missing_indices), "manifest contains missing indices"),
        (bool(unexpected_indices), "manifest contains unexpected indices"),
        (bool(duplicate_image_indices), "PNG filenames contain duplicate numeric indices"),
        (bool(missing_image_indices), "PNG filenames contain missing indices"),
        (bool(unexpected_image_indices), "PNG filenames contain unexpected indices"),
        (bool(invalid_image_filenames), "PNG filenames contain non-numeric stems"),
    ):
        if condition:
            warnings_list.append(message)
    report = {
        "run_dir": str(args.run_dir),
        "model": args.model,
        "method": args.method,
        "num_shards": args.num_shards,
        "expected_images": args.expected_images,
        "manifest_rows": len(rows),
        "image_count": image_count,
        "missing_files": missing_files,
        "missing_indices": missing_indices[:100],
        "missing_index_count": len(missing_indices),
        "extra_indices": unexpected_indices[:100],
        "extra_index_count": len(unexpected_indices),
        "duplicate_indices": duplicates[:100],
        "duplicate_index_count": len(duplicates),
        "missing_image_indices": missing_image_indices[:100],
        "missing_image_index_count": len(missing_image_indices),
        "unexpected_image_indices": unexpected_image_indices[:100],
        "unexpected_image_index_count": len(unexpected_image_indices),
        "duplicate_image_indices": duplicate_image_indices[:100],
        "duplicate_image_index_count": len(duplicate_image_indices),
        "invalid_image_filenames": invalid_image_filenames[:100],
        "warnings": warnings_list,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.strict and warnings_list:
        raise RuntimeError(f"Shard merge strict checks failed: {report}")

    cache_shards = [_read_json(path) for path in cache_paths if path.is_file()]
    latency_shards = [_read_json(path) for path in latency_paths if path.is_file()]
    meta_shards = [_read_json(path) for path in meta_paths if path.is_file()]
    launcher_meta = _read_json(args.launcher_meta) if args.launcher_meta and args.launcher_meta.is_file() else None
    unique_rows = {int(row["index"]): row for row in rows}
    write_manifest_atomic(args.run_dir / "manifest.jsonl", list(unique_rows.values()))
    _write_json(args.run_dir / "cache_stats.json", _merge_cache_stats(cache_shards))
    _write_json(args.run_dir / "latency.json", _merge_latency(latency_shards, launcher_meta, image_count))
    merged_meta = dict(meta_shards[0]) if meta_shards else {}
    merged_meta.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "num_shards": args.num_shards,
            "method_name": args.method,
            "model": args.model,
            "shard_metas": meta_shards,
            "parallel_merge_report": report,
        }
    )
    _write_json(args.run_dir / "generation_meta.json", merged_meta)
    _write_json(args.run_dir / "parallel_merge_report.json", report)
    print(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
