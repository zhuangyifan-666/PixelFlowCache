#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import list_methods_for_model  # noqa: E402
from pfc.eval.timing import normalize_timing_payload  # noqa: E402


DEFAULT_METHODS = list_methods_for_model("jit", tags={"reference", "main_baseline", "proxy_default"})
FIELDS = [
    "method",
    "num_images",
    "steps",
    "latency_sec",
    "sampling_latency_sec",
    "end_to_end_latency_sec",
    "sampling_images_per_sec",
    "timing_scope",
    "timing_schema_version",
    "legacy_timing",
    "comparable_for_algorithm_speedup",
    "batch_size",
    "num_shards",
    "gpu_count",
    "resume",
    "images_per_sec",
    "speedup_vs_no_cache",
    "FID",
    "IS",
    "PSNR",
    "SSIM",
    "LPIPS",
    "rel_l2",
    "pair_count",
    "generated_images",
    "generated_images_this_run",
    "existing_images_skipped",
    "cache_hit_rate",
    "cache_total_calls",
    "cache_hits",
    "cache_refreshes",
    "effective_compute_saving_rate",
    "safe_reuse",
    "unsafe_refresh",
    "max_age",
    "mean_age",
    "safe_lambda",
    "safe_quantile",
    "forecast_decisions",
    "forecast_committed",
    "forecast_failures",
    "mean_effective_order",
    "taylorseer_interval",
    "taylorseer_max_order",
    "speca_full_step_decisions",
    "speca_speculative_step_decisions",
    "speca_speculative_step_ratio",
    "speca_verification_steps",
    "speca_verifier_fresh_calls",
    "speca_verification_accept_decisions",
    "speca_verification_reject_decisions",
    "speca_verification_acceptance_rate",
    "speca_completed_speculative_runs",
    "speca_mean_speculative_run_length",
    "speca_max_speculative_run_length",
    "speca_verification_error_count",
    "speca_verification_error_mean",
    "speca_verification_error_std",
    "speca_verification_error_p50",
    "speca_verification_error_p90",
    "speca_verification_error_p95",
    "speca_forecast_committed",
    "speca_forecast_failures",
    "speca_mean_effective_order",
    "speca_logical_managed_calls",
    "speca_full_compute_calls",
    "speca_actual_original_module_forwards",
    "speca_effective_skipped_block_calls",
    "speca_raw_forecast_rate",
    "speca_effective_compute_saving_rate",
    "speca_verifier_overhead_rate",
    "speca_actual_compute_fraction",
    "speca_base_threshold",
    "speca_decay_rate",
    "speca_min_threshold",
    "speca_min_forecast_steps",
    "speca_max_forecast_steps",
    "speca_max_order",
    "speca_error_metric",
    "speca_verifier_module_requested",
    "speca_verifier_module_resolved",
    "speca_branch_aggregation",
    "speca_timing_semantics",
    "dicache_probe_depth",
    "dicache_reuse_threshold",
    "dicache_error_choice",
    "dicache_branch_aggregation",
    "dicache_ret_ratio",
    "dicache_dcta_enabled",
    "dicache_gamma_min",
    "dicache_gamma_max",
    "dicache_full_steps",
    "dicache_reuse_steps",
    "dicache_reuse_step_ratio",
    "dicache_probe_block_calls",
    "dicache_deep_block_calls",
    "dicache_reference_block_calls",
    "dicache_actual_block_calls",
    "dicache_effective_skipped_block_calls",
    "dicache_effective_block_compute_saving_rate",
    "dicache_actual_block_compute_fraction",
    "dicache_share_cfg_prefix",
    "dicache_schedule_variant",
    "dicache_cfg_prefix_fairness_mode",
    "dicache_reference_cfg_prefix_calls",
    "dicache_actual_cfg_prefix_calls",
    "dicache_cfg_prefix_calls_saved",
    "dicache_decision_syncs",
    "dicache_decision_syncs_per_step",
    "dicache_observed_error_count",
    "dicache_observed_error_mean",
    "dicache_observed_error_std",
    "dicache_observed_error_p95",
    "dicache_decision_error_count",
    "dicache_decision_error_mean",
    "dicache_decision_error_std",
    "dicache_decision_error_p95",
    "dicache_dcta_branch_calls",
    "dicache_dcta_branch_fallback_calls",
    "dicache_dcta_degenerate_fallback_calls",
    "dicache_dcta_insufficient_history_fallback_calls",
    "dicache_gamma_mean",
    "dicache_gamma_p50",
    "dicache_gamma_p90",
    "dicache_gamma_clip_low_count",
    "dicache_gamma_clip_high_count",
    "dicache_peak_history_storage_bytes",
    "dicache_timing_semantics",
]


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"invalid json: {path}: {exc}")
        return {}


def _get_any(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        node: Any = payload
        ok = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok:
            return node
    return ""


def _metric_value(payload: dict[str, Any], names: list[str]) -> Any:
    if not payload:
        return ""
    candidates: list[dict[str, Any]] = [payload]
    for key in ("metrics", "results", "summary"):
        if isinstance(payload.get(key), dict):
            candidates.append(payload[key])
    for candidate in candidates:
        for name in names:
            for key in (name, name.lower(), name.upper()):
                if key in candidate:
                    value = candidate[key]
                    if isinstance(value, dict):
                        for inner in ("mean", "value", "score"):
                            if inner in value:
                                if value[inner] is None and value.get("display") is not None:
                                    return value["display"]
                                return value[inner]
                    return value
    return ""


def _safe_stats(cache_stats: dict[str, Any]) -> dict[str, Any]:
    safe_policy = cache_stats.get("safe_policy") if isinstance(cache_stats, dict) else {}
    if not isinstance(safe_policy, dict):
        safe_policy = {}
    stats = safe_policy.get("stats") if isinstance(safe_policy.get("stats"), dict) else {}
    config = safe_policy.get("config") if isinstance(safe_policy.get("config"), dict) else {}
    return {
        "safe_reuse": stats.get("safe_reuse", ""),
        "unsafe_refresh": stats.get("unsafe_refresh", ""),
        "max_age": config.get("max_age", _get_any(cache_stats, ["safe_policy.stats.max_age"])),
        "mean_age": stats.get("mean_age", ""),
        "safe_lambda": config.get("safe_lambda", ""),
        "safe_quantile": config.get("quantile", ""),
    }


def _taylorseer_stats(cache_stats: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    policy = cache_stats.get("taylorseer_policy") if isinstance(cache_stats, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    stats = policy.get("stats") if isinstance(policy.get("stats"), dict) else {}
    config = policy.get("config") if isinstance(policy.get("config"), dict) else {}
    return {
        "forecast_decisions": stats.get("forecast_decisions", ""),
        "forecast_committed": stats.get("forecast_committed", ""),
        "forecast_failures": stats.get("forecast_failures", ""),
        "mean_effective_order": stats.get("mean_effective_order", ""),
        "taylorseer_interval": config.get("interval", _get_any(meta, ["taylorseer_interval", "method.taylorseer_interval"])),
        "taylorseer_max_order": config.get("max_order", _get_any(meta, ["taylorseer_max_order", "method.taylorseer_max_order"])),
    }


def _speca_stats(cache_stats: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    policy = cache_stats.get("speca_policy") if isinstance(cache_stats, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    config = policy.get("config") if isinstance(policy.get("config"), dict) else {}
    errors = policy.get("verification_errors") if isinstance(policy.get("verification_errors"), dict) else {}
    return {
        "speca_full_step_decisions": policy.get("full_step_decisions", ""),
        "speca_speculative_step_decisions": policy.get("speculative_step_decisions", ""),
        "speca_speculative_step_ratio": policy.get("speculative_step_ratio", ""),
        "speca_verification_steps": policy.get("verification_steps", ""),
        "speca_verifier_fresh_calls": policy.get("verifier_fresh_calls", ""),
        "speca_verification_accept_decisions": policy.get("verification_accept_decisions", ""),
        "speca_verification_reject_decisions": policy.get("verification_reject_decisions", ""),
        "speca_verification_acceptance_rate": policy.get("verification_acceptance_rate", ""),
        "speca_completed_speculative_runs": policy.get("completed_speculative_runs", ""),
        "speca_mean_speculative_run_length": policy.get("mean_speculative_run_length", ""),
        "speca_max_speculative_run_length": policy.get("max_speculative_run_length", ""),
        "speca_verification_error_count": errors.get("count", ""),
        "speca_verification_error_mean": errors.get("mean", ""),
        "speca_verification_error_std": errors.get("std", ""),
        "speca_verification_error_p50": errors.get("p50", ""),
        "speca_verification_error_p90": errors.get("p90", ""),
        "speca_verification_error_p95": errors.get("p95", ""),
        "speca_forecast_committed": policy.get("forecast_committed", ""),
        "speca_forecast_failures": policy.get("forecast_failures", ""),
        "speca_mean_effective_order": policy.get("mean_effective_order", ""),
        "speca_logical_managed_calls": policy.get("logical_managed_calls", ""),
        "speca_full_compute_calls": policy.get("full_compute_calls", ""),
        "speca_actual_original_module_forwards": policy.get("actual_original_module_forwards", ""),
        "speca_effective_skipped_block_calls": policy.get("effective_skipped_block_calls", ""),
        "speca_raw_forecast_rate": policy.get("raw_forecast_rate", ""),
        "speca_effective_compute_saving_rate": policy.get("effective_compute_saving_rate", ""),
        "speca_verifier_overhead_rate": policy.get("verifier_overhead_rate", ""),
        "speca_actual_compute_fraction": policy.get("actual_compute_fraction", ""),
        "speca_base_threshold": config.get("base_threshold", _get_any(meta, ["speca_base_threshold"])),
        "speca_decay_rate": config.get("decay_rate", _get_any(meta, ["speca_decay_rate"])),
        "speca_min_threshold": config.get("min_threshold", _get_any(meta, ["speca_min_threshold"])),
        "speca_min_forecast_steps": config.get("min_forecast_steps", _get_any(meta, ["speca_min_forecast_steps"])),
        "speca_max_forecast_steps": config.get("max_forecast_steps", _get_any(meta, ["speca_max_forecast_steps"])),
        "speca_max_order": config.get("max_order", _get_any(meta, ["speca_max_order"])),
        "speca_error_metric": config.get("error_metric", _get_any(meta, ["speca_error_metric"])),
        "speca_verifier_module_requested": config.get(
            "verifier_module_requested",
            _get_any(meta, ["speca_verifier_module_requested"]),
        ),
        "speca_verifier_module_resolved": config.get(
            "verifier_module_resolved",
            _get_any(meta, ["speca_verifier_module_resolved"]),
        ),
        "speca_branch_aggregation": config.get(
            "branch_aggregation",
            _get_any(meta, ["speca_branch_aggregation"]),
        ),
        "speca_timing_semantics": policy.get(
            "timing_semantics",
            config.get("timing_semantics", ""),
        ),
    }


def _dicache_stats(cache_stats: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    policy = cache_stats.get("dicache_policy") if isinstance(cache_stats, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    config = policy.get("config") if isinstance(policy.get("config"), dict) else {}
    observed = (
        policy.get("observed_branch_error_stats")
        if isinstance(policy.get("observed_branch_error_stats"), dict)
        else {}
    )
    decision = (
        policy.get("decision_branch_error_stats")
        if isinstance(policy.get("decision_branch_error_stats"), dict)
        else {}
    )
    gamma = policy.get("gamma_stats") if isinstance(policy.get("gamma_stats"), dict) else {}

    def configured(name: str) -> Any:
        return config.get(name, _get_any(meta, [name, f"dicache_cache.{name}"]))

    return {
        "dicache_probe_depth": policy.get("probe_depth", configured("probe_depth")),
        "dicache_reuse_threshold": policy.get("reuse_threshold", configured("reuse_threshold")),
        "dicache_error_choice": policy.get("error_choice", configured("error_choice")),
        "dicache_branch_aggregation": policy.get("branch_aggregation", configured("branch_aggregation")),
        "dicache_ret_ratio": configured("ret_ratio"),
        "dicache_dcta_enabled": policy.get("dcta_enabled", configured("dcta_enabled")),
        "dicache_gamma_min": configured("gamma_min"),
        "dicache_gamma_max": configured("gamma_max"),
        "dicache_full_steps": policy.get("full_step_decisions", ""),
        "dicache_reuse_steps": policy.get("reuse_step_decisions", ""),
        "dicache_reuse_step_ratio": policy.get("reuse_step_ratio", ""),
        "dicache_probe_block_calls": policy.get("probe_block_calls", ""),
        "dicache_deep_block_calls": policy.get("deep_block_calls", ""),
        "dicache_reference_block_calls": policy.get("reference_block_calls", ""),
        "dicache_actual_block_calls": policy.get("actual_block_calls", ""),
        "dicache_effective_skipped_block_calls": policy.get("effective_skipped_block_calls", ""),
        "dicache_effective_block_compute_saving_rate": policy.get(
            "effective_block_compute_saving_rate", ""
        ),
        "dicache_actual_block_compute_fraction": policy.get("actual_block_compute_fraction", ""),
        "dicache_share_cfg_prefix": policy.get(
            "share_cfg_prefix", configured("share_cfg_prefix")
        ),
        "dicache_schedule_variant": policy.get(
            "schedule_variant", configured("schedule_variant")
        ),
        "dicache_cfg_prefix_fairness_mode": _get_any(
            meta,
            ["cfg_prefix_fairness_mode", "dicache_cache.cfg_prefix_fairness_mode"],
        ),
        "dicache_reference_cfg_prefix_calls": policy.get("reference_cfg_prefix_calls", ""),
        "dicache_actual_cfg_prefix_calls": policy.get("actual_cfg_prefix_calls", ""),
        "dicache_cfg_prefix_calls_saved": policy.get("cfg_prefix_calls_saved", ""),
        "dicache_decision_syncs": policy.get("decision_device_to_host_syncs", ""),
        "dicache_decision_syncs_per_step": policy.get("decision_syncs_per_step", ""),
        "dicache_observed_error_count": observed.get("count", ""),
        "dicache_observed_error_mean": observed.get("mean", ""),
        "dicache_observed_error_std": observed.get("std", ""),
        "dicache_observed_error_p95": observed.get("p95", ""),
        "dicache_decision_error_count": decision.get("count", ""),
        "dicache_decision_error_mean": decision.get("mean", ""),
        "dicache_decision_error_std": decision.get("std", ""),
        "dicache_decision_error_p95": decision.get("p95", ""),
        "dicache_dcta_branch_calls": policy.get("dcta_branch_calls", ""),
        "dicache_dcta_branch_fallback_calls": policy.get("dcta_branch_fallback_calls", ""),
        "dicache_dcta_degenerate_fallback_calls": policy.get(
            "dcta_branch_degenerate_fallback_calls", ""
        ),
        "dicache_dcta_insufficient_history_fallback_calls": policy.get(
            "dcta_branch_insufficient_history_fallback_calls", ""
        ),
        "dicache_gamma_mean": gamma.get("mean", ""),
        "dicache_gamma_p50": gamma.get("p50", ""),
        "dicache_gamma_p90": gamma.get("p90", ""),
        "dicache_gamma_clip_low_count": policy.get("gamma_clip_low_count", ""),
        "dicache_gamma_clip_high_count": policy.get("gamma_clip_high_count", ""),
        "dicache_peak_history_storage_bytes": policy.get("peak_history_storage_bytes", ""),
        "dicache_timing_semantics": policy.get("timing_semantics", ""),
    }


def _row_for_method(args: argparse.Namespace, method: str, warnings: list[str]) -> dict[str, Any]:
    run_dir = Path(args.output_root) / "jit" / args.run_id / method
    fid_dir = Path(args.fid_root) / args.run_id / "jit" / method
    pair_dir = Path(args.pair_root) / args.run_id / "jit" / method
    meta = _read_json(run_dir / "generation_meta.json", warnings)
    latency = _read_json(run_dir / "latency.json", warnings)
    timing = normalize_timing_payload(latency)
    cache_stats = _read_json(run_dir / "cache_stats.json", warnings)
    fid = _read_json(fid_dir / "fid_results.json", warnings)
    pair = {} if method == "no_cache_50" else _read_json(pair_dir / "pair_metrics.json", warnings)
    safe = _safe_stats(cache_stats)
    taylorseer = _taylorseer_stats(cache_stats, meta)
    speca = _speca_stats(cache_stats, meta)
    dicache = _dicache_stats(cache_stats, meta)
    num_images = _get_any(meta, ["num_images"]) or _get_any(latency, ["requested_images", "generated_images"])
    generated_images = _get_any(latency, ["generated_images", "total_images_available"])
    generated_this_run = _get_any(latency, ["generated_images_this_run"])
    skipped = _get_any(latency, ["existing_images_skipped"])
    if num_images != "" and generated_images != "" and int(generated_images) != int(num_images):
        warnings.append(f"{method}: generated_images={generated_images} differs from num_images={num_images}")
    if skipped not in ("", 0, None):
        warnings.append(f"{method}: resume skipped existing images; latency/speedup is not comparable")
    row = {
        "method": method,
        "num_images": num_images,
        "steps": _get_any(meta, ["eval_steps", "steps"]),
        "latency_sec": _get_any(timing, ["end_to_end_latency_sec", "latency_sec"]),
        "sampling_latency_sec": _get_any(timing, ["sampling_latency_sec"]),
        "end_to_end_latency_sec": _get_any(timing, ["end_to_end_latency_sec"]),
        "sampling_images_per_sec": _get_any(timing, ["sampling_images_per_sec"]),
        "timing_scope": _get_any(timing, ["timing_scope"]),
        "timing_schema_version": _get_any(timing, ["timing_schema_version"]),
        "legacy_timing": bool(timing.get("legacy_timing", False)),
        "comparable_for_algorithm_speedup": bool(
            timing.get("comparable_for_algorithm_speedup", False)
        ),
        "batch_size": _get_any(meta, ["batch_size"]),
        "num_shards": _get_any(timing, ["num_shards"]) or _get_any(meta, ["num_shards"]),
        "gpu_count": _get_any(meta, ["provenance.gpu_count"]) or _get_any(timing, ["num_shards"]),
        "resume": bool(timing.get("resume", meta.get("resume", False))),
        "images_per_sec": _get_any(timing, ["sampling_images_per_sec", "images_per_sec"]),
        "generated_images": generated_images,
        "generated_images_this_run": generated_this_run,
        "existing_images_skipped": skipped,
        "speedup_vs_no_cache": "",
        "FID": _metric_value(fid, ["fid", "FID", "frechet_inception_distance"]),
        "IS": _metric_value(fid, ["is", "IS", "inception_score", "inception_score_mean"]),
        "PSNR": _metric_value(pair, ["psnr", "PSNR"]),
        "SSIM": _metric_value(pair, ["ssim", "SSIM"]),
        "LPIPS": _metric_value(pair, ["lpips", "LPIPS"]),
        "rel_l2": _metric_value(pair, ["rel_l2", "relative_l2"]),
        "pair_count": _get_any(pair, ["pair_count", "num_pairs"]),
        "cache_hit_rate": _get_any(cache_stats, ["hit_rate", "cache_hit_rate"]),
        "cache_total_calls": _get_any(cache_stats, ["total_calls", "cache_total_calls"]),
        "cache_hits": _get_any(cache_stats, ["hits", "cache_hits"]),
        "cache_refreshes": _get_any(cache_stats, ["refreshes", "cache_refreshes"]),
        **safe,
        **taylorseer,
        **speca,
        **dicache,
    }
    if method == "speca_style":
        row["forecast_committed"] = speca["speca_forecast_committed"]
        row["forecast_failures"] = speca["speca_forecast_failures"]
        row["mean_effective_order"] = speca["speca_mean_effective_order"]
        row["effective_compute_saving_rate"] = speca[
            "speca_effective_compute_saving_rate"
        ]
    elif method == "dicache_style":
        row["effective_compute_saving_rate"] = dicache[
            "dicache_effective_block_compute_saving_rate"
        ]
    else:
        row["effective_compute_saving_rate"] = _get_any(
            cache_stats,
            ["effective_compute_saving_rate"],
        )
    return row


def _as_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _timing_mismatch_reasons(
    baseline: dict[str, Any],
    method: dict[str, Any],
) -> list[str]:
    """Return every reason that makes a sampling-speed comparison unfair."""

    reasons: list[str] = []
    for label, row in (("baseline", baseline), ("method", method)):
        if bool(row.get("legacy_timing")):
            reasons.append(f"{label} uses legacy timing")
        if not bool(row.get("comparable_for_algorithm_speedup")):
            reasons.append(f"{label} timing is marked non-comparable")
        if bool(row.get("resume")):
            reasons.append(f"{label} is a resume run")
        if _as_float(row.get("sampling_latency_sec")) is None:
            reasons.append(f"{label} has no schema-v2 sampling latency")

    comparable_fields = (
        ("gpu_count", "GPU count"),
        ("batch_size", "batch size"),
        ("num_images", "image count"),
        ("timing_scope", "timing scope"),
    )
    for key, label in comparable_fields:
        base_value = baseline.get(key, "")
        method_value = method.get(key, "")
        if base_value in ("", None) or method_value in ("", None):
            reasons.append(f"missing {label}")
        elif str(base_value) != str(method_value):
            reasons.append(f"{label} differs ({base_value!r} vs {method_value!r})")
    return list(dict.fromkeys(reasons))


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "warnings": warnings}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# JiT Safe-BFC 1000-Image Proxy Summary",
        "",
        "These are 1000-image proxy results and should not be interpreted as final 50k FID/IS.",
        "If resume skipped existing images, latency/speedup is not comparable.",
        "Runtime cache hit rate is a raw forecast-return rate. For SpeCa, effective compute saving subtracts verifier fresh calls.",
        "Overall synchronized sampler latency, not per-operation host dispatch timing, is the final performance measure.",
        "DiCache block-call saving includes probe-block cost and is a proxy, not exact FLOPs.",
        "Four-card wall-clock is throughput orchestration, not an algorithmic single-card speedup.",
        "The adapted JiT DiCache schedule is batch-level shared CFG; final speed claims require single-card end-to-end latency.",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.append("| method | FID | IS | PSNR | SSIM | LPIPS | images/sec | speedup | raw cache hit | effective compute saving | safe reuse | forecast committed | mean order |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {method} | {FID} | {IS} | {PSNR} | {SSIM} | {LPIPS} | {images_per_sec} | "
            "{speedup_vs_no_cache} | {cache_hit_rate} | {effective_compute_saving_rate} | "
            "{safe_reuse} | {forecast_committed} | {mean_effective_order} |".format(**row)
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect JiT Safe-BFC 1000-image proxy results.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/stage4a/full_generation")
    parser.add_argument("--fid-root", default="logs/stage5a/fid")
    parser.add_argument("--pair-root", default="logs/stage5a/pair_metrics")
    parser.add_argument("--out-dir")
    parser.add_argument("--methods")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()] if args.methods else DEFAULT_METHODS
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs/stage5a/summary") / args.run_id
    warnings: list[str] = []
    rows = [_row_for_method(args, method, warnings) for method in methods]
    baseline = next((row for row in rows if row["method"] == "no_cache_50"), None)
    for row in rows:
        row["speedup_vs_no_cache"] = ""
        if baseline is None:
            warnings.append(f"{row['method']}: no no_cache_50 baseline for timing comparison")
            continue
        reasons = _timing_mismatch_reasons(baseline, row)
        baseline_latency = _as_float(baseline.get("sampling_latency_sec"))
        latency = _as_float(row.get("sampling_latency_sec"))
        if reasons or baseline_latency is None or latency is None or latency <= 0.0:
            if row["method"] != "no_cache_50":
                warnings.append(
                    f"{row['method']}: algorithm speedup unavailable: "
                    + "; ".join(reasons or ["missing schema-v2 sampling latency"])
                )
            continue
        row["speedup_vs_no_cache"] = baseline_latency / latency
    _write_outputs(out_dir, rows, warnings)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
