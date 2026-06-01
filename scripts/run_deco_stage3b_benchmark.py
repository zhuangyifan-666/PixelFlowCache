#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.deco_stage3b_common import (  # noqa: E402
    DeCoStage3BConfig,
    compare_outputs,
    default_deco_config,
    default_deco_dir,
    detect_deco_ckpt,
    env_float,
    env_int,
    env_optional_float,
    json_config,
    make_inputs,
    make_run_id,
    parse_int_list,
    parse_str_list,
    run_cached,
    run_no_cache,
    write_common_meta,
    write_csv,
    write_json,
)


BENCHMARK_FIELDNAMES = [
    "method_type",
    "method_name",
    "seed",
    "num_samples",
    "reference_steps",
    "eval_steps",
    "cache_units",
    "cache_interval",
    "active_t_min",
    "active_t_max",
    "latency_median_sec",
    "speedup_vs_reference",
    "cache_hit_rate",
    "same_seed_rel_l2",
    "same_seed_mse",
    "same_seed_mae",
    "same_seed_psnr",
    "low_freq_delta_ratio",
    "mid_freq_delta_ratio",
    "high_freq_delta_ratio",
    "wrapped_module_count",
    "run_dir",
]

AGGREGATE_FIELDNAMES = [
    "method_type",
    "method_name",
    "seed_count",
    "num_samples",
    "reference_steps",
    "eval_steps",
    "cache_units",
    "cache_interval",
    "active_t_min",
    "active_t_max",
    "speedup_mean",
    "speedup_std",
    "rel_l2_mean",
    "rel_l2_std",
    "mse_mean",
    "mse_std",
    "psnr_mean",
    "psnr_std",
    "hit_rate_mean",
    "hit_rate_std",
]


DEFAULT_CACHE_METHODS = [
    ("backbone_i2_t02_10", "backbone_blocks", 2, 0.2, 1.0),
    ("backbone_i2_t02_08", "backbone_blocks", 2, 0.2, 0.8),
    ("decoder_i2_t02_10", "decoder_blocks", 2, 0.2, 1.0),
    ("all_candidates_i2_t02_10", "all_candidates", 2, 0.2, 1.0),
    ("final_i2_t02_10", "final", 2, 0.2, 1.0),
]


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_name"]), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for method_name in sorted(grouped):
        group = grouped[method_name]
        first = group[0]
        speedup_mean, speedup_std = mean_std([float(row["speedup_vs_reference"]) for row in group])
        rel_l2_mean, rel_l2_std = mean_std([float(row["same_seed_rel_l2"]) for row in group])
        mse_mean, mse_std = mean_std([float(row["same_seed_mse"]) for row in group])
        psnr_values = [float(row["same_seed_psnr"]) for row in group if str(row["same_seed_psnr"]) != "inf"]
        psnr_mean, psnr_std = mean_std(psnr_values)
        hit_rate_mean, hit_rate_std = mean_std([float(row["cache_hit_rate"]) for row in group])
        aggregates.append(
            {
                "method_type": first["method_type"],
                "method_name": method_name,
                "seed_count": len(group),
                "num_samples": first["num_samples"],
                "reference_steps": first["reference_steps"],
                "eval_steps": first["eval_steps"],
                "cache_units": first["cache_units"],
                "cache_interval": first["cache_interval"],
                "active_t_min": first["active_t_min"],
                "active_t_max": first["active_t_max"],
                "speedup_mean": speedup_mean,
                "speedup_std": speedup_std,
                "rel_l2_mean": rel_l2_mean,
                "rel_l2_std": rel_l2_std,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "psnr_mean": psnr_mean,
                "psnr_std": psnr_std,
                "hit_rate_mean": hit_rate_mean,
                "hit_rate_std": hit_rate_std,
            }
        )
    return aggregates


def build_base_config(run_id: str, run_dir: Path, seed: int, reference_steps: int) -> DeCoStage3BConfig:
    return DeCoStage3BConfig(
        deco_dir=default_deco_dir(),
        ckpt_path=detect_deco_ckpt(),
        config_path=default_deco_config(),
        run_id=run_id,
        run_dir=run_dir,
        num_samples=env_int("PFC_STAGE3B_NUM_SAMPLES", 8),
        batch_size=env_int("PFC_STAGE3B_BATCH_SIZE", 4),
        steps=reference_steps,
        seed=seed,
        cfg=env_float("PFC_STAGE3B_CFG", 3.2),
        cfg_interval_min=env_float("PFC_STAGE3B_CFG_INTERVAL_MIN", 0.1),
        cfg_interval_max=env_float("PFC_STAGE3B_CFG_INTERVAL_MAX", 1.0),
        cache_interval=env_int("PFC_STAGE3B_CACHE_INTERVAL", 2),
        active_t_min=env_optional_float("PFC_STAGE3B_ACTIVE_T_MIN", 0.2),
        active_t_max=env_optional_float("PFC_STAGE3B_ACTIVE_T_MAX", 1.0),
        timing_repeats=env_int("PFC_STAGE3B_TIMING_REPEATS", 2),
        warmup_runs=env_int("PFC_STAGE3B_WARMUP_RUNS", 1),
        resolution=env_int("PFC_STAGE3B_RESOLUTION", 256),
        save_diagnostics=False,
    )


def _comparison_fields(reference_output: torch.Tensor, output: torch.Tensor, reference_latency: float, method_latency: float) -> dict[str, Any]:
    comparison = compare_outputs(reference_output, output, reference_latency, method_latency)
    frequency_delta = comparison.get("frequency_delta_bands") or {}
    return {
        "speedup_vs_reference": comparison["speedup"],
        "same_seed_rel_l2": comparison["same_seed_rel_l2"],
        "same_seed_mse": comparison["same_seed_mse"],
        "same_seed_mae": comparison["same_seed_mae"],
        "same_seed_psnr": comparison["same_seed_psnr"],
        "low_freq_delta_ratio": frequency_delta.get("low_ratio"),
        "mid_freq_delta_ratio": frequency_delta.get("mid_ratio"),
        "high_freq_delta_ratio": frequency_delta.get("high_ratio"),
    }


def _row(
    method_type: str,
    method_name: str,
    config: DeCoStage3BConfig,
    reference_steps: int,
    latency_median_sec: float,
    cache_hit_rate: float,
    comparison_fields: dict[str, Any],
    wrapped_module_count: int = 0,
) -> dict[str, Any]:
    return {
        "method_type": method_type,
        "method_name": method_name,
        "seed": config.seed,
        "num_samples": config.num_samples,
        "reference_steps": reference_steps,
        "eval_steps": config.steps,
        "cache_units": config.cache_units if method_type == "cache" else "none",
        "cache_interval": config.cache_interval if method_type == "cache" else 1,
        "active_t_min": config.active_t_min if method_type == "cache" else None,
        "active_t_max": config.active_t_max if method_type == "cache" else None,
        "latency_median_sec": latency_median_sec,
        "cache_hit_rate": cache_hit_rate,
        "wrapped_module_count": wrapped_module_count,
        "run_dir": str(config.run_dir),
        **comparison_fields,
    }


def write_summary(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DeCo Stage 3B Direct-Velocity Cache Benchmark",
        "",
        "| method | type | speedup mean | rel-L2 mean | rel-L2 std | hit rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {method_name} | {method_type} | {speedup_mean:.4f} | {rel_l2_mean:.6f} | {rel_l2_std:.6f} | {hit_rate_mean:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(run_id: str, run_dir: Path, seeds: list[int], reference_steps: int, reduced_steps: list[int]) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        base = build_base_config(run_id, run_dir, seed, reference_steps)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels, uncondition, noise = make_inputs(base, device)
        reference_timing = run_no_cache(base, labels, uncondition, noise, device)
        reference_output = reference_timing["output"]
        reference_latency = float(reference_timing["latency_median_sec"])
        rows.append(
            _row(
                "reference",
                "no_cache",
                base,
                reference_steps,
                reference_latency,
                0.0,
                {
                    "speedup_vs_reference": 1.0,
                    "same_seed_rel_l2": 0.0,
                    "same_seed_mse": 0.0,
                    "same_seed_mae": 0.0,
                    "same_seed_psnr": float("inf"),
                    "low_freq_delta_ratio": None,
                    "mid_freq_delta_ratio": None,
                    "high_freq_delta_ratio": None,
                },
            )
        )
        for method_name, cache_units, interval, active_t_min, active_t_max in DEFAULT_CACHE_METHODS:
            config = replace(
                base,
                cache_units=cache_units,
                cache_interval=interval,
                active_t_min=active_t_min,
                active_t_max=active_t_max,
                run_id=f"{run_id}_{method_name}_seed{seed}",
            )
            timing, cache_stats, wrapped_modules = run_cached(config, labels, uncondition, noise, device)
            comparison = _comparison_fields(reference_output, timing["output"], reference_latency, float(timing["latency_median_sec"]))
            rows.append(
                _row(
                    "cache",
                    method_name,
                    config,
                    reference_steps,
                    float(timing["latency_median_sec"]),
                    float(cache_stats["hit_rate"]),
                    comparison,
                    wrapped_module_count=len(wrapped_modules),
                )
            )
        for steps in reduced_steps:
            config = replace(base, steps=steps, cache_units="none", run_id=f"{run_id}_reduced{steps}_seed{seed}")
            timing = run_no_cache(config, labels, uncondition, noise, device)
            comparison = _comparison_fields(reference_output, timing["output"], reference_latency, float(timing["latency_median_sec"]))
            rows.append(
                _row(
                    "reduced_steps",
                    f"reduced_steps_{steps}",
                    config,
                    reference_steps,
                    float(timing["latency_median_sec"]),
                    0.0,
                    comparison,
                )
            )
    write_csv(run_dir / "benchmark_results.csv", rows, BENCHMARK_FIELDNAMES)
    write_json(run_dir / "benchmark_results.json", {"rows": rows})
    aggregate_rows = aggregate_benchmark_rows(rows)
    write_csv(run_dir / "benchmark_aggregate.csv", aggregate_rows, AGGREGATE_FIELDNAMES)
    write_summary(run_dir / "summary.md", aggregate_rows)
    config_for_meta = build_base_config(run_id, run_dir, seeds[0], reference_steps)
    write_common_meta(
        config_for_meta,
        "scripts/run_deco_stage3b_benchmark.py",
        extra={"seeds": seeds, "reference_steps": reference_steps, "reduced_steps": reduced_steps},
    )
    write_json(run_dir / "config.json", json_config(config_for_meta))
    return aggregate_rows


def main() -> int:
    reference_steps = env_int("PFC_STAGE3B_REFERENCE_STEPS", env_int("PFC_STAGE3B_STEPS", 20))
    seeds = parse_int_list(os.environ.get("PFC_STAGE3B_SEEDS", "0"))
    reduced_steps = parse_int_list(os.environ.get("PFC_STAGE3B_REDUCED_STEPS", "12,15,18"))
    run_id = os.environ.get("PFC_STAGE3B_BENCHMARK_RUN_ID", make_run_id(seeds[0], reference_steps, "benchmark"))
    run_dir = Path(os.environ.get("PFC_STAGE3B_BENCHMARK_DIR", ROOT / "logs/stage3b/deco_benchmark" / run_id)).resolve()
    aggregate_rows = run_benchmark(run_id, run_dir, seeds, reference_steps, reduced_steps)
    print(f"DeCo Stage 3B benchmark run dir: {run_dir}")
    print(json.dumps(aggregate_rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
