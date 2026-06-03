from __future__ import annotations

import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from scripts.deco_stage3b_common import (
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
    run_cached,
    run_no_cache,
    write_common_meta,
    write_csv,
    write_json,
)


STAGE3B2_FIELDNAMES = [
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
    "wrapped_modules",
    "has_final_cache",
    "has_backbone_cache",
    "has_decoder_cache",
    "run_dir",
]

STAGE3B2_AGGREGATE_FIELDNAMES = [
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
    "has_final_cache",
    "has_backbone_cache",
    "has_decoder_cache",
]

DECOMPOSITION_CACHE_METHODS = [
    ("backbone_only", "backbone_only", 2, 0.2, 1.0),
    ("final_only", "final_only", 2, 0.2, 1.0),
    ("decoder_only_no_final", "decoder_only_no_final", 2, 0.2, 1.0),
    ("decoder_plus_final", "decoder_plus_final", 2, 0.2, 1.0),
    ("backbone_plus_final", "backbone_plus_final", 2, 0.2, 1.0),
    ("backbone_plus_decoder_no_final", "backbone_plus_decoder_no_final", 2, 0.2, 1.0),
    ("all_candidates", "all_candidates", 2, 0.2, 1.0),
    ("late_backbone_only_6", "late_backbone_only:6", 2, 0.2, 1.0),
    ("late_backbone_plus_final_6", "late_backbone_plus_final:6", 2, 0.2, 1.0),
]

VALIDATION_CACHE_METHODS = [
    ("all_candidates", "all_candidates", 2, 0.2, 1.0),
    ("backbone_plus_final", "backbone_plus_final", 2, 0.2, 1.0),
    ("final_only", "final_only", 2, 0.2, 1.0),
    ("backbone_only", "backbone_only", 2, 0.2, 1.0),
    ("decoder_plus_final", "decoder_plus_final", 2, 0.2, 1.0),
]

SEED_SWEEP_CACHE_METHODS = [
    ("all_candidates", "all_candidates", 2, 0.2, 1.0),
    ("backbone_plus_final", "backbone_plus_final", 2, 0.2, 1.0),
    ("final_only", "final_only", 2, 0.2, 1.0),
    ("backbone_only", "backbone_only", 2, 0.2, 1.0),
]


def env_bool(name: str, default: bool) -> bool:
    value = __import__("os").environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def build_base_config(
    run_id: str,
    run_dir: Path,
    seed: int,
    reference_steps: int,
    env_prefix: str,
    default_num_samples: int,
    default_batch_size: int,
    default_save_diagnostics: bool,
) -> DeCoStage3BConfig:
    return DeCoStage3BConfig(
        deco_dir=default_deco_dir(),
        ckpt_path=detect_deco_ckpt(),
        config_path=default_deco_config(),
        run_id=run_id,
        run_dir=run_dir,
        num_samples=env_int(f"{env_prefix}_NUM_SAMPLES", default_num_samples),
        batch_size=env_int(f"{env_prefix}_BATCH_SIZE", default_batch_size),
        steps=reference_steps,
        seed=seed,
        cfg=env_float(f"{env_prefix}_CFG", 3.2),
        cfg_interval_min=env_float(f"{env_prefix}_CFG_INTERVAL_MIN", 0.1),
        cfg_interval_max=env_float(f"{env_prefix}_CFG_INTERVAL_MAX", 1.0),
        cache_interval=env_int(f"{env_prefix}_CACHE_INTERVAL", 2),
        active_t_min=env_optional_float(f"{env_prefix}_ACTIVE_T_MIN", 0.2),
        active_t_max=env_optional_float(f"{env_prefix}_ACTIVE_T_MAX", 1.0),
        timing_repeats=env_int(f"{env_prefix}_TIMING_REPEATS", 2),
        warmup_runs=env_int(f"{env_prefix}_WARMUP_RUNS", 1),
        resolution=env_int(f"{env_prefix}_RESOLUTION", 256),
        save_diagnostics=env_bool(f"{env_prefix}_SAVE_DIAGNOSTICS", default_save_diagnostics),
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_name"]), []).append(row)
    output: list[dict[str, Any]] = []
    for method_name in sorted(grouped):
        group = grouped[method_name]
        first = group[0]
        speedup_mean, speedup_std = mean_std([float(row["speedup_vs_reference"]) for row in group])
        rel_l2_mean, rel_l2_std = mean_std([float(row["same_seed_rel_l2"]) for row in group])
        mse_mean, mse_std = mean_std([float(row["same_seed_mse"]) for row in group])
        psnr_values = [float(row["same_seed_psnr"]) for row in group if str(row["same_seed_psnr"]) != "inf"]
        psnr_mean, psnr_std = mean_std(psnr_values)
        hit_rate_mean, hit_rate_std = mean_std([float(row["cache_hit_rate"]) for row in group])
        output.append(
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
                "has_final_cache": first["has_final_cache"],
                "has_backbone_cache": first["has_backbone_cache"],
                "has_decoder_cache": first["has_decoder_cache"],
            }
        )
    return output


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def run_stage3b2_matrix(
    *,
    run_id: str,
    root_dir: Path,
    script_name: str,
    seeds: list[int],
    reference_steps: int,
    cache_methods: list[tuple[str, str, int, float, float]],
    reduced_steps: list[int],
    env_prefix: str,
    default_num_samples: int,
    default_batch_size: int,
    default_save_diagnostics: bool,
) -> list[dict[str, Any]]:
    root_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        base = build_base_config(
            run_id,
            root_dir,
            seed,
            reference_steps,
            env_prefix,
            default_num_samples,
            default_batch_size,
            default_save_diagnostics,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels, uncondition, noise = make_inputs(base, device)
        reference_timing = run_no_cache(base, labels, uncondition, noise, device)
        reference_output = reference_timing["output"]
        reference_latency = float(reference_timing["latency_median_sec"])
        reference_summary = summary_without_output(reference_timing, "no_cache", base)
        reference_dir = root_dir / "runs" / f"no_cache_seed{seed}"
        write_method_artifacts(reference_dir, base, reference_summary, reference_summary, {}, {}, [])
        rows.append(reference_row(base, reference_steps, reference_latency, reference_dir))

        for method_name, cache_units, interval, active_t_min, active_t_max in cache_methods:
            method_dir = root_dir / "runs" / f"{method_name}_seed{seed}"
            config = replace(
                base,
                run_id=f"{run_id}_{method_name}_seed{seed}",
                run_dir=method_dir,
                cache_units=cache_units,
                cache_interval=interval,
                active_t_min=active_t_min,
                active_t_max=active_t_max,
            )
            timing, cache_stats, wrapped_modules = run_cached(config, labels, uncondition, noise, device)
            comparison = compare_fields(reference_output, timing["output"], reference_latency, float(timing["latency_median_sec"]))
            cache_summary = summary_without_output(timing, "cache", config)
            cache_summary.update(
                {
                    "cache_units": cache_units,
                    "cache_hit_rate": cache_stats["hit_rate"],
                    "wrapped_modules": wrapped_modules,
                }
            )
            write_method_artifacts(method_dir, config, reference_summary, cache_summary, comparison, cache_stats, wrapped_modules)
            rows.append(
                result_row(
                    "cache",
                    method_name,
                    config,
                    reference_steps,
                    float(timing["latency_median_sec"]),
                    float(cache_stats["hit_rate"]),
                    comparison,
                    wrapped_modules,
                    method_dir,
                )
            )

        for steps in reduced_steps:
            method_name = f"reduced_steps_{steps}"
            method_dir = root_dir / "runs" / f"{method_name}_seed{seed}"
            config = replace(
                base,
                run_id=f"{run_id}_{method_name}_seed{seed}",
                run_dir=method_dir,
                steps=steps,
                cache_units="none",
                save_diagnostics=False,
            )
            timing = run_no_cache(config, labels, uncondition, noise, device)
            comparison = compare_fields(reference_output, timing["output"], reference_latency, float(timing["latency_median_sec"]))
            reduced_summary = summary_without_output(timing, "reduced_steps", config)
            write_method_artifacts(method_dir, config, reference_summary, reduced_summary, comparison, {}, [])
            rows.append(
                result_row(
                    "reduced_steps",
                    method_name,
                    config,
                    reference_steps,
                    float(timing["latency_median_sec"]),
                    0.0,
                    comparison,
                    [],
                    method_dir,
                )
            )

    write_common_meta(
        build_base_config(
            run_id,
            root_dir,
            seeds[0],
            reference_steps,
            env_prefix,
            default_num_samples,
            default_batch_size,
            default_save_diagnostics,
        ),
        script_name,
        extra={"seeds": seeds, "reference_steps": reference_steps, "reduced_steps": reduced_steps},
    )
    write_json(root_dir / "config.json", {"run_id": run_id, "seeds": seeds, "reference_steps": reference_steps})
    return rows


def summary_without_output(timing: dict[str, Any], mode: str, config: DeCoStage3BConfig) -> dict[str, Any]:
    summary = {key: value for key, value in timing.items() if key != "output"}
    summary.update({"mode": mode, "num_samples": config.num_samples, "steps": config.steps})
    return summary


def compare_fields(reference_output: torch.Tensor, output: torch.Tensor, reference_latency: float, method_latency: float) -> dict[str, Any]:
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


def reference_row(config: DeCoStage3BConfig, reference_steps: int, reference_latency: float, run_dir: Path) -> dict[str, Any]:
    return result_row(
        "reference",
        "no_cache",
        config,
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
        [],
        run_dir,
    )


def result_row(
    method_type: str,
    method_name: str,
    config: DeCoStage3BConfig,
    reference_steps: int,
    latency_median_sec: float,
    cache_hit_rate: float,
    comparison: dict[str, Any],
    wrapped_modules: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    flags = cache_flags(wrapped_modules)
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
        "wrapped_module_count": len(wrapped_modules),
        "wrapped_modules": ";".join(wrapped_modules),
        "has_final_cache": flags["has_final_cache"],
        "has_backbone_cache": flags["has_backbone_cache"],
        "has_decoder_cache": flags["has_decoder_cache"],
        "run_dir": str(run_dir),
        **comparison,
    }


def cache_flags(wrapped_modules: list[str]) -> dict[str, bool]:
    return {
        "has_final_cache": any(name == "dec_net.final_layer" or name.endswith(".final_layer") for name in wrapped_modules),
        "has_backbone_cache": any(name.startswith("blocks.") for name in wrapped_modules),
        "has_decoder_cache": any(name.startswith("dec_net.res_blocks.") for name in wrapped_modules),
    }


def write_method_artifacts(
    run_dir: Path,
    config: DeCoStage3BConfig,
    no_cache_summary: dict[str, Any],
    method_summary: dict[str, Any],
    comparison: dict[str, Any],
    cache_stats: dict[str, Any],
    wrapped_modules: list[str],
) -> None:
    write_json(run_dir / "config.json", json_config(replace(config, run_dir=run_dir)))
    write_json(run_dir / "no_cache_summary.json", no_cache_summary)
    write_json(run_dir / "cache_summary.json", {**method_summary, "wrapped_modules": wrapped_modules})
    write_json(run_dir / "comparison.json", comparison)
    write_json(run_dir / "cache_stats.json", cache_stats)


def write_matrix_outputs(
    run_dir: Path,
    rows: list[dict[str, Any]],
    result_csv_name: str,
    result_json_name: str,
    aggregate_csv_name: str,
    summary_name: str,
    title: str,
) -> list[dict[str, Any]]:
    write_csv(run_dir / result_csv_name, rows, STAGE3B2_FIELDNAMES)
    write_json(run_dir / result_json_name, {"rows": rows})
    aggregate = aggregate_rows(rows)
    write_csv(run_dir / aggregate_csv_name, aggregate, STAGE3B2_AGGREGATE_FIELDNAMES)
    write_summary(run_dir / summary_name, aggregate, title)
    return aggregate


def write_summary(path: Path, aggregate_rows_: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| method | type | speedup mean | rel-L2 mean | hit rate | final | backbone | decoder |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in aggregate_rows_:
        lines.append(
            "| {method_name} | {method_type} | {speedup_mean:.4f} | {rel_l2_mean:.6f} | {hit_rate_mean:.4f} | {has_final_cache} | {has_backbone_cache} | {has_decoder_cache} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
