#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.cache.backbone_cache_presets import (  # noqa: E402
    BackboneCachePreset,
    get_jit_backbone_cache_presets,
    preset_to_config_dict,
    preset_to_policy_kwargs,
)
from pfc.cache.cache_state import RuntimeCacheState  # noqa: E402
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy  # noqa: E402
from pfc.cache.wrap import parse_layer_list, wrap_jit_blocks  # noqa: E402
from scripts.run_jit_stage2b_cache import (  # noqa: E402
    Stage2BConfig,
    _compare_outputs,
    _detect_jit_ckpt_dir,
    _load_jit_model,
    _make_inputs,
    _time_repeats,
)


BENCHMARK_FIELDNAMES = [
    "method_type",
    "method_name",
    "seed",
    "num_samples",
    "reference_steps",
    "eval_steps",
    "cache_layers",
    "cache_interval",
    "active_t_min",
    "active_t_max",
    "active_window_warmup_refreshes",
    "latency_median_sec",
    "speedup_vs_reference",
    "cache_hit_rate",
    "same_seed_rel_l2",
    "same_seed_mse",
    "same_seed_psnr",
    "low_freq_delta_ratio",
    "mid_freq_delta_ratio",
    "high_freq_delta_ratio",
    "run_dir",
]

AGGREGATE_FIELDNAMES = [
    "method_type",
    "method_name",
    "seed_count",
    "num_samples",
    "reference_steps",
    "eval_steps",
    "cache_layers",
    "cache_interval",
    "active_t_min",
    "active_t_max",
    "active_window_warmup_refreshes",
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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer list must not be empty")
    return items


def _parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("string list must not be empty")
    return items


def _make_run_id(seed_hint: int, reference_steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed_hint}_ref{reference_steps}"


def _safe_method_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def compute_speedup(reference_latency_sec: float, method_latency_sec: float) -> float:
    return reference_latency_sec / method_latency_sec if method_latency_sec > 0 else float("inf")


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
                "cache_layers": first["cache_layers"],
                "cache_interval": first["cache_interval"],
                "active_t_min": first["active_t_min"],
                "active_t_max": first["active_t_max"],
                "active_window_warmup_refreshes": first["active_window_warmup_refreshes"],
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


def _comparison_fields(reference_output: torch.Tensor, output: torch.Tensor, reference_latency: float, method_latency: float) -> dict[str, Any]:
    comparison = _compare_outputs(reference_output, output, reference_latency, method_latency)
    frequency_delta = comparison.get("frequency_delta") or {}
    return {
        "speedup_vs_reference": comparison["speedup"],
        "same_seed_rel_l2": comparison["same_seed_rel_l2"],
        "same_seed_mse": comparison["same_seed_mse"],
        "same_seed_psnr": comparison["same_seed_psnr"],
        "low_freq_delta_ratio": frequency_delta.get("low_ratio"),
        "mid_freq_delta_ratio": frequency_delta.get("mid_ratio"),
        "high_freq_delta_ratio": frequency_delta.get("high_ratio"),
    }


def _base_config(
    run_id: str,
    out_dir: Path,
    preview_root: Path,
    reference_steps: int,
    num_samples: int,
    batch_size: int,
    seed: int,
    timing_repeats: int,
    warmup_runs: int,
) -> Stage2BConfig:
    return Stage2BConfig(
        jit_dir=Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve(),
        ckpt_dir=_detect_jit_ckpt_dir(),
        run_id=run_id,
        run_dir=out_dir,
        preview_dir=preview_root,
        model=os.environ.get("PFC_STAGE3A_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE3A_IMG_SIZE", 256),
        num_samples=num_samples,
        batch_size=batch_size,
        steps=reference_steps,
        seed=seed,
        cfg=_env_float("PFC_STAGE3A_CFG", 3.0),
        interval_min=_env_float("PFC_STAGE3A_CFG_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE3A_CFG_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=1,
        cache_layers="none",
        cache_branches=os.environ.get("PFC_STAGE3A_CACHE_BRANCHES", "cond,uncond"),
        active_t_min=None,
        active_t_max=None,
        timing_repeats=timing_repeats,
        warmup_runs=warmup_runs,
        save_previews=False,
    )


def _run_no_cache_timed(
    config: Stage2BConfig,
    labels: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model = _load_jit_model(config, device)
    timing = _time_repeats(model, labels, noise, config)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return timing


def _run_cache_timed(
    config: Stage2BConfig,
    preset: BackboneCachePreset,
    labels: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _load_jit_model(config, device)
    num_blocks = len(model.net.blocks)
    selected_layer_ids = parse_layer_list(preset.cache_layers, num_blocks)
    selected_modules = [f"blocks.{idx}" for idx in selected_layer_ids]
    cache_state = RuntimeCacheState(model_name="JiT", enabled=True)
    branches = {branch.strip() for branch in config.cache_branches.split(",") if branch.strip()}
    policy = FixedIntervalCachePolicy.from_branches(
        branches,
        cache_modules=set(selected_modules),
        **preset_to_policy_kwargs(preset),
    )
    wrap_jit_blocks(model, cache_state, policy, selected_layer_ids)
    timing = _time_repeats(model, labels, noise, config, cache_state=cache_state)
    cache_stats = cache_state.summary()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return timing, cache_stats


def _row(
    method_type: str,
    method_name: str,
    config: Stage2BConfig,
    reference_steps: int,
    latency_median_sec: float,
    cache_hit_rate: float,
    comparison_fields: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "method_type": method_type,
        "method_name": method_name,
        "seed": config.seed,
        "num_samples": config.num_samples,
        "reference_steps": reference_steps,
        "eval_steps": config.steps,
        "cache_layers": config.cache_layers,
        "cache_interval": config.cache_interval,
        "active_t_min": config.active_t_min,
        "active_t_max": config.active_t_max,
        "active_window_warmup_refreshes": config.active_window_warmup_refreshes,
        "latency_median_sec": latency_median_sec,
        "cache_hit_rate": cache_hit_rate,
        "run_dir": str(run_dir),
        **comparison_fields,
    }


def _write_summary(path: Path, aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "# JiT Stage 3A BackboneCache Benchmark",
        "",
        "| method | type | speedup mean | rel-L2 mean | rel-L2 std | hit rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {method_name} | {method_type} | {speedup_mean:.4f} | {rel_l2_mean:.6f} | "
            "{rel_l2_std:.6f} | {hit_rate_mean:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    out_dir: Path,
    preview_root: Path,
    seeds: list[int],
    num_samples: int,
    batch_size: int,
    reference_steps: int,
    timing_repeats: int,
    warmup_runs: int,
    preset_names: list[str],
    reduced_steps: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    presets = get_jit_backbone_cache_presets()
    missing = [name for name in preset_names if name not in presets]
    if missing:
        raise ValueError(f"Unknown BackboneCache presets: {missing}")
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in seeds:
        seed_run_id = f"seed{seed}"
        base = _base_config(
            seed_run_id,
            out_dir / "runs" / seed_run_id,
            preview_root / seed_run_id,
            reference_steps,
            num_samples,
            batch_size,
            seed,
            timing_repeats,
            warmup_runs,
        )
        labels, noise = _make_inputs(base, device)
        reference_run_dir = out_dir / "runs" / seed_run_id / "reference"
        reference_run_dir.mkdir(parents=True, exist_ok=True)
        reference_timing = _run_no_cache_timed(base, labels, noise, device)
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
                    "same_seed_psnr": float("inf"),
                    "low_freq_delta_ratio": 0.0,
                    "mid_freq_delta_ratio": 0.0,
                    "high_freq_delta_ratio": 0.0,
                },
                reference_run_dir,
            )
        )

        for name in preset_names:
            if name == "no_cache":
                continue
            preset = presets[name]
            method_run_dir = out_dir / "runs" / seed_run_id / name
            method_run_dir.mkdir(parents=True, exist_ok=True)
            config = replace(
                base,
                run_id=f"{seed_run_id}_{name}",
                run_dir=method_run_dir,
                preview_dir=preview_root / seed_run_id / name,
                cache_layers=preset.cache_layers,
                cache_interval=preset.cache_interval,
                active_t_min=preset.active_t_min,
                active_t_max=preset.active_t_max,
                active_window_warmup_refreshes=preset.active_window_warmup_refreshes,
            )
            (method_run_dir / "preset.json").write_text(
                json.dumps(preset_to_config_dict(preset), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            timing, cache_stats = _run_cache_timed(config, preset, labels, noise, device)
            latency = float(timing["latency_median_sec"])
            comparison = _comparison_fields(reference_output, timing["output"], reference_latency, latency)
            rows.append(_row("cache", name, config, reference_steps, latency, float(cache_stats["hit_rate"]), comparison, method_run_dir))

        for steps in reduced_steps:
            method_name = f"reduced_steps_{steps}"
            method_run_dir = out_dir / "runs" / seed_run_id / method_name
            method_run_dir.mkdir(parents=True, exist_ok=True)
            config = replace(
                base,
                run_id=f"{seed_run_id}_{method_name}",
                run_dir=method_run_dir,
                preview_dir=preview_root / seed_run_id / method_name,
                steps=steps,
                cache_layers="none",
                cache_interval=1,
                active_t_min=None,
                active_t_max=None,
                active_window_warmup_refreshes=0,
            )
            timing = _run_no_cache_timed(config, labels, noise, device)
            latency = float(timing["latency_median_sec"])
            comparison = _comparison_fields(reference_output, timing["output"], reference_latency, latency)
            rows.append(_row("reduced_steps", method_name, config, reference_steps, latency, 0.0, comparison, method_run_dir))

    aggregates = aggregate_benchmark_rows(rows)
    return rows, aggregates


def main() -> int:
    seeds = _parse_int_list(os.environ.get("PFC_STAGE3A_SEEDS", "0,1,2"))
    reference_steps = _env_int("PFC_STAGE3A_REFERENCE_STEPS", 50)
    run_id = os.environ.get("PFC_STAGE3A_BENCHMARK_RUN_ID", _make_run_id(seeds[0], reference_steps))
    out_dir = Path(os.environ.get("PFC_STAGE3A_BENCHMARK_DIR", ROOT / "logs/stage3a/jit_backbone_benchmark" / run_id)).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE3A_PREVIEW_ROOT", ROOT / "outputs/stage3a/previews/jit_backbone_benchmark" / run_id)
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preset_names = _parse_str_list(
        os.environ.get(
            "PFC_STAGE3A_PRESETS",
            "no_cache,quality_t02_08,speed_t02_10,quality_t01_08_w1,quality_t01_08_w2,aggressive_i3_t02_08",
        )
    )
    reduced_steps = _parse_int_list(os.environ.get("PFC_STAGE3A_REDUCED_STEPS", "30,35,40"))
    rows, aggregates = run_benchmark(
        out_dir=out_dir,
        preview_root=preview_root,
        seeds=seeds,
        num_samples=_env_int("PFC_STAGE3A_NUM_SAMPLES", 16),
        batch_size=_env_int("PFC_STAGE3A_BATCH_SIZE", 4),
        reference_steps=reference_steps,
        timing_repeats=_env_int("PFC_STAGE3A_TIMING_REPEATS", 2),
        warmup_runs=_env_int("PFC_STAGE3A_WARMUP_RUNS", 1),
        preset_names=preset_names,
        reduced_steps=reduced_steps,
    )
    _write_csv(out_dir / "benchmark_results.csv", rows, BENCHMARK_FIELDNAMES)
    (out_dir / "benchmark_results.json").write_text(
        json.dumps({"rows": rows, "aggregates": aggregates}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(out_dir / "benchmark_aggregate.csv", aggregates, AGGREGATE_FIELDNAMES)
    _write_summary(out_dir / "summary.md", aggregates)
    print(f"JiT Stage 3A benchmark dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
