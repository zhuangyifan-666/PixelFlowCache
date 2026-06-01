#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
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
    run_cached,
    run_no_cache,
    write_common_meta,
    write_json,
)


def build_config() -> DeCoStage3BConfig:
    seed = env_int("PFC_STAGE3B_SEED", 0)
    steps = env_int("PFC_STAGE3B_STEPS", 20)
    cache_interval = env_int("PFC_STAGE3B_CACHE_INTERVAL", 2)
    cache_units = os.environ.get("PFC_STAGE3B_CACHE_UNITS", "backbone_blocks")
    run_id = os.environ.get("PFC_STAGE3B_RUN_ID", make_run_id(seed, steps, f"i{cache_interval}_{cache_units}"))
    return DeCoStage3BConfig(
        deco_dir=default_deco_dir(),
        ckpt_path=detect_deco_ckpt(),
        config_path=default_deco_config(),
        run_id=run_id,
        run_dir=Path(os.environ.get("PFC_STAGE3B_OUT_DIR", ROOT / "logs/stage3b/deco" / run_id)).resolve(),
        num_samples=env_int("PFC_STAGE3B_NUM_SAMPLES", 8),
        batch_size=env_int("PFC_STAGE3B_BATCH_SIZE", 4),
        steps=steps,
        seed=seed,
        cfg=env_float("PFC_STAGE3B_CFG", 3.2),
        cfg_interval_min=env_float("PFC_STAGE3B_CFG_INTERVAL_MIN", 0.1),
        cfg_interval_max=env_float("PFC_STAGE3B_CFG_INTERVAL_MAX", 1.0),
        cache_interval=cache_interval,
        active_t_min=env_optional_float("PFC_STAGE3B_ACTIVE_T_MIN", 0.2),
        active_t_max=env_optional_float("PFC_STAGE3B_ACTIVE_T_MAX", 1.0),
        cache_units=cache_units,
        timing_repeats=env_int("PFC_STAGE3B_TIMING_REPEATS", 2),
        warmup_runs=env_int("PFC_STAGE3B_WARMUP_RUNS", 1),
        resolution=env_int("PFC_STAGE3B_RESOLUTION", 256),
    )


def _summary_without_output(timing: dict[str, Any], mode: str, config: DeCoStage3BConfig) -> dict[str, Any]:
    summary = {key: value for key, value in timing.items() if key != "output"}
    summary.update({"mode": mode, "num_samples": config.num_samples, "steps": config.steps})
    return summary


def write_summary_md(path: Path, result: dict[str, Any]) -> None:
    comparison = result["comparison"]
    cache_summary = result["cache_summary"]
    lines = [
        "# DeCo Stage 3B Cache Feasibility",
        "",
        f"- cache units: `{result['config']['cache_units']}`",
        f"- wrapped modules: {len(result['wrapped_modules'])}",
        f"- no-cache median latency: {result['no_cache_summary']['latency_median_sec']:.4f}s",
        f"- cache median latency: {cache_summary['latency_median_sec']:.4f}s",
        f"- speedup: {comparison['speedup']:.4f}",
        f"- cache hit rate: {cache_summary['cache_hit_rate']:.4f}",
        f"- same-seed rel-L2: {comparison['same_seed_rel_l2']:.6f}",
        f"- same-seed MSE: {comparison['same_seed_mse']:.8f}",
        f"- same-seed PSNR: {comparison['same_seed_psnr']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: DeCoStage3BConfig) -> dict[str, Any]:
    if config.timing_repeats <= 0:
        raise ValueError("PFC_STAGE3B_TIMING_REPEATS must be positive")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, uncondition, noise = make_inputs(config, device)
    write_common_meta(config, "scripts/run_deco_stage3b_cache.py")
    write_json(config.run_dir / "config.json", json_config(config))

    no_cache_timing = run_no_cache(config, labels, uncondition, noise, device)
    cache_timing, cache_stats, wrapped_modules = run_cached(config, labels, uncondition, noise, device)
    comparison = compare_outputs(
        no_cache_timing["output"],
        cache_timing["output"],
        float(no_cache_timing["latency_median_sec"]),
        float(cache_timing["latency_median_sec"]),
    )

    no_cache_summary = _summary_without_output(no_cache_timing, "no_cache", config)
    cache_summary = _summary_without_output(cache_timing, "cache", config)
    cache_summary.update(
        {
            "cache_units": config.cache_units,
            "wrapped_modules": wrapped_modules,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_policy": {
                "interval": config.cache_interval,
                "active_t_min": config.active_t_min,
                "active_t_max": config.active_t_max,
            },
        }
    )
    comparison["speedup_median"] = comparison["speedup"]

    write_json(config.run_dir / "no_cache_summary.json", no_cache_summary)
    write_json(config.run_dir / "cache_summary.json", cache_summary)
    write_json(config.run_dir / "comparison.json", comparison)
    write_json(config.run_dir / "cache_stats.json", cache_stats)

    result = {
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "config": json_config(config),
        "wrapped_modules": wrapped_modules,
        "no_cache_summary": no_cache_summary,
        "cache_summary": cache_summary,
        "comparison": comparison,
        "cache_stats": cache_stats,
    }
    write_summary_md(config.run_dir / "summary.md", result)
    print(f"DeCo Stage 3B cache run dir: {config.run_dir}")
    print(f"Wrapped modules: {len(wrapped_modules)}")
    print(f"Speedup: {comparison['speedup']:.4f}")
    print(f"Cache hit rate: {cache_stats['hit_rate']:.4f}")
    print(f"same_seed_rel_l2: {comparison['same_seed_rel_l2']:.6f}")
    return result


def main() -> int:
    run_experiment(build_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
