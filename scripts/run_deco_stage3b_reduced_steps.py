#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
    json_config,
    make_inputs,
    make_run_id,
    parse_int_list,
    run_no_cache,
    write_common_meta,
    write_csv,
    write_json,
)


FIELDNAMES = [
    "method",
    "seed",
    "num_samples",
    "reference_steps",
    "eval_steps",
    "latency_median_sec",
    "reference_latency_median_sec",
    "speedup_vs_reference",
    "same_seed_rel_l2",
    "same_seed_mse",
    "same_seed_mae",
    "same_seed_psnr",
    "low_freq_delta_ratio",
    "mid_freq_delta_ratio",
    "high_freq_delta_ratio",
]


def build_config() -> DeCoStage3BConfig:
    seed = env_int("PFC_STAGE3B_SEED", 0)
    reference_steps = env_int("PFC_STAGE3B_REFERENCE_STEPS", env_int("PFC_STAGE3B_STEPS", 20))
    run_id = os.environ.get("PFC_STAGE3B_REDUCED_RUN_ID", make_run_id(seed, reference_steps, "reduced"))
    return DeCoStage3BConfig(
        deco_dir=default_deco_dir(),
        ckpt_path=detect_deco_ckpt(),
        config_path=default_deco_config(),
        run_id=run_id,
        run_dir=Path(os.environ.get("PFC_STAGE3B_REDUCED_DIR", ROOT / "logs/stage3b/deco_reduced_steps" / run_id)).resolve(),
        num_samples=env_int("PFC_STAGE3B_NUM_SAMPLES", 8),
        batch_size=env_int("PFC_STAGE3B_BATCH_SIZE", 4),
        steps=reference_steps,
        seed=seed,
        cfg=env_float("PFC_STAGE3B_CFG", 3.2),
        cfg_interval_min=env_float("PFC_STAGE3B_CFG_INTERVAL_MIN", 0.1),
        cfg_interval_max=env_float("PFC_STAGE3B_CFG_INTERVAL_MAX", 1.0),
        timing_repeats=env_int("PFC_STAGE3B_TIMING_REPEATS", 2),
        warmup_runs=env_int("PFC_STAGE3B_WARMUP_RUNS", 1),
        resolution=env_int("PFC_STAGE3B_RESOLUTION", 256),
        save_diagnostics=False,
    )


def summarize_reduced_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_quality = min(rows, key=lambda row: (float(row["same_seed_rel_l2"]), -float(row["speedup_vs_reference"]))) if rows else None
    fastest = max(rows, key=lambda row: float(row["speedup_vs_reference"])) if rows else None
    return {
        "record_count": len(rows),
        "best_quality_row": best_quality,
        "fastest_row": fastest,
        "rows": rows,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DeCo Stage 3B Reduced-Step No-Cache Baseline",
        "",
        "| eval steps | speedup | rel-L2 | PSNR |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {eval_steps} | {speedup_vs_reference:.4f} | {same_seed_rel_l2:.6f} | {same_seed_psnr:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reduced_steps(config: DeCoStage3BConfig, reduced_steps: list[int]) -> list[dict[str, Any]]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, uncondition, noise = make_inputs(config, device)
    write_common_meta(config, "scripts/run_deco_stage3b_reduced_steps.py", extra={"reduced_steps": reduced_steps})
    write_json(config.run_dir / "config.json", json_config(config))
    reference_timing = run_no_cache(config, labels, uncondition, noise, device)
    reference_output = reference_timing["output"]
    reference_latency = float(reference_timing["latency_median_sec"])
    rows: list[dict[str, Any]] = []
    for steps in reduced_steps:
        eval_config = replace(config, steps=steps, run_id=f"{config.run_id}_steps{steps}", save_diagnostics=False)
        timing = run_no_cache(eval_config, labels, uncondition, noise, device)
        comparison = compare_outputs(reference_output, timing["output"], reference_latency, float(timing["latency_median_sec"]))
        frequency_delta = comparison.get("frequency_delta_bands") or {}
        rows.append(
            {
                "method": "reduced_steps",
                "seed": config.seed,
                "num_samples": config.num_samples,
                "reference_steps": config.steps,
                "eval_steps": steps,
                "latency_median_sec": timing["latency_median_sec"],
                "reference_latency_median_sec": reference_latency,
                "speedup_vs_reference": comparison["speedup"],
                "same_seed_rel_l2": comparison["same_seed_rel_l2"],
                "same_seed_mse": comparison["same_seed_mse"],
                "same_seed_mae": comparison["same_seed_mae"],
                "same_seed_psnr": comparison["same_seed_psnr"],
                "low_freq_delta_ratio": frequency_delta.get("low_ratio"),
                "mid_freq_delta_ratio": frequency_delta.get("mid_ratio"),
                "high_freq_delta_ratio": frequency_delta.get("high_ratio"),
            }
        )
    write_csv(config.run_dir / "reduced_step_results.csv", rows, FIELDNAMES)
    write_json(config.run_dir / "reduced_step_results.json", summarize_reduced_rows(rows))
    write_summary(config.run_dir / "summary.md", rows)
    return rows


def main() -> int:
    config = build_config()
    reduced_steps = parse_int_list(os.environ.get("PFC_STAGE3B_REDUCED_STEPS", "12,15,18"))
    rows = run_reduced_steps(config, reduced_steps)
    print(f"DeCo Stage 3B reduced-step dir: {config.run_dir}")
    print(json.dumps(summarize_reduced_rows(rows), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
