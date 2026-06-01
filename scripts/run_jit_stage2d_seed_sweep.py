#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_jit_stage2d_validate_best_windows import (  # noqa: E402
    VALIDATION_FIELDNAMES,
    base_config_from_env,
    config_label,
    make_stage2d_run_id,
    mean_std,
    result_row_from_stage2d,
    window_label,
    write_csv,
)
from scripts.run_jit_stage2b_cache import run_experiment  # noqa: E402


SEED_SWEEP_FIELDNAMES = ["config_label", *VALIDATION_FIELDNAMES]


def _parse_seeds(value: str) -> list[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("seed list must not be empty")
    return seeds


def _default_configs() -> list[tuple[str, int, float | None, float | None]]:
    return [
        ("all", 2, 0.1, 1.0),
        ("all", 2, 0.1, 0.8),
        ("all", 2, 0.2, 0.8),
        ("all", 2, 0.2, 1.0),
    ]


def summarize_seed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config_label"]), []).append(row)
    summaries = []
    for label in sorted(grouped):
        group = grouped[label]
        entry: dict[str, Any] = {
            "config_label": label,
            "seed_count": len(group),
            "seeds": [int(row["seed"]) for row in group],
        }
        for key in ["speedup_median", "cache_hit_rate", "same_seed_rel_l2", "same_seed_psnr", "same_seed_mse"]:
            stats = mean_std([float(row[key]) for row in group])
            entry[f"{key}_mean"] = stats["mean"]
            entry[f"{key}_std"] = stats["std"]
        summaries.append(entry)
    return {
        "record_count": len(rows),
        "config_count": len(summaries),
        "summaries": summaries,
        "rows": rows,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# JiT Stage 2D Seed Sweep",
        "",
        "| config | seeds | speedup mean | speedup std | rel-L2 mean | rel-L2 std | PSNR mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summaries"]:
        lines.append(
            "| {config_label} | {seed_count} | {speedup_median_mean:.4f} | {speedup_median_std:.4f} | "
            "{same_seed_rel_l2_mean:.6f} | {same_seed_rel_l2_std:.6f} | {same_seed_psnr_mean:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base, base_dir, preview_root = base_config_from_env()
    seeds = _parse_seeds(os.environ.get("PFC_STAGE2D_SEEDS", "0,1,2"))
    run_id = os.environ.get("PFC_STAGE2D_SEED_SWEEP_RUN_ID", make_stage2d_run_id(seeds[0], base.steps))
    sweep_dir = Path(os.environ.get("PFC_STAGE2D_SEED_SWEEP_DIR", ROOT / "logs/stage2d/jit_seed_sweep" / run_id)).resolve()
    sweep_preview_root = Path(
        os.environ.get("PFC_STAGE2D_SEED_SWEEP_PREVIEW_ROOT", ROOT / "outputs/stage2d/previews/jit_seed_sweep" / run_id)
    ).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    del base_dir, preview_root
    seed_num_samples = int(os.environ.get("PFC_STAGE2D_SEED_NUM_SAMPLES", 16))
    seed_timing_repeats = int(os.environ.get("PFC_STAGE2D_SEED_TIMING_REPEATS", 2))
    for seed in seeds:
        for layers, interval, t_min, t_max in _default_configs():
            run_name = f"{run_id}_seed{seed}_{layers.replace(':', '-')}_i{interval}_{window_label(t_min, t_max)}"
            config = replace(
                base,
                seed=seed,
                num_samples=seed_num_samples,
                timing_repeats=seed_timing_repeats,
                run_id=run_name,
                run_dir=sweep_dir / "runs" / run_name,
                preview_dir=sweep_preview_root / run_name,
                cache_layers=layers,
                cache_interval=interval,
                active_t_min=t_min,
                active_t_max=t_max,
                active_window_warmup_refreshes=0,
                save_previews=False,
                diag_full_probe=False,
            )
            row = result_row_from_stage2d(run_experiment(config), config)
            row["config_label"] = config_label(row)
            rows.append(row)
    write_csv(sweep_dir / "seed_sweep_results.csv", rows, SEED_SWEEP_FIELDNAMES)
    summary = summarize_seed_rows(rows)
    (sweep_dir / "seed_sweep_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(sweep_dir / "summary.md", summary)
    print(f"JiT Stage 2D seed sweep dir: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
