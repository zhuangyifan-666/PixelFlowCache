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

from scripts.run_jit_stage2b_cache import run_experiment  # noqa: E402
from scripts.run_jit_stage2d_validate_best_windows import (  # noqa: E402
    result_row_from_stage2d,
    base_config_from_env,
    make_stage2d_run_id,
    window_label,
)


FIRST_HIT_FIELDNAMES = [
    "active_window_warmup_refreshes",
    "cache_layers",
    "cache_interval",
    "active_t_min",
    "active_t_max",
    "num_samples",
    "steps",
    "seed",
    "speedup_median",
    "cache_hit_rate",
    "same_seed_rel_l2",
    "same_seed_mse",
    "same_seed_psnr",
    "low_freq_delta_ratio",
    "mid_freq_delta_ratio",
    "high_freq_delta_ratio",
    "run_dir",
]


def _parse_warmups(value: str) -> list[int]:
    warmups = []
    for item in value.split(","):
        item = item.strip()
        if item:
            warmups.append(int(item))
    if not warmups:
        raise ValueError("warmup list must not be empty")
    return warmups


def summarize_first_hit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_quality = min(rows, key=lambda row: (float(row["same_seed_rel_l2"]), -float(row["speedup_median"]))) if rows else None
    fastest = max(rows, key=lambda row: float(row["speedup_median"])) if rows else None
    baseline = next((row for row in rows if int(row["active_window_warmup_refreshes"]) == 0), None)
    improvements = []
    if baseline is not None:
        base_rel_l2 = float(baseline["same_seed_rel_l2"])
        for row in rows:
            improvements.append(
                {
                    "active_window_warmup_refreshes": row["active_window_warmup_refreshes"],
                    "rel_l2_delta_vs_zero": float(row["same_seed_rel_l2"]) - base_rel_l2,
                    "speedup_delta_vs_zero": float(row["speedup_median"]) - float(baseline["speedup_median"]),
                }
            )
    return {
        "record_count": len(rows),
        "best_quality_row": best_quality,
        "fastest_row": fastest,
        "baseline_zero_warmup_row": baseline,
        "improvements_vs_zero": improvements,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIRST_HIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize_first_hit_rows(rows)
    lines = [
        "# JiT Stage 2D First-Hit Delay",
        "",
        "| warmup refreshes | speedup | hit rate | rel-L2 | PSNR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {active_window_warmup_refreshes} | {speedup_median:.4f} | {cache_hit_rate:.4f} | "
            "{same_seed_rel_l2:.6f} | {same_seed_psnr:.4f} |".format(**row)
        )
    if summary["best_quality_row"]:
        best = summary["best_quality_row"]
        lines.extend(["", f"Best quality warmup refreshes: `{best['active_window_warmup_refreshes']}`"])
    if summary["fastest_row"]:
        fastest = summary["fastest_row"]
        lines.append(f"Fastest warmup refreshes: `{fastest['active_window_warmup_refreshes']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base, _unused_dir, _unused_preview = base_config_from_env()
    first_hit_steps = int(os.environ.get("PFC_STAGE2D_FIRST_HIT_STEPS", 20))
    first_hit_num_samples = int(os.environ.get("PFC_STAGE2D_FIRST_HIT_NUM_SAMPLES", 8))
    first_hit_timing_repeats = int(os.environ.get("PFC_STAGE2D_FIRST_HIT_TIMING_REPEATS", base.timing_repeats))
    run_id = os.environ.get("PFC_STAGE2D_FIRST_HIT_RUN_ID", make_stage2d_run_id(base.seed, first_hit_steps))
    out_dir = Path(os.environ.get("PFC_STAGE2D_FIRST_HIT_DIR", ROOT / "logs/stage2d/jit_first_hit_delay" / run_id)).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE2D_FIRST_HIT_PREVIEW_ROOT", ROOT / "outputs/stage2d/previews/jit_first_hit_delay" / run_id)
    ).resolve()
    warmups = _parse_warmups(os.environ.get("PFC_STAGE2D_FIRST_HIT_WARMUPS", "0,1,2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for warmup_refreshes in warmups:
        run_name = f"{run_id}_all_i{base.cache_interval}_{window_label(base.active_t_min, base.active_t_max)}_w{warmup_refreshes}"
        config = replace(
            base,
            run_id=run_name,
            run_dir=out_dir / "runs" / run_name,
            preview_dir=preview_root / run_name,
            num_samples=first_hit_num_samples,
            steps=first_hit_steps,
            timing_repeats=first_hit_timing_repeats,
            cache_layers=os.environ.get("PFC_STAGE2D_FIRST_HIT_CACHE_LAYERS", "all"),
            cache_interval=int(os.environ.get("PFC_STAGE2D_FIRST_HIT_CACHE_INTERVAL", 2)),
            active_t_min=float(os.environ.get("PFC_STAGE2D_FIRST_HIT_ACTIVE_T_MIN", 0.1)),
            active_t_max=float(os.environ.get("PFC_STAGE2D_FIRST_HIT_ACTIVE_T_MAX", 0.8)),
            active_window_warmup_refreshes=warmup_refreshes,
            save_previews=False,
            diag_full_probe=False,
        )
        row = result_row_from_stage2d(run_experiment(config), config)
        row["active_window_warmup_refreshes"] = warmup_refreshes
        rows.append(row)
    _write_csv(out_dir / "first_hit_delay_results.csv", rows)
    (out_dir / "first_hit_delay_results.json").write_text(
        json.dumps(summarize_first_hit_rows(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(out_dir / "summary.md", rows)
    print(f"JiT Stage 2D first-hit delay dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
