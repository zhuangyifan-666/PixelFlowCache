#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_jit_stage2b_cache import Stage2BConfig, build_config_from_args, run_experiment  # noqa: E402


def _make_sweep_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _configs(fast: bool) -> list[tuple[str, int, float | None, float | None]]:
    if fast:
        return [
            ("all", 2, 0.1, 1.0),
            ("all", 2, 0.1, 0.8),
            ("prefix:6", 2, 0.1, 0.8),
            ("suffix:6", 2, 0.1, 0.8),
            ("middle", 2, 0.1, 0.8),
            ("all", 3, 0.1, 0.8),
        ]
    layer_groups = ["all", "prefix:6", "suffix:6", "middle", "range:0:6", "range:6:12"]
    windows = [(None, None), (0.1, 1.0), (0.1, 0.9), (0.1, 0.8), (0.2, 0.8)]
    return [(layers, interval, t_min, t_max) for interval in [2] for layers in layer_groups for t_min, t_max in windows]


def _window_label(t_min: float | None, t_max: float | None) -> str:
    left = "none" if t_min is None else str(t_min).replace(".", "p")
    right = "none" if t_max is None else str(t_max).replace(".", "p")
    return f"t{left}-{right}"


def _row(result: dict[str, Any], config: Stage2BConfig) -> dict[str, Any]:
    comparison = result["comparison"]
    cache_stats = result["cache_stats"]
    frequency_delta = comparison.get("frequency_delta") or {}
    return {
        "cache_layers": config.cache_layers,
        "selected_layer_ids": " ".join(str(item) for item in result["selected_layer_ids"]),
        "cache_interval": config.cache_interval,
        "active_t_min": config.active_t_min,
        "active_t_max": config.active_t_max,
        "steps": config.steps,
        "num_samples": config.num_samples,
        "no_cache_latency_median_sec": comparison["no_cache_latency_sec"],
        "cache_latency_median_sec": comparison["cache_latency_sec"],
        "speedup_median": comparison["speedup_median"],
        "cache_hit_rate": cache_stats["hit_rate"],
        "same_seed_mse": comparison["same_seed_mse"],
        "same_seed_rel_l2": comparison["same_seed_rel_l2"],
        "same_seed_psnr": comparison["same_seed_psnr"],
        "low_freq_delta_ratio": frequency_delta.get("low_ratio"),
        "mid_freq_delta_ratio": frequency_delta.get("mid_ratio"),
        "high_freq_delta_ratio": frequency_delta.get("high_ratio"),
        "run_dir": result["run_dir"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "cache_layers",
        "selected_layer_ids",
        "cache_interval",
        "active_t_min",
        "active_t_max",
        "steps",
        "num_samples",
        "no_cache_latency_median_sec",
        "cache_latency_median_sec",
        "speedup_median",
        "cache_hit_rate",
        "same_seed_mse",
        "same_seed_rel_l2",
        "same_seed_psnr",
        "low_freq_delta_ratio",
        "mid_freq_delta_ratio",
        "high_freq_delta_ratio",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    best = min(rows, key=lambda row: (float(row["same_seed_rel_l2"]), -float(row["speedup_median"]))) if rows else None
    lines = [
        "# JiT Stage 2B Sweep",
        "",
        "| layers | interval | t min | t max | speedup | hit rate | rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cache_layers} | {cache_interval} | {active_t_min} | {active_t_max} | "
            "{speedup_median:.4f} | {cache_hit_rate:.4f} | {same_seed_rel_l2:.6f} |".format(**row)
        )
    if best:
        lines.extend(["", f"Best rel-L2 row: `{best['run_dir']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base = build_config_from_args([])
    fast = os.environ.get("PFC_STAGE2B_SWEEP_FAST", "1").lower() in {"1", "true", "yes", "on"}
    sweep_id = os.environ.get("PFC_STAGE2B_SWEEP_RUN_ID", _make_sweep_id(base.seed, base.steps))
    sweep_dir = Path(os.environ.get("PFC_STAGE2B_SWEEP_DIR", ROOT / "logs/stage2b/jit_sweep" / sweep_id)).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE2B_SWEEP_PREVIEW_ROOT", ROOT / "outputs/stage2b/previews/jit_sweep" / sweep_id)
    ).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for layers, interval, t_min, t_max in _configs(fast):
        run_id = f"{sweep_id}_{layers.replace(':', '-')}_i{interval}_{_window_label(t_min, t_max)}"
        config = replace(
            base,
            run_id=run_id,
            run_dir=sweep_dir / "runs" / run_id,
            preview_dir=preview_root / run_id,
            cache_layers=layers,
            cache_interval=interval,
            active_t_min=t_min,
            active_t_max=t_max,
            save_previews=False,
            diag_full_probe=False,
        )
        result = run_experiment(config)
        rows.append(_row(result, config))
    _write_csv(sweep_dir / "sweep_results.csv", rows)
    (sweep_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(sweep_dir / "summary.md", rows)
    print(f"JiT Stage 2B sweep dir: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
