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

from scripts.run_jit_stage2_cache import Stage2Config, build_config_from_args, run_experiment  # noqa: E402


def _make_grid_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _grid_configs(fast: bool) -> list[tuple[str, int]]:
    if fast:
        return [("none", 1), ("middle", 2), ("middle", 3), ("all", 2)]
    layers = ["none", "middle", "early", "late", "all"]
    intervals = [1, 2, 3, 4]
    return [(layer, interval) for layer in layers for interval in intervals]


def _row_from_result(result: dict[str, Any], cache_layers: str, cache_interval: int, config: Stage2Config) -> dict[str, Any]:
    comparison = result["comparison"]
    cache_stats = result["cache_stats"]
    frequency_delta = comparison.get("frequency_delta") or {}
    return {
        "cache_layers": cache_layers,
        "cache_interval": cache_interval,
        "selected_layer_ids": " ".join(str(item) for item in result["selected_layer_ids"]),
        "steps": config.steps,
        "num_samples": config.num_samples,
        "no_cache_latency_sec": comparison["no_cache_latency_sec"],
        "cache_latency_sec": comparison["cache_latency_sec"],
        "speedup": comparison["speedup"],
        "cache_hit_rate": cache_stats["hit_rate"],
        "same_seed_mse": comparison["same_seed_mse"],
        "same_seed_rel_l2": comparison["same_seed_rel_l2"],
        "high_freq_delta_ratio": frequency_delta.get("high_ratio"),
        "run_dir": result["run_dir"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cache_layers",
        "cache_interval",
        "selected_layer_ids",
        "steps",
        "num_samples",
        "no_cache_latency_sec",
        "cache_latency_sec",
        "speedup",
        "cache_hit_rate",
        "same_seed_mse",
        "same_seed_rel_l2",
        "high_freq_delta_ratio",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# JiT Stage 2 Fixed-Interval Block Cache Grid",
        "",
        "| layers | interval | speedup | hit rate | rel-L2 | run dir |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {cache_layers} | {cache_interval} | {speedup:.4f} | {cache_hit_rate:.4f} | "
            "{same_seed_rel_l2:.6f} | `{run_dir}` |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base = build_config_from_args([])
    fast = os.environ.get("PFC_STAGE2_GRID_FAST", "1").lower() in {"1", "true", "yes", "on"}
    grid_id = os.environ.get("PFC_STAGE2_GRID_RUN_ID", _make_grid_id(base.seed, base.steps))
    grid_dir = Path(os.environ.get("PFC_STAGE2_GRID_DIR", ROOT / "logs/stage2/jit_grid" / grid_id)).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE2_GRID_PREVIEW_ROOT", ROOT / "outputs/stage2/previews/jit_grid" / grid_id)
    ).resolve()
    grid_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cache_layers, cache_interval in _grid_configs(fast):
        run_id = f"{grid_id}_{cache_layers}_i{cache_interval}"
        run_dir = grid_dir / "runs" / run_id
        config = replace(
            base,
            run_id=run_id,
            run_dir=run_dir,
            preview_dir=preview_root / run_id,
            cache_layers=cache_layers,
            cache_interval=cache_interval,
            save_previews=False,
        )
        result = run_experiment(config)
        rows.append(_row_from_result(result, cache_layers, cache_interval, config))

    _write_csv(grid_dir / "grid_results.csv", rows)
    (grid_dir / "grid_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(grid_dir / "summary.md", rows)
    print(f"JiT Stage 2 grid dir: {grid_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
