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

from scripts.run_jit_stage2b_cache import Stage2BConfig, _detect_jit_ckpt_dir, run_experiment  # noqa: E402


WINDOW_FIELDNAMES = [
    "ablation_group",
    "cache_layers",
    "selected_layer_ids",
    "active_t_min",
    "active_t_max",
    "cache_interval",
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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _make_run_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _safe_float(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _window_label(t_min: float | None, t_max: float | None) -> str:
    left = "none" if t_min is None else str(t_min).replace(".", "p")
    right = "none" if t_max is None else str(t_max).replace(".", "p")
    return f"t{left}-{right}"


def base_config_from_env() -> tuple[Stage2BConfig, Path, Path]:
    seed = _env_int("PFC_STAGE2C_SEED", 0)
    steps = _env_int("PFC_STAGE2C_STEPS", 20)
    num_samples = _env_int("PFC_STAGE2C_NUM_SAMPLES", 8)
    batch_size = _env_int("PFC_STAGE2C_BATCH_SIZE", 4)
    run_id = os.environ.get("PFC_STAGE2C_WINDOW_RUN_ID", _make_run_id(seed, steps))
    window_dir = Path(
        os.environ.get("PFC_STAGE2C_WINDOW_DIR", ROOT / "logs/stage2c/jit_window_ablation" / run_id)
    ).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE2C_PREVIEW_ROOT", ROOT / "outputs/stage2c/previews/jit_window_ablation" / run_id)
    ).resolve()
    base = Stage2BConfig(
        jit_dir=Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve(),
        ckpt_dir=_detect_jit_ckpt_dir(),
        run_id=run_id,
        run_dir=window_dir,
        preview_dir=preview_root,
        model=os.environ.get("PFC_STAGE2C_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE2C_IMG_SIZE", 256),
        num_samples=num_samples,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        cfg=_env_float("PFC_STAGE2C_CFG", 3.0),
        interval_min=_env_float("PFC_STAGE2C_CFG_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE2C_CFG_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=2,
        cache_layers=os.environ.get("PFC_STAGE2C_CACHE_LAYERS", "all"),
        cache_branches=os.environ.get("PFC_STAGE2C_CACHE_BRANCHES", "cond,uncond"),
        active_t_min=0.1,
        active_t_max=0.8,
        timing_repeats=_env_int("PFC_STAGE2C_TIMING_REPEATS", 3),
        warmup_runs=_env_int("PFC_STAGE2C_WARMUP_RUNS", 1),
        diag_full_probe=False,
        diag_probe_steps="all",
        save_previews=False,
    )
    return base, window_dir, preview_root


def result_row_from_stage2b(result: dict[str, Any], config: Stage2BConfig, ablation_group: str) -> dict[str, Any]:
    comparison = result["comparison"]
    cache_stats = result["cache_stats"]
    frequency_delta = comparison.get("frequency_delta") or {}
    return {
        "ablation_group": ablation_group,
        "cache_layers": config.cache_layers,
        "selected_layer_ids": " ".join(str(item) for item in result.get("selected_layer_ids", [])),
        "active_t_min": config.active_t_min,
        "active_t_max": config.active_t_max,
        "cache_interval": config.cache_interval,
        "speedup_median": comparison["speedup_median"],
        "cache_hit_rate": cache_stats["hit_rate"],
        "same_seed_rel_l2": comparison["same_seed_rel_l2"],
        "same_seed_mse": comparison["same_seed_mse"],
        "same_seed_psnr": comparison["same_seed_psnr"],
        "low_freq_delta_ratio": frequency_delta.get("low_ratio"),
        "mid_freq_delta_ratio": frequency_delta.get("mid_ratio"),
        "high_freq_delta_ratio": frequency_delta.get("high_ratio"),
        "run_dir": result["run_dir"],
    }


def choose_best_window(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    candidates = [
        row
        for row in rows
        if int(row["cache_interval"]) == 2 and row.get("cache_layers") == "all" and row.get("ablation_group") in {"t_min", "t_max"}
    ]
    if not candidates:
        return 0.1, 0.8
    best = min(candidates, key=lambda row: (_safe_float(row["same_seed_rel_l2"]), -_safe_float(row["speedup_median"])))
    return best["active_t_min"], best["active_t_max"]


def summarize_window_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_quality = min(rows, key=lambda row: (_safe_float(row["same_seed_rel_l2"]), -_safe_float(row["speedup_median"]))) if rows else None
    fastest = max(rows, key=lambda row: _safe_float(row["speedup_median"])) if rows else None
    return {
        "record_count": len(rows),
        "best_quality_row": best_quality,
        "fastest_row": fastest,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize_window_results(rows)
    lines = [
        "# JiT Stage 2C Window Ablation",
        "",
        "| group | t min | t max | interval | speedup | hit rate | rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {ablation_group} | {active_t_min} | {active_t_max} | {cache_interval} | "
            "{speedup_median:.4f} | {cache_hit_rate:.4f} | {same_seed_rel_l2:.6f} |".format(**row)
        )
    if summary["best_quality_row"]:
        best = summary["best_quality_row"]
        lines.extend(
            [
                "",
                "Best same-seed rel-L2 row: "
                f"`{best['ablation_group']}` t=[{best['active_t_min']}, {best['active_t_max']}) "
                f"interval={best['cache_interval']} run=`{best['run_dir']}`",
            ]
        )
    if summary["fastest_row"]:
        fastest = summary["fastest_row"]
        lines.append(
            "Fastest row: "
            f"`{fastest['ablation_group']}` t=[{fastest['active_t_min']}, {fastest['active_t_max']}) "
            f"interval={fastest['cache_interval']} run=`{fastest['run_dir']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_or_reuse(
    base: Stage2BConfig,
    window_dir: Path,
    preview_root: Path,
    cache: dict[tuple[int, float | None, float | None], dict[str, Any]],
    cache_interval: int,
    t_min: float | None,
    t_max: float | None,
) -> tuple[dict[str, Any], Stage2BConfig]:
    key = (cache_interval, t_min, t_max)
    run_id = f"{base.run_id}_all_i{cache_interval}_{_window_label(t_min, t_max)}"
    config = replace(
        base,
        run_id=run_id,
        run_dir=window_dir / "runs" / run_id,
        preview_dir=preview_root / run_id,
        cache_layers="all",
        cache_interval=cache_interval,
        active_t_min=t_min,
        active_t_max=t_max,
        save_previews=False,
        diag_full_probe=False,
    )
    if key not in cache:
        cache[key] = run_experiment(config)
    return cache[key], config


def main() -> int:
    base, window_dir, preview_root = base_config_from_env()
    window_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_cache: dict[tuple[int, float | None, float | None], dict[str, Any]] = {}

    for t_min in [0.0, 0.05, 0.1, 0.2]:
        result, config = _run_or_reuse(base, window_dir, preview_root, run_cache, 2, t_min, 0.8)
        rows.append(result_row_from_stage2b(result, config, "t_min"))

    for t_max in [0.7, 0.8, 0.9, 1.0]:
        result, config = _run_or_reuse(base, window_dir, preview_root, run_cache, 2, 0.1, t_max)
        rows.append(result_row_from_stage2b(result, config, "t_max"))

    best_t_min, best_t_max = choose_best_window(rows)
    intervals = [2, 3]
    if _env_bool("PFC_STAGE2C_INCLUDE_INTERVAL4", False):
        intervals.append(4)
    for interval in intervals:
        result, config = _run_or_reuse(base, window_dir, preview_root, run_cache, interval, best_t_min, best_t_max)
        rows.append(result_row_from_stage2b(result, config, "interval"))

    _write_csv(window_dir / "window_ablation_results.csv", rows)
    (window_dir / "window_ablation_results.json").write_text(
        json.dumps(summarize_window_results(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(window_dir / "summary.md", rows)
    print(f"JiT Stage 2C window ablation dir: {window_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
