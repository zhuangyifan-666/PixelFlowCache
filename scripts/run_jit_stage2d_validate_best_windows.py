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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_jit_stage2b_cache import Stage2BConfig, _detect_jit_ckpt_dir, run_experiment  # noqa: E402


VALIDATION_FIELDNAMES = [
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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def make_stage2d_run_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def window_label(t_min: float | None, t_max: float | None) -> str:
    left = "none" if t_min is None else str(t_min).replace(".", "p")
    right = "none" if t_max is None else str(t_max).replace(".", "p")
    return f"t{left}-{right}"


def config_label(row: dict[str, Any]) -> str:
    return f"{row['cache_layers']}/i{row['cache_interval']} [{row['active_t_min']},{row['active_t_max']})"


def base_config_from_env() -> tuple[Stage2BConfig, Path, Path]:
    seed = _env_int("PFC_STAGE2D_SEED", 0)
    steps = _env_int("PFC_STAGE2D_STEPS", 50)
    run_id = os.environ.get("PFC_STAGE2D_VALIDATE_RUN_ID", make_stage2d_run_id(seed, steps))
    validate_dir = Path(
        os.environ.get("PFC_STAGE2D_VALIDATE_DIR", ROOT / "logs/stage2d/jit_validate_best" / run_id)
    ).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE2D_PREVIEW_ROOT", ROOT / "outputs/stage2d/previews/jit_validate_best" / run_id)
    ).resolve()
    base = Stage2BConfig(
        jit_dir=Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve(),
        ckpt_dir=_detect_jit_ckpt_dir(),
        run_id=run_id,
        run_dir=validate_dir,
        preview_dir=preview_root,
        model=os.environ.get("PFC_STAGE2D_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE2D_IMG_SIZE", 256),
        num_samples=_env_int("PFC_STAGE2D_NUM_SAMPLES", 32),
        batch_size=_env_int("PFC_STAGE2D_BATCH_SIZE", 4),
        steps=steps,
        seed=seed,
        cfg=_env_float("PFC_STAGE2D_CFG", 3.0),
        interval_min=_env_float("PFC_STAGE2D_CFG_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE2D_CFG_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=_env_int("PFC_STAGE2D_CACHE_INTERVAL", 2),
        cache_layers=os.environ.get("PFC_STAGE2D_CACHE_LAYERS", "all"),
        cache_branches=os.environ.get("PFC_STAGE2D_CACHE_BRANCHES", "cond,uncond"),
        active_t_min=0.1,
        active_t_max=0.8,
        timing_repeats=_env_int("PFC_STAGE2D_TIMING_REPEATS", 3),
        warmup_runs=_env_int("PFC_STAGE2D_WARMUP_RUNS", 1),
        diag_full_probe=False,
        diag_probe_steps="all",
        save_previews=False,
    )
    return base, validate_dir, preview_root


def validation_configs(include_i3: bool = True) -> list[tuple[str, int, float | None, float | None]]:
    configs = [
        ("none", 1, None, None),
        ("all", 2, 0.1, 0.8),
        ("all", 2, 0.1, 1.0),
        ("all", 2, 0.2, 0.8),
        ("all", 2, 0.2, 1.0),
    ]
    if include_i3:
        configs.append(("all", 3, 0.2, 0.8))
    return configs


def result_row_from_stage2d(result: dict[str, Any], config: Stage2BConfig) -> dict[str, Any]:
    comparison = result["comparison"]
    cache_stats = result["cache_stats"]
    frequency_delta = comparison.get("frequency_delta") or {}
    return {
        "cache_layers": config.cache_layers,
        "cache_interval": config.cache_interval,
        "active_t_min": config.active_t_min,
        "active_t_max": config.active_t_max,
        "num_samples": config.num_samples,
        "steps": config.steps,
        "seed": config.seed,
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def summarize_validation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache_rows = [row for row in rows if row.get("cache_layers") != "none"]
    best_quality = min(cache_rows, key=lambda row: (_safe_float(row["same_seed_rel_l2"]), -_safe_float(row["speedup_median"]))) if cache_rows else None
    fastest = max(cache_rows, key=lambda row: _safe_float(row["speedup_median"])) if cache_rows else None
    return {
        "record_count": len(rows),
        "best_quality_row": best_quality,
        "fastest_row": fastest,
        "rows": rows,
    }


def _write_summary(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    summary = summarize_validation_rows(rows)
    lines = [
        f"# {title}",
        "",
        "| layers | interval | t min | t max | speedup | hit rate | rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cache_layers} | {cache_interval} | {active_t_min} | {active_t_max} | "
            "{speedup_median:.4f} | {cache_hit_rate:.4f} | {same_seed_rel_l2:.6f} |".format(**row)
        )
    if summary["best_quality_row"]:
        best = summary["best_quality_row"]
        lines.extend(["", f"Best quality row: `{config_label(best)}` run=`{best['run_dir']}`"])
    if summary["fastest_row"]:
        fastest = summary["fastest_row"]
        lines.append(f"Fastest row: `{config_label(fastest)}` run=`{fastest['run_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}


def main() -> int:
    base, validate_dir, preview_root = base_config_from_env()
    validate_dir.mkdir(parents=True, exist_ok=True)
    include_i3 = _env_bool("PFC_STAGE2D_INCLUDE_I3", True)
    rows: list[dict[str, Any]] = []
    for layers, interval, t_min, t_max in validation_configs(include_i3=include_i3):
        run_id = f"{base.run_id}_{layers.replace(':', '-')}_i{interval}_{window_label(t_min, t_max)}"
        config = replace(
            base,
            run_id=run_id,
            run_dir=validate_dir / "runs" / run_id,
            preview_dir=preview_root / run_id,
            cache_layers=layers,
            cache_interval=interval,
            active_t_min=t_min,
            active_t_max=t_max,
            active_window_warmup_refreshes=0,
            save_previews=False,
            diag_full_probe=False,
        )
        rows.append(result_row_from_stage2d(run_experiment(config), config))
    write_csv(validate_dir / "validation_results.csv", rows, VALIDATION_FIELDNAMES)
    (validate_dir / "validation_results.json").write_text(
        json.dumps(summarize_validation_rows(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(validate_dir / "summary.md", "JiT Stage 2D Best-Window Validation", rows)
    print(f"JiT Stage 2D validation dir: {validate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
