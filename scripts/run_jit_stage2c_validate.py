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
from scripts.run_jit_stage2c_window_ablation import WINDOW_FIELDNAMES, result_row_from_stage2b  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _make_run_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _window_label(t_min: float | None, t_max: float | None) -> str:
    left = "none" if t_min is None else str(t_min).replace(".", "p")
    right = "none" if t_max is None else str(t_max).replace(".", "p")
    return f"t{left}-{right}"


def base_config_from_env() -> tuple[Stage2BConfig, Path, Path]:
    seed = _env_int("PFC_STAGE2C_SEED", 0)
    steps = _env_int("PFC_STAGE2C_STEPS", 50)
    run_id = os.environ.get("PFC_STAGE2C_VALIDATE_RUN_ID", _make_run_id(seed, steps))
    validate_dir = Path(
        os.environ.get("PFC_STAGE2C_VALIDATE_DIR", ROOT / "logs/stage2c/jit_validate" / run_id)
    ).resolve()
    preview_root = Path(
        os.environ.get("PFC_STAGE2C_PREVIEW_ROOT", ROOT / "outputs/stage2c/previews/jit_validate" / run_id)
    ).resolve()
    base = Stage2BConfig(
        jit_dir=Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve(),
        ckpt_dir=_detect_jit_ckpt_dir(),
        run_id=run_id,
        run_dir=validate_dir,
        preview_dir=preview_root,
        model=os.environ.get("PFC_STAGE2C_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE2C_IMG_SIZE", 256),
        num_samples=_env_int("PFC_STAGE2C_NUM_SAMPLES", 32),
        batch_size=_env_int("PFC_STAGE2C_BATCH_SIZE", 4),
        steps=steps,
        seed=seed,
        cfg=_env_float("PFC_STAGE2C_CFG", 3.0),
        interval_min=_env_float("PFC_STAGE2C_CFG_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE2C_CFG_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=2,
        cache_layers="all",
        cache_branches=os.environ.get("PFC_STAGE2C_CACHE_BRANCHES", "cond,uncond"),
        active_t_min=0.1,
        active_t_max=0.8,
        timing_repeats=_env_int("PFC_STAGE2C_TIMING_REPEATS", 3),
        warmup_runs=_env_int("PFC_STAGE2C_WARMUP_RUNS", 1),
        diag_full_probe=False,
        diag_probe_steps="all",
        save_previews=False,
    )
    return base, validate_dir, preview_root


def _base_validation_configs() -> list[tuple[str, int, float | None, float | None]]:
    return [
        ("none", 1, None, None),
        ("all", 2, 0.1, 0.8),
        ("all", 2, 0.1, 1.0),
        ("all", 3, 0.1, 0.8),
    ]


def _best_config_from_env() -> tuple[str, int, float | None, float | None] | None:
    path_value = os.environ.get("PFC_STAGE2C_BEST_CONFIG_JSON")
    if not path_value:
        return None
    data = json.loads(Path(path_value).read_text(encoding="utf-8"))
    row = data.get("best_quality_row", data)
    return (
        str(row.get("cache_layers", "all")),
        int(row.get("cache_interval", 2)),
        None if row.get("active_t_min") is None else float(row["active_t_min"]),
        None if row.get("active_t_max") is None else float(row["active_t_max"]),
    )


def validation_configs() -> list[tuple[str, int, float | None, float | None]]:
    configs = _base_validation_configs()
    best = _best_config_from_env()
    if best and best not in configs:
        configs.append(best)
    return configs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# JiT Stage 2C Validation",
        "",
        "| layers | t min | t max | interval | speedup | hit rate | rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cache_layers} | {active_t_min} | {active_t_max} | {cache_interval} | "
            "{speedup_median:.4f} | {cache_hit_rate:.4f} | {same_seed_rel_l2:.6f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base, validate_dir, preview_root = base_config_from_env()
    validate_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for layers, interval, t_min, t_max in validation_configs():
        run_id = f"{base.run_id}_{layers.replace(':', '-')}_i{interval}_{_window_label(t_min, t_max)}"
        config = replace(
            base,
            run_id=run_id,
            run_dir=validate_dir / "runs" / run_id,
            preview_dir=preview_root / run_id,
            cache_layers=layers,
            cache_interval=interval,
            active_t_min=t_min,
            active_t_max=t_max,
            save_previews=False,
            diag_full_probe=False,
        )
        result = run_experiment(config)
        rows.append(result_row_from_stage2b(result, config, "validation"))
    _write_csv(validate_dir / "validation_results.csv", rows)
    (validate_dir / "validation_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(validate_dir / "summary.md", rows)
    print(f"JiT Stage 2C validation dir: {validate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
