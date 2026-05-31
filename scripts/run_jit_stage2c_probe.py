#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _make_run_id(seed: int, steps: int, cache_interval: int, cache_layers: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_layers = "".join(ch if ch.isalnum() else "-" for ch in cache_layers)[:48].strip("-")
    return f"{stamp}_seed{seed}_steps{steps}_i{cache_interval}_{safe_layers or 'layers'}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_var = sum(value * value for value in x_centered)
    y_var = sum(value * value for value in y_centered)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum(x * y for x, y in zip(x_centered, y_centered))
    return cov / math.sqrt(x_var * y_var)


def summarize_probe_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, dict[str, list[float]]] = {}
    trajectory_values: list[float] = []
    probe_values: list[float] = []
    amplification_for_probe: list[float] = []

    for record in records:
        step_idx = int(record["step_idx"])
        step = by_step.setdefault(step_idx, {"trajectory": [], "probe": [], "amplification": []})
        trajectory = _numeric((record.get("trajectory_error") or {}).get("rel_l2"))
        probe = _numeric((record.get("probe_error") or {}).get("rel_l2"))
        amplification = _numeric(record.get("amplification"))
        if trajectory is not None:
            step["trajectory"].append(trajectory)
            trajectory_values.append(trajectory)
        if probe is not None:
            step["probe"].append(probe)
            probe_values.append(probe)
            if amplification is not None:
                amplification_for_probe.append(amplification)
        if amplification is not None:
            step["amplification"].append(amplification)

    step_means = []
    for step_idx in sorted(by_step):
        values = by_step[step_idx]
        step_means.append(
            {
                "step_idx": step_idx,
                "trajectory_rel_l2_mean": _mean(values["trajectory"]),
                "probe_rel_l2_mean": _mean(values["probe"]),
                "amplification_mean": _mean(values["amplification"]),
                "record_count": len(values["trajectory"]),
                "probe_record_count": len(values["probe"]),
            }
        )

    mean_trajectory = _mean(trajectory_values)
    mean_probe = _mean(probe_values)
    if mean_probe is None or mean_trajectory is None:
        dominance = "insufficient_probe_data"
    elif mean_probe > mean_trajectory:
        dominance = "local_error_dominates"
    else:
        dominance = "accumulated_trajectory_drift_dominates"

    return {
        "record_count": len(records),
        "probe_record_count": len(probe_values),
        "step_means": step_means,
        "mean_trajectory_rel_l2": mean_trajectory,
        "mean_probe_rel_l2": mean_probe,
        "max_probe_rel_l2": max(probe_values) if probe_values else None,
        "correlation_amplification_probe_rel_l2": pearson_correlation(amplification_for_probe, probe_values),
        "dominance": dominance,
    }


def base_config_from_env() -> Stage2BConfig:
    seed = _env_int("PFC_STAGE2C_SEED", 0)
    steps = _env_int("PFC_STAGE2C_STEPS", 20)
    cache_interval = _env_int("PFC_STAGE2C_CACHE_INTERVAL", 2)
    cache_layers = os.environ.get("PFC_STAGE2C_CACHE_LAYERS", "all")
    run_id = os.environ.get("PFC_STAGE2C_PROBE_RUN_ID", _make_run_id(seed, steps, cache_interval, cache_layers))
    run_dir = Path(os.environ.get("PFC_STAGE2C_PROBE_DIR", ROOT / "logs/stage2c/jit_probe" / run_id)).resolve()
    preview_dir = Path(
        os.environ.get("PFC_STAGE2C_PREVIEW_DIR", ROOT / "outputs/stage2c/previews/jit_probe" / run_id)
    ).resolve()
    return Stage2BConfig(
        jit_dir=Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve(),
        ckpt_dir=_detect_jit_ckpt_dir(),
        run_id=run_id,
        run_dir=run_dir,
        preview_dir=preview_dir,
        model=os.environ.get("PFC_STAGE2C_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE2C_IMG_SIZE", 256),
        num_samples=_env_int("PFC_STAGE2C_NUM_SAMPLES", 4),
        batch_size=_env_int("PFC_STAGE2C_BATCH_SIZE", 4),
        steps=steps,
        seed=seed,
        cfg=_env_float("PFC_STAGE2C_CFG", 3.0),
        interval_min=_env_float("PFC_STAGE2C_CFG_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE2C_CFG_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=cache_interval,
        cache_layers=cache_layers,
        cache_branches=os.environ.get("PFC_STAGE2C_CACHE_BRANCHES", "cond,uncond"),
        active_t_min=_env_float("PFC_STAGE2C_ACTIVE_T_MIN", 0.1),
        active_t_max=_env_float("PFC_STAGE2C_ACTIVE_T_MAX", 0.8),
        timing_repeats=_env_int("PFC_STAGE2C_TIMING_REPEATS", 3),
        warmup_runs=_env_int("PFC_STAGE2C_WARMUP_RUNS", 1),
        diag_full_probe=True,
        diag_probe_steps=os.environ.get("PFC_STAGE2C_DIAG_PROBE_STEPS", "all"),
        save_previews=False,
    )


def _write_summary(path: Path, config: Stage2BConfig, summary: dict[str, Any], result: dict[str, Any]) -> None:
    comparison = result["comparison"]
    cache_stats = result["cache_stats"]
    lines = [
        "# JiT Stage 2C Full-Probe Diagnostics",
        "",
        f"Run dir: `{config.run_dir}`",
        f"Cache setting: layers=`{config.cache_layers}`, interval={config.cache_interval}, "
        f"t=[{config.active_t_min}, {config.active_t_max})",
        "",
        f"Median speedup from main timing: {comparison['speedup_median']:.4f}",
        f"Cache hit rate: {cache_stats['hit_rate']:.4f}",
        f"Final same-seed rel-L2: {comparison['same_seed_rel_l2']:.6f}",
        "",
        f"Mean trajectory rel-L2: {summary['mean_trajectory_rel_l2']}",
        f"Mean local probe rel-L2: {summary['mean_probe_rel_l2']}",
        f"Max local probe rel-L2: {summary['max_probe_rel_l2']}",
        f"Correlation amplification vs local probe rel-L2: {summary['correlation_amplification_probe_rel_l2']}",
        f"Dominance: {summary['dominance']}",
        "",
        "| step | trajectory rel-L2 | local probe rel-L2 | amplification |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["step_means"]:
        lines.append(
            "| {step_idx} | {trajectory_rel_l2_mean} | {probe_rel_l2_mean} | {amplification_mean} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = base_config_from_env()
    config = replace(config, diag_full_probe=True, save_previews=False)
    result = run_experiment(config)
    records = _read_jsonl(config.run_dir / "step_error_stats.jsonl")
    summary = summarize_probe_records(records)
    (config.run_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(config.run_dir / "summary.md", config, summary, result)
    print(f"JiT Stage 2C probe dir: {config.run_dir}")
    print(f"Probe dominance: {summary['dominance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
