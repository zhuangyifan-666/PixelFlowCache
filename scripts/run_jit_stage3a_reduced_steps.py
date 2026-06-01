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

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_jit_stage2b_cache import Stage2BConfig, _detect_jit_ckpt_dir, _load_jit_model, _make_inputs, _time_repeats  # noqa: E402
from scripts.run_jit_stage3a_backbone_benchmark import _comparison_fields, _parse_int_list  # noqa: E402


REDUCED_STEP_FIELDNAMES = [
    "method",
    "reference_steps",
    "eval_steps",
    "num_samples",
    "seed",
    "latency_median_sec",
    "reference_latency_median_sec",
    "speedup_vs_reference",
    "same_seed_rel_l2",
    "same_seed_mse",
    "same_seed_psnr",
    "low_freq_delta_ratio",
    "mid_freq_delta_ratio",
    "high_freq_delta_ratio",
]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _make_run_id(seed: int, reference_steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_ref{reference_steps}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REDUCED_STEP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _base_config(run_id: str, out_dir: Path, reference_steps: int) -> Stage2BConfig:
    seed = _env_int("PFC_STAGE3A_SEED", 0)
    return Stage2BConfig(
        jit_dir=Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve(),
        ckpt_dir=_detect_jit_ckpt_dir(),
        run_id=run_id,
        run_dir=out_dir,
        preview_dir=ROOT / "outputs/stage3a/previews/jit_reduced_steps" / run_id,
        model=os.environ.get("PFC_STAGE3A_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE3A_IMG_SIZE", 256),
        num_samples=_env_int("PFC_STAGE3A_NUM_SAMPLES", 32),
        batch_size=_env_int("PFC_STAGE3A_BATCH_SIZE", 4),
        steps=reference_steps,
        seed=seed,
        cfg=_env_float("PFC_STAGE3A_CFG", 3.0),
        interval_min=_env_float("PFC_STAGE3A_CFG_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE3A_CFG_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=1,
        cache_layers="none",
        active_t_min=None,
        active_t_max=None,
        timing_repeats=_env_int("PFC_STAGE3A_TIMING_REPEATS", 3),
        warmup_runs=_env_int("PFC_STAGE3A_WARMUP_RUNS", 1),
        save_previews=False,
    )


def _run_no_cache(config: Stage2BConfig, labels: torch.Tensor, noise: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model = _load_jit_model(config, device)
    timing = _time_repeats(model, labels, noise, config)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return timing


def summarize_reduced_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_quality = min(rows, key=lambda row: (float(row["same_seed_rel_l2"]), -float(row["speedup_vs_reference"]))) if rows else None
    fastest = max(rows, key=lambda row: float(row["speedup_vs_reference"])) if rows else None
    return {
        "record_count": len(rows),
        "best_quality_row": best_quality,
        "fastest_row": fastest,
        "rows": rows,
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# JiT Stage 3A Reduced-Step No-Cache Baseline",
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


def main() -> int:
    reference_steps = _env_int("PFC_STAGE3A_REFERENCE_STEPS", 50)
    seed = _env_int("PFC_STAGE3A_SEED", 0)
    run_id = os.environ.get("PFC_STAGE3A_REDUCED_RUN_ID", _make_run_id(seed, reference_steps))
    out_dir = Path(os.environ.get("PFC_STAGE3A_REDUCED_DIR", ROOT / "logs/stage3a/jit_reduced_steps" / run_id)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reduced_steps = _parse_int_list(os.environ.get("PFC_STAGE3A_REDUCED_STEPS", "30,35,40"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = _base_config(run_id, out_dir, reference_steps)
    labels, noise = _make_inputs(base, device)
    reference_timing = _run_no_cache(base, labels, noise, device)
    reference_output = reference_timing["output"]
    reference_latency = float(reference_timing["latency_median_sec"])
    rows: list[dict[str, Any]] = []
    for steps in reduced_steps:
        config = replace(base, steps=steps, run_id=f"{run_id}_steps{steps}", run_dir=out_dir / "runs" / f"steps{steps}")
        timing = _run_no_cache(config, labels, noise, device)
        latency = float(timing["latency_median_sec"])
        comparison = _comparison_fields(reference_output, timing["output"], reference_latency, latency)
        rows.append(
            {
                "method": "reduced_steps",
                "reference_steps": reference_steps,
                "eval_steps": steps,
                "num_samples": config.num_samples,
                "seed": seed,
                "latency_median_sec": latency,
                "reference_latency_median_sec": reference_latency,
                **comparison,
            }
        )
    _write_csv(out_dir / "reduced_step_results.csv", rows)
    (out_dir / "reduced_step_results.json").write_text(
        json.dumps(summarize_reduced_rows(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(out_dir / "summary.md", rows)
    print(f"JiT Stage 3A reduced-step dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
