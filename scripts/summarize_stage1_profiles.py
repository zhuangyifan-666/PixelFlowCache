#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_features(run_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in _read_jsonl(run_dir / "feature_stats.jsonl"):
        delta = record.get("temporal_delta") or {}
        rel = delta.get("rel_l2_delta")
        if isinstance(rel, (int, float)):
            grouped[record.get("module_name", "unknown")].append(float(rel))
    rows = []
    for module_name, values in sorted(grouped.items()):
        rows.append(
            {
                "module_name": module_name,
                "count": len(values),
                "mean_rel_l2_delta": statistics.fmean(values),
                "median_rel_l2_delta": statistics.median(values),
                "min_rel_l2_delta": min(values),
                "max_rel_l2_delta": max(values),
            }
        )
    return rows


def summarize_velocity(run_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    amp_by_step: dict[int, list[float]] = defaultdict(list)
    for record in _read_jsonl(run_dir / "velocity_stats.jsonl"):
        step = int(record.get("step_idx", -1))
        branch = str(record.get("branch", "unknown"))
        v_stats = record.get("v") or {}
        v_l2 = v_stats.get("l2")
        if isinstance(v_l2, (int, float)):
            grouped[(step, branch)].append(float(v_l2))
        amplification = record.get("amplification")
        if amplification is None:
            amplification = (record.get("extra") or {}).get("amplification")
        if isinstance(amplification, (int, float)):
            amp_by_step[step].append(float(amplification))

    rows = []
    for (step, branch), values in sorted(grouped.items()):
        row = {
            "step_idx": step,
            "branch": branch,
            "count": len(values),
            "mean_v_l2": statistics.fmean(values),
            "median_v_l2": statistics.median(values),
        }
        if amp_by_step.get(step):
            row["mean_amplification"] = statistics.fmean(amp_by_step[step])
        rows.append(row)
    return rows


def summarize_frequency(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for record in _read_jsonl(run_dir / "frequency_stats.jsonl"):
        freq = record.get("frequency") or {}
        rows.append(
            {
                "step_idx": record.get("step_idx"),
                "branch": record.get("branch", "cfg"),
                "low_ratio": freq.get("low_ratio"),
                "mid_ratio": freq.get("mid_ratio"),
                "high_ratio": freq.get("high_ratio"),
                "high_to_low": freq.get("high_to_low"),
                "total_energy": freq.get("total_energy"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    feature_rows = summarize_features(run_dir)
    velocity_rows = summarize_velocity(run_dir)
    frequency_rows = summarize_frequency(run_dir)

    _write_csv(run_dir / "feature_delta_by_module.csv", feature_rows)
    _write_csv(run_dir / "velocity_by_step.csv", velocity_rows)
    _write_csv(run_dir / "frequency_by_step.csv", frequency_rows)

    summary = {
        "run_dir": str(run_dir),
        "feature_modules": len(feature_rows),
        "velocity_rows": len(velocity_rows),
        "frequency_rows": len(frequency_rows),
        "feature_delta_by_module": feature_rows,
        "velocity_by_step": velocity_rows,
        "frequency_by_step": frequency_rows,
        "top_smooth_modules": sorted(feature_rows, key=lambda row: row["mean_rel_l2_delta"])[:10],
    }
    existing_summary_path = run_dir / "summary.json"
    if existing_summary_path.exists():
        try:
            existing = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            existing.update(summary)
            summary = existing
        except json.JSONDecodeError:
            pass
    existing_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote Stage 1 summaries for {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
