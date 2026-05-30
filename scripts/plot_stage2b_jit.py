#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for env_name, default in {"MPLCONFIGDIR": "/tmp/pfc_matplotlib", "XDG_CACHE_HOME": "/tmp/pfc_xdg_cache"}.items():
    path = Path(os.environ.get(env_name, default))
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(path))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None, "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _out_dir() -> Path:
    path = ROOT / "outputs/stage2b/figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_speed_quality(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    points = []
    labels = []
    for row in rows:
        speedup = _float(row, "speedup_median")
        rel_l2 = _float(row, "same_seed_rel_l2")
        if speedup is None or rel_l2 is None:
            continue
        points.append((speedup, rel_l2))
        labels.append(str(row.get("cache_layers", "")))
    if not points:
        print(f"Skipping {out_path.name}: no speed/quality data")
        return
    plt.figure()
    plt.scatter([p[0] for p in points], [p[1] for p in points])
    for label, (x, y) in zip(labels, points):
        plt.annotate(label, (x, y), fontsize=8)
    plt.xlabel("Median speedup")
    plt.ylabel("Same-seed rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _window_key(row: dict[str, Any]) -> str:
    return f"{row.get('active_t_min')}-{row.get('active_t_max')}"


def _bar_mean(rows: list[dict[str, Any]], group_fn: Any, value_key: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row, value_key)
        if value is not None:
            grouped[group_fn(row)].append(value)
    if not grouped:
        print(f"Skipping {out_path.name}: missing {value_key}")
        return
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Group")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _read_step_errors_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir", "")))
        path = run_dir / "step_error_stats.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _plot_velocity_error_by_step(records: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        error = record.get("trajectory_error") or {}
        value = error.get("rel_l2")
        if isinstance(value, (int, float)):
            grouped[int(record["step_idx"])].append(float(value))
    if not grouped:
        print(f"Skipping {out_path.name}: no step error data")
        return
    xs = sorted(grouped)
    ys = [sum(grouped[idx]) / len(grouped[idx]) for idx in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Mean trajectory velocity rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_amplification_vs_error(records: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    xs, ys = [], []
    for record in records:
        error = record.get("trajectory_error") or {}
        value = error.get("rel_l2")
        amplification = record.get("amplification")
        if isinstance(value, (int, float)) and isinstance(amplification, (int, float)):
            xs.append(float(amplification))
            ys.append(float(value))
    if not xs:
        print(f"Skipping {out_path.name}: no amplification/error data")
        return
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("1 / max(1 - t, eps)")
    plt.ylabel("Trajectory velocity rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    if bool(args.sweep_dir) == bool(args.validation_dir):
        parser.error("Pass exactly one of --sweep-dir or --validation-dir")
    source_dir = args.sweep_dir or args.validation_dir
    assert source_dir is not None
    csv_name = "sweep_results.csv" if args.sweep_dir else "validation_results.csv"
    rows = _read_csv(source_dir / csv_name)
    out = _out_dir()
    _plot_speed_quality(rows, out / "jit_stage2b_speed_quality.png")
    _bar_mean(rows, _window_key, "speedup_median", "Mean median speedup", out / "jit_stage2b_speedup_by_window.png")
    _bar_mean(rows, _window_key, "same_seed_rel_l2", "Mean same-seed rel-L2", out / "jit_stage2b_rel_l2_by_window.png")
    _bar_mean(
        rows,
        lambda row: str(row.get("cache_layers", "")),
        "same_seed_rel_l2",
        "Mean same-seed rel-L2",
        out / "jit_stage2b_rel_l2_by_layer_group.png",
    )
    step_records = _read_step_errors_from_rows(rows)
    _plot_velocity_error_by_step(step_records, out / "jit_stage2b_velocity_error_by_step.png")
    _plot_amplification_vs_error(step_records, out / "jit_stage2b_amplification_vs_error.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
