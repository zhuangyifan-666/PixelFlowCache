#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for env_name, default in {"MPLCONFIGDIR": "/tmp/pfc_matplotlib", "XDG_CACHE_HOME": "/tmp/pfc_xdg_cache"}.items():
    path = Path(os.environ.get(env_name, default))
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(path))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Skipping missing CSV: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None, "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _out_dir() -> Path:
    path = ROOT / "outputs/stage3a/figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_speed_quality(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    points, labels = [], []
    for row in rows:
        if row.get("method_type") == "reference":
            continue
        speedup = _float(row, "speedup_mean")
        rel_l2 = _float(row, "rel_l2_mean")
        if speedup is None or rel_l2 is None:
            continue
        points.append((speedup, rel_l2))
        labels.append(str(row["method_name"]))
    if not points:
        print(f"Skipping {out_path.name}: no speed/quality data")
        return
    plt.figure()
    plt.scatter([item[0] for item in points], [item[1] for item in points])
    for label, (x_value, y_value) in zip(labels, points):
        plt.annotate(label, (x_value, y_value), fontsize=8)
    plt.xlabel("Speedup vs 50-step reference")
    plt.ylabel("Same-seed rel-L2 mean")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_bar(rows: list[dict[str, Any]], value_key: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels, values = [], []
    for row in rows:
        value = _float(row, value_key)
        if value is None:
            continue
        labels.append(str(row["method_name"]))
        values.append(value)
    if not values:
        print(f"Skipping {out_path.name}: no {value_key}")
        return
    xs = list(range(len(labels)))
    plt.figure()
    plt.bar(xs, values)
    plt.ylabel(ylabel)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_frequency(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels, low, mid, high = [], [], [], []
    for row in rows:
        if row.get("method_type") == "reference":
            continue
        values = [_float(row, key) for key in ["low_freq_delta_ratio", "mid_freq_delta_ratio", "high_freq_delta_ratio"]]
        if any(value is None for value in values):
            continue
        labels.append(str(row["method_name"]))
        low.append(float(values[0]))
        mid.append(float(values[1]))
        high.append(float(values[2]))
    if not labels:
        print(f"Skipping {out_path.name}: no frequency data")
        return
    xs = list(range(len(labels)))
    plt.figure()
    plt.bar(xs, low, label="low")
    plt.bar(xs, mid, bottom=low, label="mid")
    bottoms = [a + b for a, b in zip(low, mid)]
    plt.bar(xs, high, bottom=bottoms, label="high")
    plt.ylabel("Frequency delta ratio")
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--reduced-dir", type=Path)
    parser.add_argument("--larger-dir", type=Path)
    args = parser.parse_args()
    if not (args.benchmark_dir or args.reduced_dir or args.larger_dir):
        parser.error("Pass at least one input directory")
    out = _out_dir()
    if args.benchmark_dir:
        aggregate_rows = _read_csv(args.benchmark_dir / "benchmark_aggregate.csv")
        result_rows = _read_csv(args.benchmark_dir / "benchmark_results.csv")
        _plot_speed_quality(aggregate_rows, out / "jit_stage3a_speed_quality_cache_vs_reduced.png")
        _plot_bar(aggregate_rows, "rel_l2_mean", "Same-seed rel-L2 mean", out / "jit_stage3a_rel_l2_by_method.png")
        _plot_bar(aggregate_rows, "speedup_mean", "Speedup vs reference mean", out / "jit_stage3a_speedup_by_method.png")
        _plot_bar(aggregate_rows, "hit_rate_mean", "Cache hit rate mean", out / "jit_stage3a_cache_hit_rate_by_method.png")
        _plot_frequency(result_rows, out / "jit_stage3a_frequency_delta_by_method.png")
    if args.reduced_dir:
        _read_csv(args.reduced_dir / "reduced_step_results.csv")
    if args.larger_dir:
        _read_csv(args.larger_dir / "benchmark_aggregate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
