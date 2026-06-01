#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["method_name"]) for row in rows]


def _save_bar(rows: list[dict[str, Any]], key: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = _labels(rows)
    values = [_float(row, key) for row in rows]
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_speed_quality(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        ax.scatter(_float(row, "speedup_mean"), _float(row, "rel_l2_mean"))
        ax.annotate(str(row["method_name"]), (_float(row, "speedup_mean"), _float(row, "rel_l2_mean")), fontsize=8)
    ax.set_xlabel("Speedup vs reference")
    ax.set_ylabel("Same-seed rel-L2")
    ax.set_title("DeCo Stage 3B speed-quality")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _mean_by_method(rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            grouped.setdefault(row["method_name"], []).append(float(value))
        except ValueError:
            continue
    output = []
    for method_name in sorted(grouped):
        values = grouped[method_name]
        output.append({"method_name": method_name, key: sum(values) / len(values)})
    return output


def plot_stage3b_deco(benchmark_dir: Path, output_dir: Path) -> list[Path]:
    aggregate_rows = _read_csv(benchmark_dir / "benchmark_aggregate.csv")
    result_rows = _read_csv(benchmark_dir / "benchmark_results.csv")
    output_paths = [
        output_dir / "deco_stage3b_speed_quality.png",
        output_dir / "deco_stage3b_rel_l2_by_method.png",
        output_dir / "deco_stage3b_speedup_by_method.png",
        output_dir / "deco_stage3b_cache_hit_rate_by_method.png",
        output_dir / "deco_stage3b_frequency_delta_by_method.png",
    ]
    _save_speed_quality(aggregate_rows, output_paths[0])
    _save_bar(aggregate_rows, "rel_l2_mean", "Same-seed rel-L2", "DeCo Stage 3B rel-L2 by method", output_paths[1])
    _save_bar(aggregate_rows, "speedup_mean", "Speedup vs reference", "DeCo Stage 3B speedup by method", output_paths[2])
    _save_bar(aggregate_rows, "hit_rate_mean", "Cache hit rate", "DeCo Stage 3B cache hit rate by method", output_paths[3])
    frequency_rows = _mean_by_method(result_rows, "high_freq_delta_ratio")
    _save_bar(frequency_rows, "high_freq_delta_ratio", "High-frequency delta ratio", "DeCo Stage 3B frequency delta by method", output_paths[4])
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/stage3b/figures")
    args = parser.parse_args()
    paths = plot_stage3b_deco(args.benchmark_dir.resolve(), args.output_dir.resolve())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
