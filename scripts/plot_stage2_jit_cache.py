#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/pfc_matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/pfc_xdg_cache"))
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing grid results CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _ensure_output_dir() -> Path:
    out_dir = ROOT / "outputs/stage2/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_by_interval(rows: list[dict[str, Any]], y_key: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        interval = _float(row, "cache_interval")
        value = _float(row, y_key)
        if interval is None or value is None:
            continue
        grouped[str(row.get("cache_layers", "unknown"))].append((interval, value))
    if not grouped:
        print(f"Skipping {out_path.name}: missing cache_interval/{y_key}")
        return
    plt.figure()
    for layer_name, values in sorted(grouped.items()):
        values.sort()
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        plt.plot(xs, ys, marker="o", label=layer_name)
    plt.xlabel("Cache interval")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_quality_vs_speedup(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    points = []
    labels = []
    for row in rows:
        speedup = _float(row, "speedup")
        rel_l2 = _float(row, "same_seed_rel_l2")
        if speedup is None or rel_l2 is None:
            continue
        points.append((speedup, rel_l2))
        labels.append(f"{row.get('cache_layers')} i{row.get('cache_interval')}")
    if not points:
        print(f"Skipping {out_path.name}: missing speedup/same_seed_rel_l2")
        return
    plt.figure()
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    plt.scatter(xs, ys)
    for label, x, y in zip(labels, xs, ys):
        plt.annotate(label, (x, y), fontsize=8)
    plt.xlabel("Speedup")
    plt.ylabel("Same-seed rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_rel_l2_by_layers(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rel_l2 = _float(row, "same_seed_rel_l2")
        if rel_l2 is not None:
            grouped[str(row.get("cache_layers", "unknown"))].append(rel_l2)
    if not grouped:
        print(f"Skipping {out_path.name}: missing same_seed_rel_l2")
        return
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Cache layers")
    plt.ylabel("Mean same-seed rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = _read_rows(args.grid_dir / "grid_results.csv")
    out_dir = _ensure_output_dir()
    _plot_by_interval(rows, "speedup", "Speedup", out_dir / "jit_speedup_by_interval.png")
    _plot_quality_vs_speedup(rows, out_dir / "jit_quality_vs_speedup.png")
    _plot_by_interval(rows, "cache_hit_rate", "Cache hit rate", out_dir / "jit_hit_rate_by_interval.png")
    _plot_rel_l2_by_layers(rows, out_dir / "jit_rel_l2_by_layers.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
