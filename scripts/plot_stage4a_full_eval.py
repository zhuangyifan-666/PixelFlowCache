#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for env_name, default in {"MPLCONFIGDIR": "/tmp/pfc_matplotlib", "XDG_CACHE_HOME": "/tmp/pfc_xdg_cache"}.items():
    path = Path(os.environ.get(env_name, default))
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(path))


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _theta_from_run_id(run_id: str) -> str | None:
    match = re.search(r"theta(\d+)p(\d+)", run_id)
    if not match:
        return None
    return f"{int(match.group(1))}.{match.group(2)}"


def _label(row: dict[str, Any]) -> str:
    model = str(row.get("model") or "")
    method = str(row.get("method") or "")
    run_id = str(row.get("run_id") or "")
    theta = _theta_from_run_id(run_id)
    if method == "seacache_style" and theta:
        return f"{model} SeaCache-style θ={theta}"
    if method == "teacache_style" and theta:
        return f"{model} TeaCache-style θ={theta}"
    return f"{model}:{method}"


def _int_value(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value in ("", None, "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def filter_rows_for_plot(
    rows: list[dict[str, str]],
    *,
    run_id: str | None = None,
    num_images: int | None = None,
    warn: bool = True,
) -> list[dict[str, str]]:
    filtered = [row for row in rows if not run_id or row.get("run_id") == run_id]
    available_num_images = sorted({value for row in filtered if (value := _int_value(row, "num_images")) is not None})
    selected_num_images = num_images
    if selected_num_images is None and len(available_num_images) > 1:
        selected_num_images = available_num_images[-1]
        if warn:
            print(
                "Warning: summary contains multiple num_images values "
                f"{available_num_images}; plotting only {selected_num_images}. "
                "Pass --num-images to choose explicitly.",
                file=sys.stderr,
            )
    if selected_num_images is not None:
        filtered = [row for row in filtered if _int_value(row, "num_images") == selected_num_images]
    return filtered


def _scatter(rows: list[dict[str, str]], y_key: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        x = _float(row, "speedup_vs_no_cache")
        y = _float(row, y_key)
        if x is None or y is None:
            continue
        ax.scatter(x, y)
        ax.annotate(_label(row), (x, y), fontsize=8)
    ax.set_xlabel("Speedup vs no_cache_50")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _bar(rows: list[dict[str, str]], key: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels, values = [], []
    for row in rows:
        value = _float(row, key)
        if value is None:
            continue
        labels.append(_label(row))
        values.append(value)
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(len(labels)))
    ax.bar(xs, values)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stage4a(
    summary_dir: Path,
    output_dir: Path | None = None,
    *,
    run_id: str | None = None,
    num_images: int | None = None,
) -> list[Path]:
    rows = _read_csv(summary_dir / "stage4a_results.csv")
    rows = filter_rows_for_plot(rows, run_id=run_id, num_images=num_images)
    if not rows:
        raise ValueError("No Stage 4A rows remain after plot filters.")
    out = output_dir or ROOT / "outputs/stage4a/figures"
    paths = [
        out / "stage4a_fid_vs_speedup.png",
        out / "stage4a_is_vs_speedup.png",
        out / "stage4a_latency_by_method.png",
        out / "stage4a_cache_hit_rate_by_method.png",
    ]
    _scatter(rows, "FID", "FID", paths[0])
    _scatter(rows, "IS", "Inception Score", paths[1])
    _bar(rows, "latency_sec", "Latency seconds", paths[2])
    _bar(rows, "cache_hit_rate", "Cache hit rate", paths[3])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Stage 4A full evaluation summary.")
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--num-images", type=int)
    args = parser.parse_args()
    for path in plot_stage4a(
        args.summary_dir.resolve(),
        args.output_dir.resolve() if args.output_dir else None,
        run_id=args.run_id,
        num_images=args.num_images,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
