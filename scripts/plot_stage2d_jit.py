#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for env_name, default in {"MPLCONFIGDIR": "/tmp/pfc_matplotlib", "XDG_CACHE_HOME": "/tmp/pfc_xdg_cache"}.items():
    path = Path(os.environ.get(env_name, default))
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(path))


def _out_dir() -> Path:
    path = ROOT / "outputs/stage2d/figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Skipping missing CSV: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"Skipping missing JSON: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None, "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label(row: dict[str, Any]) -> str:
    return f"{row.get('cache_layers')}/i{row.get('cache_interval')} [{row.get('active_t_min')},{row.get('active_t_max')})"


def _plot_validate_speed_quality(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    points = []
    labels = []
    for row in rows:
        speedup = _float(row, "speedup_median")
        rel_l2 = _float(row, "same_seed_rel_l2")
        if speedup is None or rel_l2 is None:
            continue
        points.append((speedup, rel_l2))
        labels.append(_label(row))
    if not points:
        print(f"Skipping {out_path.name}: no validation rows")
        return
    plt.figure()
    plt.scatter([item[0] for item in points], [item[1] for item in points])
    for label, (x_value, y_value) in zip(labels, points):
        plt.annotate(label, (x_value, y_value), fontsize=8)
    plt.xlabel("Median speedup")
    plt.ylabel("Same-seed rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_seed_summary(summary: dict[str, Any], metric: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary.get("summaries") or []
    labels = [str(row["config_label"]) for row in rows]
    means = [row.get(f"{metric}_mean") for row in rows]
    stds = [row.get(f"{metric}_std") for row in rows]
    if not labels or any(value is None for value in means):
        print(f"Skipping {out_path.name}: no seed summary for {metric}")
        return
    xs = list(range(len(labels)))
    plt.figure()
    plt.bar(xs, [float(value) for value in means], yerr=[float(value or 0.0) for value in stds], capsize=4)
    plt.ylabel(ylabel)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_first_hit(rows: list[dict[str, Any]], metric: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    points = []
    for row in rows:
        warmup = _float(row, "active_window_warmup_refreshes")
        value = _float(row, metric)
        if warmup is not None and value is not None:
            points.append((warmup, value))
    if not points:
        print(f"Skipping {out_path.name}: no first-hit rows")
        return
    points.sort()
    plt.figure()
    plt.plot([item[0] for item in points], [item[1] for item in points], marker="o")
    plt.xlabel("Active-window warmup refreshes")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-dir", type=Path)
    parser.add_argument("--seed-sweep-dir", type=Path)
    parser.add_argument("--first-hit-dir", type=Path)
    args = parser.parse_args()
    if not (args.validate_dir or args.seed_sweep_dir or args.first_hit_dir):
        parser.error("Pass at least one input directory")

    out = _out_dir()
    if args.validate_dir:
        rows = _read_csv(args.validate_dir / "validation_results.csv")
        _plot_validate_speed_quality(rows, out / "jit_stage2d_validate_speed_quality.png")

    if args.seed_sweep_dir:
        summary = _read_json(args.seed_sweep_dir / "seed_sweep_results.json")
        if summary:
            _plot_seed_summary(
                summary,
                "same_seed_rel_l2",
                "Same-seed rel-L2 mean +/- std",
                out / "jit_stage2d_seed_rel_l2_mean_std.png",
            )
            _plot_seed_summary(
                summary,
                "speedup_median",
                "Median speedup mean +/- std",
                out / "jit_stage2d_seed_speedup_mean_std.png",
            )

    if args.first_hit_dir:
        rows = _read_csv(args.first_hit_dir / "first_hit_delay_results.csv")
        _plot_first_hit(
            rows,
            "same_seed_rel_l2",
            "Same-seed rel-L2",
            out / "jit_stage2d_first_hit_delay_rel_l2.png",
        )
        _plot_first_hit(
            rows,
            "speedup_median",
            "Median speedup",
            out / "jit_stage2d_first_hit_delay_speedup.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
