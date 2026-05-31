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
    path = ROOT / "outputs/stage2c/figures"
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


def _plot_window_line(
    rows: list[dict[str, Any]],
    group: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    points = []
    for row in rows:
        if row.get("ablation_group") != group:
            continue
        x_value = _float(row, x_key)
        y_value = _float(row, y_key)
        if x_value is not None and y_value is not None:
            points.append((x_value, y_value))
    if not points:
        print(f"Skipping {out_path.name}: no {group} data")
        return
    points.sort()
    plt.figure()
    plt.plot([item[0] for item in points], [item[1] for item in points], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_probe_error(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary.get("step_means") or []
    xs, trajectory, probe = [], [], []
    for row in rows:
        step_idx = row.get("step_idx")
        trajectory_value = row.get("trajectory_rel_l2_mean")
        probe_value = row.get("probe_rel_l2_mean")
        if isinstance(step_idx, int) and isinstance(trajectory_value, (int, float)):
            xs.append(step_idx)
            trajectory.append(float(trajectory_value))
            probe.append(float(probe_value) if isinstance(probe_value, (int, float)) else None)
    if not xs:
        print(f"Skipping {out_path.name}: no probe step data")
        return
    plt.figure()
    plt.plot(xs, trajectory, marker="o", label="trajectory")
    probe_points = [(x, y) for x, y in zip(xs, probe) if y is not None]
    if probe_points:
        plt.plot([item[0] for item in probe_points], [item[1] for item in probe_points], marker="o", label="local probe")
    plt.xlabel("Step")
    plt.ylabel("Mean velocity rel-L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_probe_amplification(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    xs, ys = [], []
    for row in summary.get("step_means") or []:
        amplification = row.get("amplification_mean")
        probe_value = row.get("probe_rel_l2_mean")
        if isinstance(amplification, (int, float)) and isinstance(probe_value, (int, float)):
            xs.append(float(amplification))
            ys.append(float(probe_value))
    if not xs:
        print(f"Skipping {out_path.name}: no amplification/probe data")
        return
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Mean amplification")
    plt.ylabel("Mean local probe rel-L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(out_path)


def _plot_validation_speed_quality(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    points = []
    labels = []
    for row in rows:
        speedup = _float(row, "speedup_median")
        rel_l2 = _float(row, "same_seed_rel_l2")
        if speedup is None or rel_l2 is None:
            continue
        points.append((speedup, rel_l2))
        labels.append(f"{row.get('cache_layers')} i{row.get('cache_interval')} [{row.get('active_t_min')},{row.get('active_t_max')})")
    if not points:
        print(f"Skipping {out_path.name}: no validation speed/quality data")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-dir", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    if not (args.window_dir or args.probe_dir or args.validation_dir):
        parser.error("Pass at least one of --window-dir, --probe-dir, or --validation-dir")

    out = _out_dir()
    if args.window_dir:
        rows = _read_csv(args.window_dir / "window_ablation_results.csv")
        _plot_window_line(
            rows,
            "t_min",
            "active_t_min",
            "same_seed_rel_l2",
            "active_t_min at active_t_max=0.8",
            "Same-seed rel-L2",
            out / "jit_tmin_ablation_rel_l2.png",
        )
        _plot_window_line(
            rows,
            "t_min",
            "active_t_min",
            "speedup_median",
            "active_t_min at active_t_max=0.8",
            "Median speedup",
            out / "jit_tmin_ablation_speedup.png",
        )
        _plot_window_line(
            rows,
            "t_max",
            "active_t_max",
            "same_seed_rel_l2",
            "active_t_max at active_t_min=0.1",
            "Same-seed rel-L2",
            out / "jit_tmax_ablation_rel_l2.png",
        )
        _plot_window_line(
            rows,
            "t_max",
            "active_t_max",
            "speedup_median",
            "active_t_max at active_t_min=0.1",
            "Median speedup",
            out / "jit_tmax_ablation_speedup.png",
        )

    if args.probe_dir:
        summary = _read_json(args.probe_dir / "probe_summary.json")
        if summary:
            _plot_probe_error(summary, out / "jit_probe_local_vs_trajectory_error.png")
            _plot_probe_amplification(summary, out / "jit_probe_amplification_vs_local_error.png")

    if args.validation_dir:
        rows = _read_csv(args.validation_dir / "validation_results.csv")
        _plot_validation_speed_quality(rows, out / "jit_validation_speed_quality.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
