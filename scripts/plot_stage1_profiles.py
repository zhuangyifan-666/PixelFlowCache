#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfc")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pfc-cache")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _model_name(run_dir: Path) -> str:
    parts = {part.lower() for part in run_dir.parts}
    if "deco" in parts:
        return "deco"
    return "jit"


def _ensure_summary(run_dir: Path) -> None:
    needed = ["feature_delta_by_module.csv", "velocity_by_step.csv", "frequency_by_step.csv"]
    if not all((run_dir / name).exists() for name in needed):
        import subprocess
        import sys

        subprocess.run([sys.executable, "scripts/summarize_stage1_profiles.py", "--run-dir", str(run_dir)], check=True)


def _plot_feature(run_dir: Path, figure_dir: Path, model: str) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = _read_csv(run_dir / "feature_delta_by_module.csv")
    if not rows:
        return []
    rows = rows[:40]
    labels = [row["module_name"] for row in rows]
    values = [float(row["mean_rel_l2_delta"]) for row in rows]
    plt.figure(figsize=(max(8, len(labels) * 0.35), 5))
    plt.imshow([values], aspect="auto")
    plt.yticks([0], ["mean rel L2"])
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.tight_layout()
    name = f"{model}_block_temporal_delta_heatmap.png" if model == "jit" else f"{model}_module_temporal_delta_heatmap.png"
    path = figure_dir / name
    plt.savefig(path, dpi=160)
    plt.close()
    return [path]


def _plot_velocity(run_dir: Path, figure_dir: Path, model: str) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = _read_csv(run_dir / "velocity_by_step.csv")
    if not rows:
        return []
    cfg_rows = [row for row in rows if row.get("branch") == "cfg"] or rows
    steps = [int(row["step_idx"]) for row in cfg_rows]
    values = [float(row["mean_v_l2"]) for row in cfg_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, marker="o")
    plt.xlabel("step")
    plt.ylabel("velocity L2")
    plt.tight_layout()
    path = figure_dir / f"{model}_velocity_norm_by_step.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths = [path]

    if model == "jit" and any(row.get("mean_amplification") for row in cfg_rows):
        amp_steps = [int(row["step_idx"]) for row in cfg_rows if row.get("mean_amplification")]
        amps = [float(row["mean_amplification"]) for row in cfg_rows if row.get("mean_amplification")]
        plt.figure(figsize=(7, 4))
        plt.plot(amp_steps, amps, marker="o")
        plt.xlabel("step")
        plt.ylabel("1 / max(1 - t, eps)")
        plt.tight_layout()
        amp_path = figure_dir / "jit_xpred_amplification_by_step.png"
        plt.savefig(amp_path, dpi=160)
        plt.close()
        paths.append(amp_path)
    return paths


def _plot_frequency(run_dir: Path, figure_dir: Path, model: str) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = _read_csv(run_dir / "frequency_by_step.csv")
    if not rows:
        return []
    steps = [int(row["step_idx"]) for row in rows]
    plt.figure(figsize=(7, 4))
    for key in ("low_ratio", "mid_ratio", "high_ratio"):
        values = [float(row[key]) for row in rows if row.get(key)]
        if values:
            plt.plot(steps[: len(values)], values, marker="o", label=key)
    plt.xlabel("step")
    plt.ylabel("energy ratio")
    plt.legend()
    plt.tight_layout()
    path = figure_dir / f"{model}_frequency_ratio_by_step.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return [path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/stage1/figures"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    _ensure_summary(run_dir)
    model = _model_name(run_dir)
    paths: list[Path] = []
    paths.extend(_plot_feature(run_dir, figure_dir, model))
    paths.extend(_plot_velocity(run_dir, figure_dir, model))
    paths.extend(_plot_frequency(run_dir, figure_dir, model))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
