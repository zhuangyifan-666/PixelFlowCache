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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in ("", None, "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source(row: dict[str, Any]) -> str:
    for part in str(row.get("notes", "")).split(";"):
        stripped = part.strip()
        if stripped.startswith("source="):
            return stripped.split("=", 1)[1]
    return ""


def _label(row: dict[str, Any]) -> str:
    model = row.get("model", "")
    method = row.get("method_name", "")
    source = _source(row)
    if source and source not in {"jit_stage3a", "deco_validation"}:
        return f"{model}:{method}:{source.replace('deco_', '')}"
    return f"{model}:{method}"


def _save_speed_quality(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row.get("method_type") != "reference"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in selected:
        ax.scatter(_float(row, "speedup_mean"), _float(row, "rel_l2_mean"))
        ax.annotate(_label(row), (_float(row, "speedup_mean"), _float(row, "rel_l2_mean")), fontsize=7)
    ax.set_xlabel("Speedup vs same-source no-cache reference")
    ax.set_ylabel("Same-seed rel-L2 mean")
    ax.set_title("Stage 3C JiT and DeCo speed-quality")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _main_cache_vs_reduced_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = {
        "quality_t02_08",
        "speed_t02_10",
        "all_candidates",
        "backbone_plus_final",
        "final_only",
        "backbone_only",
        "reduced_steps_30",
        "reduced_steps_35",
    }
    output = [row for row in rows if row.get("method_name") in keep]
    return output


def _save_bar(rows: list[dict[str, str]], key: str, ylabel: str, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = _main_cache_vs_reduced_rows(rows)
    labels = [_label(row) for row in selected]
    values = [_float(row, key) for row in selected]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(len(labels)))
    ax.bar(xs, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_deco_boundary(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row.get("model") == "DeCo" and row.get("method_type") == "cache" and _source(row) in {"deco_validation", "deco_decomposition"}
    ]
    labels = [_label(row) for row in selected]
    values = [_float(row, "rel_l2_mean") for row in selected]
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = list(range(len(labels)))
    ax.bar(xs, values)
    ax.set_ylabel("Same-seed rel-L2 mean")
    ax.set_title("Stage 3C DeCo boundary ablation")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _best_row(rows: list[dict[str, str]], model: str, method: str, preferred_source: str | None = None) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get("model") == model and row.get("method_name") == method]
    if preferred_source:
        preferred = [row for row in candidates if _source(row) == preferred_source]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def _save_best_methods(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in [
            _best_row(rows, "JiT", "quality_t02_08"),
            _best_row(rows, "JiT", "speed_t02_10"),
            _best_row(rows, "DeCo", "all_candidates", "deco_validation"),
            _best_row(rows, "DeCo", "backbone_plus_final", "deco_validation"),
        ]
        if row is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in selected:
        ax.scatter(_float(row, "speedup_mean"), _float(row, "rel_l2_mean"))
        ax.annotate(_label(row), (_float(row, "speedup_mean"), _float(row, "rel_l2_mean")), fontsize=8)
    ax.set_xlabel("Speedup vs same-source no-cache reference")
    ax.set_ylabel("Same-seed rel-L2 mean")
    ax.set_title("Stage 3C best JiT and DeCo methods")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stage3c(unified_dir: Path, output_dir: Path | None = None) -> list[Path]:
    rows = _read_csv(unified_dir / "unified_results.csv")
    out = output_dir or (ROOT / "outputs/stage3c/figures")
    paths = [
        out / "stage3c_speed_quality_jit_deco.png",
        out / "stage3c_rel_l2_cache_vs_reduced.png",
        out / "stage3c_speedup_cache_vs_reduced.png",
        out / "stage3c_boundary_ablation_deco.png",
        out / "stage3c_jit_vs_deco_best_methods.png",
    ]
    _save_speed_quality(rows, paths[0])
    _save_bar(rows, "rel_l2_mean", "Same-seed rel-L2 mean", "Stage 3C cache vs reduced rel-L2", paths[1])
    _save_bar(rows, "speedup_mean", "Speedup vs reference mean", "Stage 3C cache vs reduced speedup", paths[2])
    _save_deco_boundary(rows, paths[3])
    _save_best_methods(rows, paths[4])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in plot_stage3c(args.unified_dir.resolve(), args.output_dir.resolve() if args.output_dir else None):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
