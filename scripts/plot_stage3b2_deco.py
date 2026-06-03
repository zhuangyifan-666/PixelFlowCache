#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def _bool(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true", "yes"}


def _save_bar(rows: list[dict[str, Any]], key: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [str(row["method_name"]) for row in rows]
    values = [_float(row, key) for row in rows]
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_scatter(rows: list[dict[str, Any]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        ax.scatter(_float(row, "speedup_mean"), _float(row, "rel_l2_mean"))
        ax.annotate(str(row["method_name"]), (_float(row, "speedup_mean"), _float(row, "rel_l2_mean")), fontsize=8)
    ax.set_xlabel("Speedup vs reference")
    ax.set_ylabel("Same-seed rel-L2")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_final_effect(rows: list[dict[str, Any]], path: Path) -> None:
    cache_rows = [row for row in rows if row.get("method_type") == "cache"]
    with_final = [row for row in cache_rows if _bool(row, "has_final_cache")]
    without_final = [row for row in cache_rows if not _bool(row, "has_final_cache")]
    labels = ["with final", "without final"]
    values = [
        sum(_float(row, "rel_l2_mean") for row in with_final) / max(len(with_final), 1),
        sum(_float(row, "rel_l2_mean") for row in without_final) / max(len(without_final), 1),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Mean same-seed rel-L2")
    ax.set_title("DeCo Stage 3B2 final cache effect")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_cache_vs_reduced(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row.get("method_type") in {"cache", "reduced_steps"}]
    _save_scatter(selected, "DeCo Stage 3B2 cache vs reduced steps", path)


def plot_stage3b2(
    decomposition_dir: Path,
    validation_dir: Path | None,
    seed_sweep_dir: Path | None,
    output_dir: Path,
) -> list[Path]:
    decomposition = _read_csv(decomposition_dir / "decomposition_aggregate.csv")
    validation = _read_csv(validation_dir / "validation_aggregate.csv") if validation_dir else []
    output_paths = [
        output_dir / "deco_stage3b2_decomposition_speed_quality.png",
        output_dir / "deco_stage3b2_rel_l2_by_cache_unit.png",
        output_dir / "deco_stage3b2_speedup_by_cache_unit.png",
        output_dir / "deco_stage3b2_final_cache_effect.png",
        output_dir / "deco_stage3b2_cache_vs_reduced_steps.png",
        output_dir / "deco_stage3b2_validation_speed_quality.png",
    ]
    _save_scatter(decomposition, "DeCo Stage 3B2 decomposition speed-quality", output_paths[0])
    _save_bar(decomposition, "rel_l2_mean", "Same-seed rel-L2", "DeCo Stage 3B2 rel-L2 by cache unit", output_paths[1])
    _save_bar(decomposition, "speedup_mean", "Speedup vs reference", "DeCo Stage 3B2 speedup by cache unit", output_paths[2])
    _save_final_effect(decomposition, output_paths[3])
    _save_cache_vs_reduced(decomposition, output_paths[4])
    if validation:
        _save_scatter(validation, "DeCo Stage 3B2 validation speed-quality", output_paths[5])
    else:
        _save_scatter([], "DeCo Stage 3B2 validation pending", output_paths[5])
    if seed_sweep_dir:
        _read_csv(seed_sweep_dir / "seed_sweep_aggregate.csv")
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--seed-sweep-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/stage3b2/figures")
    args = parser.parse_args()
    paths = plot_stage3b2(
        args.decomposition_dir.resolve(),
        args.validation_dir.resolve() if args.validation_dir else None,
        args.seed_sweep_dir.resolve() if args.seed_sweep_dir else None,
        args.output_dir.resolve(),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
