#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


HIGHLIGHT_METHODS = {"final_only", "backbone_only", "backbone_plus_final", "all_candidates"}


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


def make_table_rows(aggregate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    reduced = [row for row in aggregate_rows if row.get("method_type") == "reduced_steps"]
    output = []
    for row in aggregate_rows:
        method = row["method_name"]
        if method not in HIGHLIGHT_METHODS and row.get("method_type") != "reduced_steps":
            continue
        notes = []
        if method in HIGHLIGHT_METHODS and reduced:
            closest = min(reduced, key=lambda other: abs(_float(other, "speedup_mean") - _float(row, "speedup_mean")))
            notes.append(
                "closest reduced-step: {name} speedup {speedup:.3f} rel-L2 {rel:.4f}".format(
                    name=closest["method_name"],
                    speedup=_float(closest, "speedup_mean"),
                    rel=_float(closest, "rel_l2_mean"),
                )
            )
        output.append(
            {
                "method": method,
                "type": row["method_type"],
                "speedup_mean": row["speedup_mean"],
                "rel_l2_mean": row["rel_l2_mean"],
                "psnr_mean": row["psnr_mean"],
                "hit_rate_mean": row["hit_rate_mean"],
                "has_final_cache": row["has_final_cache"],
                "has_backbone_cache": row["has_backbone_cache"],
                "has_decoder_cache": row["has_decoder_cache"],
                "notes": "; ".join(notes),
            }
        )
    return output


def write_outputs(run_dir: Path, aggregate_name: str, table_stem: str) -> tuple[Path, Path] | None:
    aggregate_path = run_dir / aggregate_name
    if not aggregate_path.exists():
        return None
    rows = make_table_rows(_read_csv(aggregate_path))
    csv_path = run_dir / f"{table_stem}.csv"
    md_path = run_dir / f"{table_stem}.md"
    fieldnames = [
        "method",
        "type",
        "speedup_mean",
        "rel_l2_mean",
        "psnr_mean",
        "hit_rate_mean",
        "has_final_cache",
        "has_backbone_cache",
        "has_decoder_cache",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {table_stem}",
        "",
        "| method | type | speedup | rel-L2 | PSNR | hit rate | final | backbone | decoder | notes |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {type} | {speedup_mean} | {rel_l2_mean} | {psnr_mean} | {hit_rate_mean} | {has_final_cache} | {has_backbone_cache} | {has_decoder_cache} | {notes} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    outputs = []
    decomposition = write_outputs(args.decomposition_dir.resolve(), "decomposition_aggregate.csv", "paper_table_deco_decomposition")
    if decomposition:
        outputs.extend(decomposition)
    if args.validation_dir:
        validation = write_outputs(args.validation_dir.resolve(), "validation_aggregate.csv", "paper_table_deco_validation")
        if validation:
            outputs.extend(validation)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
