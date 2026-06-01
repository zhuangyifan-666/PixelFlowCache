#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PAPER_TABLE_FIELDNAMES = [
    "method",
    "type",
    "speedup_mean",
    "rel_l2_mean",
    "rel_l2_std",
    "psnr_mean",
    "hit_rate",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_TABLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in ("", None, "None"):
        return float("nan")
    return float(value)


def _closest_reduced_step_note(row: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if row.get("method_type") != "cache":
        return ""
    reduced = [item for item in rows if item.get("method_type") == "reduced_steps"]
    if not reduced:
        return ""
    speedup = _float(row, "speedup_mean")
    closest = min(reduced, key=lambda item: abs(_float(item, "speedup_mean") - speedup))
    return (
        f"closest reduced-step speed: {closest['method_name']} "
        f"(speedup {float(closest['speedup_mean']):.3f}, rel-L2 {float(closest['rel_l2_mean']):.4f})"
    )


def make_paper_table_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes_by_method = {
        "quality_t02_08": "Quality-first BackboneCache.",
        "speed_t02_10": "Speed-quality BackboneCache.",
    }
    rows = []
    for row in aggregate_rows:
        method = str(row["method_name"])
        note_parts = []
        if method in notes_by_method:
            note_parts.append(notes_by_method[method])
        closest_note = _closest_reduced_step_note(row, aggregate_rows)
        if closest_note:
            note_parts.append(closest_note)
        rows.append(
            {
                "method": method,
                "type": row["method_type"],
                "speedup_mean": row["speedup_mean"],
                "rel_l2_mean": row["rel_l2_mean"],
                "rel_l2_std": row["rel_l2_std"],
                "psnr_mean": row["psnr_mean"],
                "hit_rate": row["hit_rate_mean"],
                "notes": " ".join(note_parts),
            }
        )
    return rows


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 3A JiT BackboneCache Report Table",
        "",
        "| method | type | speedup mean | rel-L2 mean | rel-L2 std | PSNR mean | hit rate | notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {type} | {speedup_mean} | {rel_l2_mean} | {rel_l2_std} | "
            "{psnr_mean} | {hit_rate} | {notes} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate_rows = _read_csv(args.benchmark_dir / "benchmark_aggregate.csv")
    rows = make_paper_table_rows(aggregate_rows)
    _write_csv(args.benchmark_dir / "paper_table.csv", rows)
    _write_markdown(args.benchmark_dir / "paper_table.md", rows)
    print(args.benchmark_dir / "paper_table.md")
    print(args.benchmark_dir / "paper_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
