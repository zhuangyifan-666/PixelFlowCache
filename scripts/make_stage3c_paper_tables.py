#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TABLE_FIELDNAMES = [
    "model",
    "method",
    "type",
    "speedup",
    "rel-L2",
    "PSNR",
    "hit rate",
    "prediction type",
    "note",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = TABLE_FIELDNAMES) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, title: str, rows: list[dict[str, Any]], fieldnames: list[str] = TABLE_FIELDNAMES) -> None:
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "|" + "|".join("---" for _ in fieldnames) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in ("", None, "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None, "None"):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _source(row: dict[str, Any]) -> str:
    for part in str(row.get("notes", "")).split(";"):
        stripped = part.strip()
        if stripped.startswith("source="):
            return stripped.split("=", 1)[1]
    return ""


def closest_reduced_step(row: dict[str, str], rows: list[dict[str, str]]) -> dict[str, str] | None:
    source = _source(row)
    reduced = [
        other
        for other in rows
        if other.get("model") == row.get("model")
        and other.get("method_type") == "reduced_steps"
        and (not source or _source(other) == source)
    ]
    if not reduced:
        reduced = [other for other in rows if other.get("model") == row.get("model") and other.get("method_type") == "reduced_steps"]
    if not reduced:
        return None
    speedup = _float(row, "speedup_mean")
    return min(reduced, key=lambda other: abs(_float(other, "speedup_mean") - speedup))


def _first_by_method(
    rows: list[dict[str, str]],
    *,
    model: str,
    method_name: str,
    preferred_source: str | None = None,
) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get("model") == model and row.get("method_name") == method_name]
    if preferred_source:
        preferred = [row for row in candidates if _source(row) == preferred_source]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def _table_row(row: dict[str, str], note: str = "") -> dict[str, str]:
    return {
        "model": row.get("model", ""),
        "method": row.get("method_name", ""),
        "type": row.get("method_type", ""),
        "speedup": _fmt(row.get("speedup_mean"), 3),
        "rel-L2": _fmt(row.get("rel_l2_mean"), 4),
        "PSNR": _fmt(row.get("psnr_mean"), 2),
        "hit rate": _fmt(row.get("hit_rate_mean"), 3),
        "prediction type": row.get("prediction_type", ""),
        "note": note,
    }


def make_main_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    selected_specs = [
        ("JiT", "quality_t02_08", None, "BackboneCache quality preset."),
        ("JiT", "speed_t02_10", None, "BackboneCache speed preset."),
        ("DeCo", "all_candidates", "deco_validation", "Output plus backbone/decoder boundary cache."),
        ("DeCo", "backbone_plus_final", "deco_validation", "Backbone plus final/output boundary cache."),
        ("DeCo", "final_only", "deco_validation", "Final/output boundary only."),
        ("DeCo", "backbone_only", "deco_validation", "Backbone boundary without final/output cache."),
    ]
    seen_reduced: set[tuple[str, str, str]] = set()
    for model, method, source, note in selected_specs:
        row = _first_by_method(rows, model=model, method_name=method, preferred_source=source)
        if not row:
            continue
        output.append(_table_row(row, note))
        if method in {"quality_t02_08", "speed_t02_10", "all_candidates"}:
            reduced = closest_reduced_step(row, rows)
            if reduced:
                key = (reduced.get("model", ""), reduced.get("method_name", ""), _source(reduced))
                if key not in seen_reduced:
                    output.append(_table_row(reduced, f"Closest reduced-step match for `{method}`."))
                    seen_reduced.add(key)
    return output


def make_boundary_ablation_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred_methods = {
        "all_candidates",
        "backbone_plus_final",
        "final_only",
        "backbone_only",
        "decoder_plus_final",
        "decoder_only_no_final",
        "backbone_plus_decoder_no_final",
        "late_backbone_only_6",
        "late_backbone_plus_final_6",
    }
    output: list[dict[str, str]] = []
    for row in rows:
        if row.get("model") != "DeCo" or row.get("method_type") != "cache":
            continue
        if row.get("method_name") not in preferred_methods:
            continue
        source = _source(row)
        if source not in {"deco_validation", "deco_decomposition"}:
            continue
        note = f"source={source}; boundary={row.get('boundary_type', '')}"
        output.append(_table_row(row, note))
    output.sort(key=lambda row: (row["note"], row["method"]))
    return output


def make_seed_stability_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    keep_methods = {
        "quality_t02_08",
        "speed_t02_10",
        "all_candidates",
        "backbone_plus_final",
        "final_only",
        "backbone_only",
    }
    for row in rows:
        if row.get("method_name") not in keep_methods:
            continue
        try:
            seed_count = int(float(row.get("seed_count", "0")))
        except ValueError:
            seed_count = 0
        if seed_count <= 1:
            continue
        note = f"seed_count={seed_count}; source={_source(row)}; speedup_std={_fmt(row.get('speedup_std'), 4)}; rel_l2_std={_fmt(row.get('rel_l2_std'), 4)}"
        output.append(_table_row(row, note))
    output.sort(key=lambda row: (row["model"], row["method"]))
    return output


def write_stage3c_tables(unified_dir: Path) -> list[Path]:
    rows = _read_csv(unified_dir / "unified_results.csv")
    outputs: list[Path] = []
    tables = [
        ("paper_table_main_cache_vs_reduced", "Stage 3C Main Cache Vs Reduced-Step Table", make_main_table(rows)),
        ("paper_table_boundary_ablation", "Stage 3C Boundary Ablation Table", make_boundary_ablation_table(rows)),
        ("paper_table_seed_stability", "Stage 3C Seed Stability Table", make_seed_stability_table(rows)),
    ]
    for stem, title, table_rows in tables:
        csv_path = unified_dir / f"{stem}.csv"
        md_path = unified_dir / f"{stem}.md"
        _write_csv(csv_path, table_rows)
        _write_markdown(md_path, title, table_rows)
        outputs.extend([md_path, csv_path])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in write_stage3c_tables(args.unified_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
