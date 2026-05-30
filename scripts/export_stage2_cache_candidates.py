#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.profiling.module_selectors import categorize_deco_module


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _feature_rows_from_jsonl(run_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in _read_jsonl(run_dir / "feature_stats.jsonl"):
        delta = record.get("temporal_delta") or {}
        value = delta.get("rel_l2_delta")
        if isinstance(value, (int, float)):
            grouped[str(record.get("module_name", "unknown"))].append(float(value))
    rows = []
    for module_name, values in grouped.items():
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        median = values_sorted[mid] if len(values_sorted) % 2 else (values_sorted[mid - 1] + values_sorted[mid]) / 2
        rows.append(
            {
                "module_name": module_name,
                "count": len(values),
                "mean_rel_l2_delta": sum(values) / len(values),
                "median_rel_l2_delta": median,
            }
        )
    return rows


def _load_feature_rows(run_dir: Path) -> list[dict[str, Any]]:
    csv_rows = _read_csv(run_dir / "feature_delta_by_module.csv")
    if csv_rows:
        return csv_rows
    return _feature_rows_from_jsonl(run_dir)


def _is_jit(run_dir: Path) -> bool:
    if "jit" in {part.lower() for part in run_dir.parts}:
        return True
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(meta.get("model_name", "")).lower() == "jit"
    return False


def _is_jit_block(module_name: str) -> bool:
    return bool(re.search(r"(?:^|\.?)blocks\.\d+$", module_name))


def _is_deco_cacheable_level(module_name: str, category: str) -> bool:
    lower = module_name.lower()
    if any(token in lower for token in ("norm", "modulation", "adaln", "q_norm", "k_norm", ".attn", ".mlp", ".linear")):
        return False
    if category == "block":
        return bool(re.fullmatch(r"(?:cond_)?blocks\.\d+", module_name))
    if category == "decoder":
        return bool(re.fullmatch(r"dec_net(?:\.res_blocks\.\d+)?", module_name))
    if category == "final":
        return True
    return False


def _policy(mean_delta: float) -> str:
    if mean_delta <= 0.10:
        return "aggressive_fixed_interval"
    if mean_delta <= 0.25:
        return "conservative_fixed_interval"
    return "do_not_cache"


def export_candidates(run_dir: Path) -> Path:
    rows = _load_feature_rows(run_dir)
    jit = _is_jit(run_dir)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        module_name = str(row["module_name"])
        mean_delta = float(row["mean_rel_l2_delta"])
        median_delta = float(row["median_rel_l2_delta"])
        record_count = int(float(row.get("count", row.get("record_count", 0))))
        if jit:
            if not _is_jit_block(module_name):
                continue
            category = "block"
        else:
            category = categorize_deco_module(module_name, object())  # type: ignore[arg-type]
            if not _is_deco_cacheable_level(module_name, category):
                continue
        output_rows.append(
            {
                "module_name": module_name,
                "module_category": category,
                "mean_rel_l2_delta": mean_delta,
                "median_rel_l2_delta": median_delta,
                "record_count": record_count,
                "recommended_stage2_policy": _policy(mean_delta),
            }
        )

    output_rows.sort(key=lambda item: item["mean_rel_l2_delta"])
    out_path = run_dir / "stage2_cache_candidates.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "module_name",
            "module_category",
            "mean_rel_l2_delta",
            "median_rel_l2_delta",
            "record_count",
            "recommended_stage2_policy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    out_path = export_candidates(args.run_dir.resolve())
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
