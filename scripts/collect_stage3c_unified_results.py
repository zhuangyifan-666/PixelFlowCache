#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

UNIFIED_FIELDNAMES = [
    "model",
    "prediction_type",
    "method_type",
    "method_name",
    "boundary_type",
    "reference_steps",
    "eval_steps",
    "num_samples",
    "seed_count",
    "speedup_mean",
    "speedup_std",
    "rel_l2_mean",
    "rel_l2_std",
    "psnr_mean",
    "psnr_std",
    "hit_rate_mean",
    "reduced_step_reference_match",
    "notes",
]

CACHE_VS_REDUCED_FIELDNAMES = [
    "model",
    "prediction_type",
    "source",
    "method_name",
    "boundary_type",
    "speedup_mean",
    "rel_l2_mean",
    "psnr_mean",
    "hit_rate_mean",
    "matched_reduced_method",
    "matched_reduced_speedup_mean",
    "matched_reduced_rel_l2_mean",
    "speedup_gap",
    "rel_l2_advantage",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _latest_dir(parent: Path, required_file: str) -> Path | None:
    if not parent.exists():
        return None
    candidates = [path for path in parent.iterdir() if path.is_dir() and (path / required_file).exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(row: dict[str, Any], key: str) -> float:
    value = _to_float(row.get(key))
    if value is None:
        return 0.0
    return value


def _source_from_notes(row: dict[str, Any]) -> str:
    for part in str(row.get("notes", "")).split(";"):
        stripped = part.strip()
        if stripped.startswith("source="):
            return stripped.split("=", 1)[1]
    return "unknown"


def _deco_boundary_type(method_type: str, method_name: str, row: dict[str, str]) -> str:
    if method_type != "cache":
        return "none"
    explicit = {
        "all_candidates": "all_candidates",
        "backbone_plus_final": "backbone_plus_final",
        "final_only": "output_final",
        "backbone_only": "backbone_only",
        "decoder_plus_final": "output_final",
        "decoder_only_no_final": "decoder",
        "backbone_plus_decoder_no_final": "backbone_plus_decoder",
    }
    if method_name in explicit:
        return explicit[method_name]
    if method_name.startswith("late_backbone_plus_final"):
        return "backbone_plus_final"
    if method_name.startswith("late_backbone"):
        return "backbone_only"
    has_final = str(row.get("has_final_cache", "")).lower() == "true"
    has_backbone = str(row.get("has_backbone_cache", "")).lower() == "true"
    has_decoder = str(row.get("has_decoder_cache", "")).lower() == "true"
    if has_final and has_backbone and has_decoder:
        return "all_candidates"
    if has_final and has_backbone:
        return "backbone_plus_final"
    if has_final:
        return "output_final"
    if has_backbone:
        return "backbone_only"
    if has_decoder:
        return "decoder"
    return "other"


def _notes(parts: list[str]) -> str:
    return "; ".join(part for part in parts if part)


def convert_jit_rows(rows: list[dict[str, str]], run_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        method_type = row.get("method_type", "")
        notes = _notes(
            [
                "source=jit_stage3a",
                f"run_dir={run_dir}",
                f"cache_layers={row.get('cache_layers', '')}",
                f"interval={row.get('cache_interval', '')}",
                f"active_t=[{row.get('active_t_min', '')},{row.get('active_t_max', '')})",
                f"warmup_refreshes={row.get('active_window_warmup_refreshes', '')}",
            ]
        )
        output.append(
            {
                "_source": "jit_stage3a",
                "model": "JiT",
                "prediction_type": "xpred",
                "method_type": method_type,
                "method_name": row.get("method_name", ""),
                "boundary_type": "backbone" if method_type == "cache" else "none",
                "reference_steps": _to_int(row.get("reference_steps")),
                "eval_steps": _to_int(row.get("eval_steps")),
                "num_samples": _to_int(row.get("num_samples")),
                "seed_count": _to_int(row.get("seed_count")),
                "speedup_mean": _to_float(row.get("speedup_mean")),
                "speedup_std": _to_float(row.get("speedup_std")),
                "rel_l2_mean": _to_float(row.get("rel_l2_mean")),
                "rel_l2_std": _to_float(row.get("rel_l2_std")),
                "psnr_mean": _to_float(row.get("psnr_mean")),
                "psnr_std": _to_float(row.get("psnr_std")),
                "hit_rate_mean": _to_float(row.get("hit_rate_mean")),
                "reduced_step_reference_match": "",
                "notes": notes,
            }
        )
    return output


def convert_deco_rows(rows: list[dict[str, str]], run_dir: Path, source: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        method_type = row.get("method_type", "")
        method_name = row.get("method_name", "")
        notes = _notes(
            [
                f"source={source}",
                f"run_dir={run_dir}",
                f"cache_units={row.get('cache_units', '')}",
                f"interval={row.get('cache_interval', '')}",
                f"active_t=[{row.get('active_t_min', '')},{row.get('active_t_max', '')})",
                f"has_final={row.get('has_final_cache', '')}",
                f"has_backbone={row.get('has_backbone_cache', '')}",
                f"has_decoder={row.get('has_decoder_cache', '')}",
            ]
        )
        output.append(
            {
                "_source": source,
                "model": "DeCo",
                "prediction_type": "vpred",
                "method_type": method_type,
                "method_name": method_name,
                "boundary_type": _deco_boundary_type(method_type, method_name, row),
                "reference_steps": _to_int(row.get("reference_steps")),
                "eval_steps": _to_int(row.get("eval_steps")),
                "num_samples": _to_int(row.get("num_samples")),
                "seed_count": _to_int(row.get("seed_count")),
                "speedup_mean": _to_float(row.get("speedup_mean")),
                "speedup_std": _to_float(row.get("speedup_std")),
                "rel_l2_mean": _to_float(row.get("rel_l2_mean")),
                "rel_l2_std": _to_float(row.get("rel_l2_std")),
                "psnr_mean": _to_float(row.get("psnr_mean")),
                "psnr_std": _to_float(row.get("psnr_std")),
                "hit_rate_mean": _to_float(row.get("hit_rate_mean")),
                "reduced_step_reference_match": "",
                "notes": notes,
            }
        )
    return output


def add_reduced_step_matches(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row.get("_source", ""))), []).append(row)
    for group in grouped.values():
        reduced = [row for row in group if row.get("method_type") == "reduced_steps"]
        if not reduced:
            continue
        for row in group:
            if row.get("method_type") != "cache" or _to_float(row.get("speedup_mean")) is None:
                continue
            closest = min(reduced, key=lambda item: abs(_float(item, "speedup_mean") - _float(row, "speedup_mean")))
            row["reduced_step_reference_match"] = str(closest["method_name"])


def make_cache_vs_reduced_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row.get("_source", ""))), []).append(row)
    output: list[dict[str, Any]] = []
    for (model, source), group in grouped.items():
        reduced = [row for row in group if row.get("method_type") == "reduced_steps"]
        if not reduced:
            continue
        for row in group:
            if row.get("method_type") != "cache":
                continue
            closest = min(reduced, key=lambda item: abs(_float(item, "speedup_mean") - _float(row, "speedup_mean")))
            speedup = _float(row, "speedup_mean")
            reduced_speedup = _float(closest, "speedup_mean")
            rel_l2 = _float(row, "rel_l2_mean")
            reduced_rel_l2 = _float(closest, "rel_l2_mean")
            output.append(
                {
                    "model": model,
                    "prediction_type": row.get("prediction_type"),
                    "source": source,
                    "method_name": row.get("method_name"),
                    "boundary_type": row.get("boundary_type"),
                    "speedup_mean": row.get("speedup_mean"),
                    "rel_l2_mean": row.get("rel_l2_mean"),
                    "psnr_mean": row.get("psnr_mean"),
                    "hit_rate_mean": row.get("hit_rate_mean"),
                    "matched_reduced_method": closest.get("method_name"),
                    "matched_reduced_speedup_mean": closest.get("speedup_mean"),
                    "matched_reduced_rel_l2_mean": closest.get("rel_l2_mean"),
                    "speedup_gap": speedup - reduced_speedup,
                    "rel_l2_advantage": reduced_rel_l2 - rel_l2,
                    "notes": row.get("notes"),
                }
            )
    return output


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in UNIFIED_FIELDNAMES} for row in rows]


def _find_row(rows: list[dict[str, Any]], model: str, method: str, source: str | None = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("model") == model and row.get("method_name") == method]
    if source is not None:
        candidates = [row for row in candidates if row.get("_source") == source]
    return candidates[0] if candidates else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def write_summary(path: Path, rows: list[dict[str, Any]], cache_vs_reduced: list[dict[str, Any]]) -> None:
    jit_quality = _find_row(rows, "JiT", "quality_t02_08")
    jit_speed = _find_row(rows, "JiT", "speed_t02_10")
    deco_best = _find_row(rows, "DeCo", "all_candidates", "deco_validation") or _find_row(rows, "DeCo", "all_candidates")
    deco_bpf = _find_row(rows, "DeCo", "backbone_plus_final", "deco_validation")
    deco_final = _find_row(rows, "DeCo", "final_only", "deco_validation")
    lines = [
        "# Stage 3C Unified BoundaryFlowCache Summary",
        "",
        "Stage 3C consolidates existing JiT Stage 3A and DeCo Stage 3B2 results. It does not add token cache, adaptive policy, calibration, or new model execution.",
        "",
        "## Best Current Rows",
        "",
    ]
    if jit_quality:
        lines.append(
            f"- JiT quality preset `quality_t02_08`: speedup {_fmt(jit_quality['speedup_mean'], 3)}x, rel-L2 {_fmt(jit_quality['rel_l2_mean'])}."
        )
    if jit_speed:
        lines.append(
            f"- JiT speed preset `speed_t02_10`: speedup {_fmt(jit_speed['speedup_mean'], 3)}x, rel-L2 {_fmt(jit_speed['rel_l2_mean'])}."
        )
    if deco_best:
        lines.append(
            f"- DeCo `all_candidates`: speedup {_fmt(deco_best['speedup_mean'], 3)}x, rel-L2 {_fmt(deco_best['rel_l2_mean'])}."
        )
    if deco_bpf:
        lines.append(
            f"- DeCo `backbone_plus_final`: speedup {_fmt(deco_bpf['speedup_mean'], 3)}x, rel-L2 {_fmt(deco_bpf['rel_l2_mean'])}."
        )
    if deco_final:
        lines.append(
            f"- DeCo `final_only`: speedup {_fmt(deco_final['speedup_mean'], 3)}x, rel-L2 {_fmt(deco_final['rel_l2_mean'])}."
        )
    lines.extend(
        [
            "",
            "## Cache Versus Reduced Steps",
            "",
            "Closest reduced-step matches are selected by absolute speedup difference within the same model and source run.",
        ]
    )
    for row in cache_vs_reduced:
        if row["method_name"] not in {"quality_t02_08", "speed_t02_10", "all_candidates", "backbone_plus_final", "final_only", "backbone_only"}:
            continue
        lines.append(
            "- {model} `{method}` vs `{reduced}`: cache rel-L2 {cache_l2}, reduced rel-L2 {reduced_l2}, rel-L2 advantage {advantage}.".format(
                model=row["model"],
                method=row["method_name"],
                reduced=row["matched_reduced_method"],
                cache_l2=_fmt(row["rel_l2_mean"]),
                reduced_l2=_fmt(row["matched_reduced_rel_l2_mean"]),
                advantage=_fmt(row["rel_l2_advantage"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_observations(path: Path) -> None:
    lines = [
        "# Stage 3C Boundary Observations",
        "",
        "1. Pixel-space flow cache is boundary-sensitive. Whole-block reuse acts through the point where stale features overwrite the fresh trajectory.",
        "2. JiT x-pred benefits from a whole-backbone boundary cache with early-step suppression; the best debug presets avoid the most unstable high-noise region.",
        "3. DeCo v-pred quality is controlled by the final/output velocity boundary. Backbone and decoder cache mainly add speed when paired with that output boundary.",
        "4. Cache preserves the full-step trajectory better than fewer-step no-cache sampling at similar speed in the current same-seed diagnostics.",
        "",
        "These observations support the working method name `BoundaryFlowCache`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_stage3c_results(
    *,
    jit_benchmark_dir: Path | None,
    deco_validation_dir: Path | None,
    deco_seed_dir: Path | None,
    deco_decomposition_dir: Path | None,
    output_dir: Path | None,
    auto_detect: bool = True,
) -> Path:
    jit_dir = jit_benchmark_dir
    validation_dir = deco_validation_dir
    seed_dir = deco_seed_dir
    decomposition_dir = deco_decomposition_dir
    if auto_detect:
        jit_dir = jit_dir or _latest_dir(ROOT / "logs/stage3a/jit_backbone_benchmark", "benchmark_aggregate.csv")
        validation_dir = validation_dir or _latest_dir(ROOT / "logs/stage3b2/deco_validate", "validation_aggregate.csv")
        seed_dir = seed_dir or _latest_dir(ROOT / "logs/stage3b2/deco_seed_sweep", "seed_sweep_aggregate.csv")
        decomposition_dir = decomposition_dir or _latest_dir(
            ROOT / "logs/stage3b2/deco_decomposition", "decomposition_aggregate.csv"
        )

    rows: list[dict[str, Any]] = []
    if jit_dir:
        rows.extend(convert_jit_rows(_read_csv(jit_dir / "benchmark_aggregate.csv"), jit_dir))
    else:
        print("Skipping JiT Stage 3A: no benchmark_aggregate.csv found")
    if validation_dir:
        rows.extend(convert_deco_rows(_read_csv(validation_dir / "validation_aggregate.csv"), validation_dir, "deco_validation"))
    else:
        print("Skipping DeCo validation: no validation_aggregate.csv found")
    if seed_dir:
        rows.extend(convert_deco_rows(_read_csv(seed_dir / "seed_sweep_aggregate.csv"), seed_dir, "deco_seed_sweep"))
    else:
        print("Skipping DeCo seed sweep: no seed_sweep_aggregate.csv found")
    if decomposition_dir:
        rows.extend(convert_deco_rows(_read_csv(decomposition_dir / "decomposition_aggregate.csv"), decomposition_dir, "deco_decomposition"))
    else:
        print("Skipping DeCo decomposition: no decomposition_aggregate.csv found")

    if not rows:
        raise FileNotFoundError("No Stage 3A/3B2 aggregate CSV files were found")

    add_reduced_step_matches(rows)
    cache_vs_reduced = make_cache_vs_reduced_rows(rows)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (output_dir or (ROOT / "logs/stage3c/unified" / run_id)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    visible_rows = public_rows(rows)
    _write_csv(out_dir / "unified_results.csv", visible_rows, UNIFIED_FIELDNAMES)
    _write_json(out_dir / "unified_results.json", visible_rows)
    _write_csv(out_dir / "unified_cache_vs_reduced.csv", cache_vs_reduced, CACHE_VS_REDUCED_FIELDNAMES)
    write_observations(out_dir / "unified_boundary_observations.md")
    write_summary(out_dir / "summary.md", rows, cache_vs_reduced)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jit-benchmark-dir", type=Path)
    parser.add_argument("--deco-validation-dir", type=Path)
    parser.add_argument("--deco-seed-dir", type=Path)
    parser.add_argument("--deco-decomposition-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out_dir = collect_stage3c_results(
        jit_benchmark_dir=args.jit_benchmark_dir.resolve() if args.jit_benchmark_dir else None,
        deco_validation_dir=args.deco_validation_dir.resolve() if args.deco_validation_dir else None,
        deco_seed_dir=args.deco_seed_dir.resolve() if args.deco_seed_dir else None,
        deco_decomposition_dir=args.deco_decomposition_dir.resolve() if args.deco_decomposition_dir else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        auto_detect=True,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
