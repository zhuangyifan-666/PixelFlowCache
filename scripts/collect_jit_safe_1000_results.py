#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = [
    "no_cache_50",
    "safe_bfc_quality",
    "safe_bfc_speed",
    "seacache_style",
    "reduced_steps_35",
    "reduced_steps_30",
]
FIELDS = [
    "method",
    "num_images",
    "steps",
    "latency_sec",
    "images_per_sec",
    "speedup_vs_no_cache",
    "FID",
    "IS",
    "PSNR",
    "SSIM",
    "LPIPS",
    "rel_l2",
    "cache_hit_rate",
    "cache_total_calls",
    "cache_hits",
    "cache_refreshes",
    "safe_reuse",
    "unsafe_refresh",
    "max_age",
    "mean_age",
    "safe_lambda",
    "safe_quantile",
]


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"invalid json: {path}: {exc}")
        return {}


def _get_any(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        node: Any = payload
        ok = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok:
            return node
    return ""


def _metric_value(payload: dict[str, Any], names: list[str]) -> Any:
    if not payload:
        return ""
    candidates: list[dict[str, Any]] = [payload]
    for key in ("metrics", "results", "summary"):
        if isinstance(payload.get(key), dict):
            candidates.append(payload[key])
    for candidate in candidates:
        for name in names:
            for key in (name, name.lower(), name.upper()):
                if key in candidate:
                    value = candidate[key]
                    if isinstance(value, dict):
                        for inner in ("mean", "value", "score"):
                            if inner in value:
                                return value[inner]
                    return value
    return ""


def _safe_stats(cache_stats: dict[str, Any]) -> dict[str, Any]:
    safe_policy = cache_stats.get("safe_policy") if isinstance(cache_stats, dict) else {}
    if not isinstance(safe_policy, dict):
        safe_policy = {}
    stats = safe_policy.get("stats") if isinstance(safe_policy.get("stats"), dict) else {}
    config = safe_policy.get("config") if isinstance(safe_policy.get("config"), dict) else {}
    return {
        "safe_reuse": stats.get("safe_reuse", ""),
        "unsafe_refresh": stats.get("unsafe_refresh", ""),
        "max_age": config.get("max_age", _get_any(cache_stats, ["safe_policy.stats.max_age"])),
        "mean_age": stats.get("mean_age", ""),
        "safe_lambda": config.get("safe_lambda", ""),
        "safe_quantile": config.get("quantile", ""),
    }


def _row_for_method(args: argparse.Namespace, method: str, warnings: list[str]) -> dict[str, Any]:
    run_dir = Path(args.output_root) / "jit" / args.run_id / method
    fid_dir = Path(args.fid_root) / args.run_id / "jit" / method
    pair_dir = Path(args.pair_root) / args.run_id / "jit" / method
    meta = _read_json(run_dir / "generation_meta.json", warnings)
    latency = _read_json(run_dir / "latency.json", warnings)
    cache_stats = _read_json(run_dir / "cache_stats.json", warnings)
    fid = _read_json(fid_dir / "fid_results.json", warnings)
    pair = {} if method == "no_cache_50" else _read_json(pair_dir / "pair_metrics.json", warnings)
    safe = _safe_stats(cache_stats)
    return {
        "method": method,
        "num_images": _get_any(meta, ["num_images"]) or _get_any(latency, ["generated_images"]),
        "steps": _get_any(meta, ["eval_steps", "steps"]),
        "latency_sec": _get_any(latency, ["latency_sec"]),
        "images_per_sec": _get_any(latency, ["images_per_sec"]),
        "speedup_vs_no_cache": "",
        "FID": _metric_value(fid, ["fid", "FID", "frechet_inception_distance"]),
        "IS": _metric_value(fid, ["is", "IS", "inception_score", "inception_score_mean"]),
        "PSNR": _metric_value(pair, ["psnr", "PSNR"]),
        "SSIM": _metric_value(pair, ["ssim", "SSIM"]),
        "LPIPS": _metric_value(pair, ["lpips", "LPIPS"]),
        "rel_l2": _metric_value(pair, ["rel_l2", "relative_l2"]),
        "cache_hit_rate": _get_any(cache_stats, ["hit_rate", "cache_hit_rate"]),
        "cache_total_calls": _get_any(cache_stats, ["total_calls", "cache_total_calls"]),
        "cache_hits": _get_any(cache_stats, ["hits", "cache_hits"]),
        "cache_refreshes": _get_any(cache_stats, ["refreshes", "cache_refreshes"]),
        **safe,
    }


def _as_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "warnings": warnings}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# JiT Safe-BFC 1000-Image Proxy Summary",
        "",
        "These are 1000-image proxy results and should not be interpreted as final 50k FID/IS.",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.append("| method | FID | IS | PSNR | SSIM | LPIPS | images/sec | speedup | cache hit | safe reuse |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {method} | {FID} | {IS} | {PSNR} | {SSIM} | {LPIPS} | {images_per_sec} | "
            "{speedup_vs_no_cache} | {cache_hit_rate} | {safe_reuse} |".format(**row)
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect JiT Safe-BFC 1000-image proxy results.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/stage4a/full_generation")
    parser.add_argument("--fid-root", default="logs/stage5a/fid")
    parser.add_argument("--pair-root", default="logs/stage5a/pair_metrics")
    parser.add_argument("--out-dir")
    parser.add_argument("--methods")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()] if args.methods else DEFAULT_METHODS
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs/stage5a/summary") / args.run_id
    warnings: list[str] = []
    rows = [_row_for_method(args, method, warnings) for method in methods]
    baseline_ips = _as_float(next((row["images_per_sec"] for row in rows if row["method"] == "no_cache_50"), ""))
    for row in rows:
        ips = _as_float(row.get("images_per_sec"))
        row["speedup_vs_no_cache"] = ips / baseline_ips if ips is not None and baseline_ips else ""
    _write_outputs(out_dir, rows, warnings)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
