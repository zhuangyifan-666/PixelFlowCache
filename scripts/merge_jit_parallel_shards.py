#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.generation_io import count_images  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sum_by_module(shards: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for shard in shards:
        for module, stats in (shard.get("by_module") or {}).items():
            for key, value in stats.items():
                if isinstance(value, int):
                    merged[module][key] += value
    return {module: dict(values) for module, values in sorted(merged.items())}


def _merge_nested_counts(values: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for payload in values:
        for outer, inner in (payload or {}).items():
            if isinstance(inner, dict):
                for key, value in inner.items():
                    if isinstance(value, int):
                        merged[str(outer)][str(key)] += value
    return {key: dict(value) for key, value in sorted(merged.items())}


def _merge_safe_policy(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    policies = [shard.get("safe_policy") for shard in shards if isinstance(shard.get("safe_policy"), dict)]
    if not policies:
        return None
    merged_stats: dict[str, Any] = defaultdict(int)
    reuse_weighted_age = 0.0
    reuse_count = 0
    nested_keys = ["by_reason", "by_boundary", "by_age", "by_step", "by_branch", "by_solver_stage"]
    nested_values: dict[str, list[dict[str, Any]]] = {key: [] for key in nested_keys}
    for policy in policies:
        stats = policy.get("stats") or {}
        for key, value in stats.items():
            if key in nested_keys:
                nested_values[key].append(value)
            elif isinstance(value, int):
                merged_stats[key] += value
        count = int(stats.get("safe_reuse_committed", stats.get("safe_reuse", 0)) or 0)
        reuse_count += count
        reuse_weighted_age += float(stats.get("mean_age_of_reuse", stats.get("mean_age", 0.0)) or 0.0) * count
    mean_age = reuse_weighted_age / reuse_count if reuse_count else 0.0
    merged_stats["mean_age"] = mean_age
    merged_stats["mean_age_of_reuse"] = mean_age
    merged_stats["max_age"] = max((int((policy.get("stats") or {}).get("max_age", 0) or 0) for policy in policies), default=0)
    merged_stats["max_age_of_reuse"] = max(
        (int((policy.get("stats") or {}).get("max_age_of_reuse", 0) or 0) for policy in policies),
        default=0,
    )
    for key, payloads in nested_values.items():
        if key == "by_reason":
            flat: dict[str, int] = defaultdict(int)
            for payload in payloads:
                for reason, value in payload.items():
                    if isinstance(value, int):
                        flat[str(reason)] += value
            merged_stats[key] = dict(sorted(flat.items()))
        else:
            merged_stats[key] = _merge_nested_counts(payloads)
    return {"policy": policies[0].get("policy"), "config": policies[0].get("config"), "stats": dict(merged_stats)}


def _merge_taylorseer_policy(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    policies = [shard.get("taylorseer_policy") for shard in shards if isinstance(shard.get("taylorseer_policy"), dict)]
    if not policies:
        return None
    merged_stats: dict[str, Any] = defaultdict(int)
    nested_keys = ["by_module", "by_branch", "by_step", "by_order"]
    nested_values: dict[str, list[dict[str, Any]]] = {key: [] for key in nested_keys}
    weighted_order = 0.0
    committed = 0
    for policy in policies:
        stats = policy.get("stats") or {}
        for key, value in stats.items():
            if key in nested_keys:
                nested_values[key].append(value)
            elif isinstance(value, int):
                merged_stats[key] += value
        count = int(stats.get("forecast_committed", 0) or 0)
        committed += count
        weighted_order += float(stats.get("mean_effective_order", 0.0) or 0.0) * count
    merged_stats["mean_effective_order"] = weighted_order / committed if committed else 0.0
    for key, payloads in nested_values.items():
        if key == "by_order":
            flat: dict[str, int] = defaultdict(int)
            for payload in payloads:
                for order, value in (payload or {}).items():
                    if isinstance(value, int):
                        flat[str(order)] += value
            merged_stats[key] = dict(sorted(flat.items()))
        else:
            merged_stats[key] = _merge_nested_counts(payloads)
    return {"policy": policies[0].get("policy"), "config": policies[0].get("config"), "stats": dict(merged_stats)}


def _merge_cache_stats(shards: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["total_calls", "hits", "misses", "refreshes", "disabled"]
    merged = {key: sum(int(shard.get(key, 0) or 0) for shard in shards) for key in keys}
    merged["hit_rate"] = merged["hits"] / merged["total_calls"] if merged["total_calls"] else 0.0
    merged["model_name"] = next((shard.get("model_name") for shard in shards if shard.get("model_name")), "JiT")
    merged["enabled"] = any(bool(shard.get("enabled")) for shard in shards)
    merged["num_entries"] = sum(int(shard.get("num_entries", 0) or 0) for shard in shards)
    merged["by_module"] = _sum_by_module(shards)
    safe_policy = _merge_safe_policy(shards)
    if safe_policy is not None:
        merged["safe_policy"] = safe_policy
    taylorseer_policy = _merge_taylorseer_policy(shards)
    if taylorseer_policy is not None:
        merged["taylorseer_policy"] = taylorseer_policy
    dynamic = [shard.get("dynamic_cache") for shard in shards if isinstance(shard.get("dynamic_cache"), dict)]
    if dynamic:
        merged["dynamic_cache"] = dynamic[0]
    return merged


def _merge_latency(shards: list[dict[str, Any]], launcher_meta: dict[str, Any] | None, image_count: int) -> dict[str, Any]:
    worker_latencies = [float(shard.get("latency_sec", 0.0) or 0.0) for shard in shards]
    parallel_latency = max(worker_latencies) if worker_latencies else 0.0
    if launcher_meta and launcher_meta.get("wall_time_sec"):
        parallel_latency = float(launcher_meta["wall_time_sec"])
    generated_this_run = sum(int(shard.get("generated_images_this_run", shard.get("generated_images", 0)) or 0) for shard in shards)
    total_shard_images = sum(int(shard.get("total_shard_images", 0) or 0) for shard in shards)
    generated_images = generated_this_run or total_shard_images
    skipped = sum(int(shard.get("existing_images_skipped", 0) or 0) for shard in shards)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latency_sec": parallel_latency,
        "parallel_latency_sec": parallel_latency,
        "worker_latency_sec": worker_latencies,
        "generated_images": generated_images,
        "generated_images_this_run": generated_this_run,
        "existing_images_skipped": skipped,
        "total_images_available": image_count,
        "images_per_sec": generated_images / parallel_latency if parallel_latency > 0 else float("inf"),
        "num_shards": len(shards),
        "resume": any(bool(shard.get("resume")) for shard in shards),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge JiT sharded generation outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--launcher-meta", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifests = [args.run_dir / f"manifest_shard{idx}.jsonl" for idx in range(args.num_shards)]
    cache_paths = [args.run_dir / f"cache_stats_shard{idx}.json" for idx in range(args.num_shards)]
    latency_paths = [args.run_dir / f"latency_shard{idx}.json" for idx in range(args.num_shards)]
    meta_paths = [args.run_dir / f"generation_meta_shard{idx}.json" for idx in range(args.num_shards)]
    missing = [str(path) for path in [*manifests, *cache_paths, *latency_paths, *meta_paths] if not path.is_file()]
    rows = [row for path in manifests for row in _read_manifest(path)]
    rows.sort(key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in rows]
    image_count = count_images(args.run_dir / "images")
    expected = set(range(args.expected_images))
    present = set(indices)
    duplicates = sorted({idx for idx in indices if indices.count(idx) > 1})
    report = {
        "run_dir": str(args.run_dir),
        "method": args.method,
        "num_shards": args.num_shards,
        "expected_images": args.expected_images,
        "manifest_rows": len(rows),
        "image_count": image_count,
        "missing_files": missing,
        "missing_indices": sorted(expected - present)[:100],
        "missing_index_count": len(expected - present),
        "extra_indices": sorted(present - expected)[:100],
        "extra_index_count": len(present - expected),
        "duplicate_indices": duplicates[:100],
        "duplicate_index_count": len(duplicates),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.strict and (missing or image_count != args.expected_images or report["missing_index_count"] or report["extra_index_count"] or duplicates):
        raise RuntimeError(f"Shard merge strict checks failed: {report}")

    cache_shards = [_read_json(path) for path in cache_paths if path.is_file()]
    latency_shards = [_read_json(path) for path in latency_paths if path.is_file()]
    meta_shards = [_read_json(path) for path in meta_paths if path.is_file()]
    launcher_meta = _read_json(args.launcher_meta) if args.launcher_meta and args.launcher_meta.is_file() else None
    with (args.run_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(args.run_dir / "cache_stats.json", _merge_cache_stats(cache_shards))
    _write_json(args.run_dir / "latency.json", _merge_latency(latency_shards, launcher_meta, image_count))
    merged_meta = dict(meta_shards[0]) if meta_shards else {}
    merged_meta.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "num_shards": args.num_shards,
            "method_name": args.method,
            "shard_metas": meta_shards,
            "parallel_merge_report": report,
        }
    )
    _write_json(args.run_dir / "generation_meta.json", merged_meta)
    _write_json(args.run_dir / "parallel_merge_report.json", report)
    print(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
