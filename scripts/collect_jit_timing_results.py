#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import list_methods_for_model  # noqa: E402
from pfc.eval.timing import normalize_timing_payload  # noqa: E402


DEFAULT_METHODS = list_methods_for_model(
    "jit", tags={"reference", "main_baseline", "proxy_default"}
)
FIELDS = [
    "method",
    "mean_sampling_latency",
    "std_sampling_latency",
    "median_sampling_latency",
    "mean_images_per_sec",
    "speedup_vs_no_cache",
    "peak_memory_mean",
    "peak_memory_max",
    "repeat_count",
    "comparable",
    "comparison_signature",
    "signature_diff",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _checkpoint_identity(meta: dict[str, Any]) -> tuple[str | None, dict[str, Any] | str | None]:
    provenance = meta.get("provenance") or {}
    checkpoint = provenance.get("checkpoint") or {}
    path = _first_not_none(
        meta.get("checkpoint_path"),
        meta.get("jit_ckpt_dir"),
        checkpoint.get("path") if isinstance(checkpoint, dict) else None,
    )
    explicit = meta.get("checkpoint_identity")
    if explicit is not None:
        return str(path) if path is not None else None, explicit
    if isinstance(checkpoint, dict) and checkpoint:
        identity = {
            "path": checkpoint.get("path") or path,
            "sha256": checkpoint.get("sha256"),
            "size": checkpoint.get("size"),
        }
        return str(path) if path is not None else None, identity
    return str(path) if path is not None else None, str(path) if path is not None else None


def build_comparison_signature(
    meta: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    provenance = meta.get("provenance") or {}
    gpu_names = provenance.get("gpu_names") or []
    checkpoint_path, checkpoint_identity = _checkpoint_identity(meta)
    device = str(meta.get("device", ""))
    cfg_interval = _first_not_none(meta.get("cfg_interval"), meta.get("guidance_interval"))
    if cfg_interval is None and (
        meta.get("cfg_interval_min") is not None or meta.get("cfg_interval_max") is not None
    ):
        cfg_interval = [meta.get("cfg_interval_min"), meta.get("cfg_interval_max")]
    return {
        "model_name": _first_not_none(meta.get("model_name"), meta.get("model")),
        "checkpoint_path": checkpoint_path,
        "checkpoint_identity": checkpoint_identity,
        "eval_steps": meta.get("eval_steps"),
        "reference_steps": meta.get("reference_steps"),
        "batch_size": meta.get("batch_size"),
        "num_images": _first_not_none(meta.get("num_images"), timing.get("requested_images")),
        "dtype": meta.get("dtype"),
        "amp_enabled": _first_not_none(meta.get("amp_enabled"), meta.get("amp")),
        "autocast_enabled": _first_not_none(meta.get("autocast_enabled"), meta.get("autocast")),
        "compile_enabled": meta.get("compile_enabled"),
        "device_type": _first_not_none(meta.get("device_type"), device.split(":", 1)[0] or None),
        "gpu_name": _first_not_none(meta.get("gpu_name"), gpu_names[0] if gpu_names else None),
        "gpu_count": provenance.get("gpu_count"),
        "timing_scope": timing.get("timing_scope"),
        "warmup_batches": meta.get("warmup_batches"),
        "seed": meta.get("seed"),
        "cfg_scale": _first_not_none(meta.get("cfg_scale"), meta.get("cfg")),
        "cfg_interval": cfg_interval,
        "sampler": _first_not_none(meta.get("sampler"), meta.get("sampling_method")),
        "solver": _first_not_none(meta.get("solver"), meta.get("solver_stage")),
        "image_size": _first_not_none(meta.get("image_size"), meta.get("img_size")),
        "num_shards": _first_not_none(timing.get("num_shards"), meta.get("num_shards")),
        "resume": _first_not_none(timing.get("resume"), meta.get("resume")),
        "save_png": meta.get("save_png"),
        "save_npz": meta.get("save_npz"),
    }


def comparison_signature_diff(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key) != candidate.get(key)
    }


def _collect_method(args: argparse.Namespace, method: str, warnings: list[str]) -> dict[str, Any]:
    latencies: list[float] = []
    throughputs: list[float] = []
    memories: list[int] = []
    comparison_signature: dict[str, Any] | None = None
    signature_diffs: list[dict[str, Any]] = []
    comparable = True
    for repeat in range(1, args.repeats + 1):
        run_id = f"{args.run_id}_r{repeat:02d}"
        run_dir = args.output_root / "jit" / run_id / method
        latency_path = run_dir / "latency.json"
        meta_path = run_dir / "generation_meta.json"
        if not latency_path.is_file() or not meta_path.is_file():
            warnings.append(f"{method} repeat {repeat}: missing timing artifacts under {run_dir}")
            comparable = False
            continue
        timing = normalize_timing_payload(_load(latency_path))
        meta = _load(meta_path)
        signature = build_comparison_signature(meta, timing)
        if comparison_signature is None:
            comparison_signature = signature
        else:
            diff = comparison_signature_diff(comparison_signature, signature)
            if diff:
                signature_diffs.append({"repeat": repeat, "diff": diff})
                warnings.append(f"{method} repeat {repeat}: comparison signature differs: {diff}")
                comparable = False
                continue
        if timing.get("legacy_timing") or not timing.get("comparable_for_algorithm_speedup"):
            warnings.append(f"{method} repeat {repeat}: timing is not schema-v2 comparable")
            comparable = False
            continue
        sampling = timing.get("sampling_latency_sec")
        throughput = timing.get("sampling_images_per_sec")
        if not isinstance(sampling, (int, float)) or sampling <= 0:
            warnings.append(f"{method} repeat {repeat}: invalid sampling latency")
            comparable = False
            continue
        latencies.append(float(sampling))
        if isinstance(throughput, (int, float)):
            throughputs.append(float(throughput))
        memories.append(int(timing.get("peak_memory_allocated_bytes", 0) or 0))
    comparable = comparable and bool(latencies) and not signature_diffs
    return {
        "method": method,
        "mean_sampling_latency": statistics.fmean(latencies) if latencies else None,
        "std_sampling_latency": statistics.pstdev(latencies) if latencies else None,
        "median_sampling_latency": statistics.median(latencies) if latencies else None,
        "mean_images_per_sec": statistics.fmean(throughputs) if throughputs else None,
        "speedup_vs_no_cache": None,
        "peak_memory_mean": statistics.fmean(memories) if memories else None,
        "peak_memory_max": max(memories) if memories else None,
        "repeat_count": len(latencies),
        "comparable": comparable,
        "comparison_signature": comparison_signature,
        "signature_diff": signature_diffs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect schema-v2 JiT single-GPU timing repeats.")
    parser.add_argument("--run-id", default="stage5a_jit_single_gpu_timing_seed0")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/timing"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/timing"))
    parser.add_argument("--methods")
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()] if args.methods else DEFAULT_METHODS
    warnings: list[str] = []
    rows = [_collect_method(args, method, warnings) for method in methods]
    baseline = next((row for row in rows if row["method"] == "no_cache_50"), None)
    if baseline and baseline["mean_sampling_latency"] and baseline["comparable"]:
        for row in rows:
            if row["method"] == "no_cache_50":
                row["speedup_vs_no_cache"] = 1.0
                continue
            diff = comparison_signature_diff(
                baseline["comparison_signature"] or {},
                row["comparison_signature"] or {},
            )
            if diff:
                row["comparable"] = False
                row["signature_diff"].append({"comparison": "vs_no_cache", "diff": diff})
                warnings.append(f"{row['method']}: speedup unavailable; signature diff vs no_cache_50: {diff}")
            elif row["comparable"] and row["mean_sampling_latency"]:
                row["speedup_vs_no_cache"] = baseline["mean_sampling_latency"] / row["mean_sampling_latency"]
            else:
                warnings.append(f"{row['method']}: speedup unavailable because timing data is not comparable")
    elif any(row["method"] != "no_cache_50" for row in rows):
        warnings.append("speedup unavailable because a comparable no_cache_50 baseline is missing")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [{key: row[key] for key in FIELDS} for row in rows]
    with (args.out_dir / "jit_timing_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(public_rows)
    (args.out_dir / "jit_timing_summary.json").write_text(
        json.dumps({"rows": public_rows, "warnings": warnings}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
