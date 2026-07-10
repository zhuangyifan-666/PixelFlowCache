#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.timing import normalize_timing_payload  # noqa: E402

FIELDNAMES = [
    "model",
    "run_id",
    "method",
    "num_images",
    "reference_key",
    "steps",
    "latency_sec",
    "sampling_latency_sec",
    "end_to_end_latency_sec",
    "timing_scope",
    "comparable_for_algorithm_speedup",
    "legacy_timing",
    "batch_size",
    "gpu_count",
    "resume",
    "images_per_sec",
    "speedup_vs_no_cache",
    "cache_hit_rate",
    "FID",
    "IS",
    "KID",
    "backend",
    "fake_dir",
    "result_json",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_model_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    models = {item.strip().lower() for item in value.split(",") if item.strip()}
    return models or None


def _parse_run_id_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    run_ids = {item.strip() for item in value.split(",") if item.strip()}
    return run_ids or None


def _method_dirs(root: Path, run_id: str | None = None, model: str | None = None) -> list[Path]:
    model_filter = _parse_model_filter(model)
    run_id_filter = _parse_run_id_filter(run_id)
    paths = []
    for path in sorted(root.glob("*/*/*")):
        if not path.is_dir() or not (path / "generation_meta.json").exists():
            continue
        path_model = path.parents[1].name
        path_run_id = path.parent.name
        if run_id_filter and path_run_id not in run_id_filter:
            continue
        if model_filter and path_model.lower() not in model_filter:
            continue
        paths.append(path)
    return paths


def _fid_results(fid_root: Path, result_jsons: list[Path]) -> dict[str, dict[str, Any]]:
    paths = list(result_jsons)
    if fid_root.exists():
        paths.extend(fid_root.rglob("fid_results.json"))
    mapping: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        fake_dir = payload.get("fake_dir")
        if fake_dir:
            payload["_result_json"] = str(path)
            mapping[str(Path(fake_dir).resolve())] = payload
    return mapping


def _reference_key(model: Any, run_id: Any, num_images: Any) -> str:
    return f"{model}:{run_id}:n{num_images}"


def collect_results(
    root: Path,
    fid_root: Path,
    result_jsons: list[Path],
    *,
    run_id: str | None = None,
    num_images: int | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    fid_by_fake = _fid_results(fid_root, result_jsons)
    rows: list[dict[str, Any]] = []
    for method_dir in _method_dirs(root, run_id=run_id, model=model):
        meta = _load_json(method_dir / "generation_meta.json")
        latency = _load_json(method_dir / "latency.json")
        timing = normalize_timing_payload(latency)
        cache = _load_json(method_dir / "cache_stats.json")
        method = (meta.get("method") or {}).get("method_name") or method_dir.name
        model = meta.get("model") or method_dir.parents[1].name
        method_run_id = meta.get("run_id") or method_dir.parent.name
        row_num_images = _int_or_none(meta.get("num_images") or latency.get("generated_images"))
        if num_images is not None and row_num_images != num_images:
            continue
        fake_dir = str((method_dir / "images").resolve())
        fid = fid_by_fake.get(fake_dir, {})
        reference_key = _reference_key(model, method_run_id, row_num_images)
        rows.append(
            {
                "model": model,
                "run_id": method_run_id,
                "method": method,
                "num_images": row_num_images,
                "reference_key": reference_key,
                "steps": (meta.get("method") or {}).get("eval_steps"),
                "latency_sec": _float(timing, "end_to_end_latency_sec", "latency_sec"),
                "sampling_latency_sec": _float(timing, "sampling_latency_sec"),
                "end_to_end_latency_sec": _float(timing, "end_to_end_latency_sec"),
                "timing_scope": timing.get("timing_scope"),
                "comparable_for_algorithm_speedup": bool(timing.get("comparable_for_algorithm_speedup", False)),
                "legacy_timing": bool(timing.get("legacy_timing", False)),
                "batch_size": meta.get("batch_size"),
                "gpu_count": (meta.get("provenance") or {}).get("gpu_count", timing.get("num_shards")),
                "resume": bool(timing.get("resume", meta.get("resume", False))),
                "images_per_sec": _float(timing, "sampling_images_per_sec"),
                "speedup_vs_no_cache": None,
                "cache_hit_rate": _float(cache, "hit_rate"),
                "FID": _float(fid, "FID", "frechet_inception_distance"),
                "IS": _float(fid, "IS", "inception_score_mean"),
                "KID": _float(fid, "KID", "kernel_inception_distance_mean"),
                "backend": fid.get("backend"),
                "fake_dir": fake_dir,
                "result_json": fid.get("_result_json"),
            }
        )
    _add_speedups(rows)
    return rows


def _add_speedups(rows: list[dict[str, Any]]) -> None:
    refs: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["method"] == "no_cache_50" and _row_is_comparable(row):
            refs[str(row["reference_key"])] = row
    for row in rows:
        reference = refs.get(str(row["reference_key"]))
        latency = row.get("sampling_latency_sec")
        same_signature = bool(reference and _comparison_signature(reference) == _comparison_signature(row))
        row["speedup_vs_no_cache"] = (
            float(reference["sampling_latency_sec"]) / float(latency)
            if same_signature and _row_is_comparable(row) and latency
            else None
        )


def _row_is_comparable(row: dict[str, Any]) -> bool:
    return bool(
        row.get("comparable_for_algorithm_speedup")
        and not row.get("legacy_timing")
        and not row.get("resume")
        and row.get("sampling_latency_sec")
    )


def _comparison_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("gpu_count"), row.get("batch_size"), row.get("num_images"), row.get("timing_scope")
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDNAMES} for row in rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 4A FID Summary",
        "",
        "| model | run_id | method | images | steps | speedup | FID | IS | cache hit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {run_id} | {method} | {num_images} | {steps} | {speedup_vs_no_cache} | {FID} | {IS} | {cache_hit_rate} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 4A generation metadata and FID results.")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/stage4a/full_generation")
    parser.add_argument("--fid-root", type=Path, default=ROOT / "logs/stage4a/fid")
    parser.add_argument("--result-json", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out_dir or ROOT / "logs/stage4a/summary" / run_id).resolve()
    rows = collect_results(
        args.root.resolve(),
        args.fid_root.resolve(),
        [path.resolve() for path in args.result_json],
        run_id=args.run_id,
        num_images=args.num_images,
        model=args.model,
    )
    _write_csv(out_dir / "stage4a_results.csv", rows)
    (out_dir / "stage4a_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_summary(out_dir / "summary.md", rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
