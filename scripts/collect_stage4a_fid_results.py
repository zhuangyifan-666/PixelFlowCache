#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIELDNAMES = [
    "model",
    "method",
    "num_images",
    "steps",
    "latency_sec",
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


def _method_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*/*") if path.is_dir() and (path / "generation_meta.json").exists())


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


def collect_results(root: Path, fid_root: Path, result_jsons: list[Path]) -> list[dict[str, Any]]:
    fid_by_fake = _fid_results(fid_root, result_jsons)
    rows: list[dict[str, Any]] = []
    for method_dir in _method_dirs(root):
        meta = _load_json(method_dir / "generation_meta.json")
        latency = _load_json(method_dir / "latency.json")
        cache = _load_json(method_dir / "cache_stats.json")
        method = (meta.get("method") or {}).get("method_name") or method_dir.name
        model = meta.get("model") or method_dir.parents[1].name
        fake_dir = str((method_dir / "images").resolve())
        fid = fid_by_fake.get(fake_dir, {})
        rows.append(
            {
                "model": model,
                "method": method,
                "num_images": meta.get("num_images") or latency.get("generated_images"),
                "steps": (meta.get("method") or {}).get("eval_steps"),
                "latency_sec": _float(latency, "latency_sec"),
                "images_per_sec": _float(latency, "images_per_sec"),
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
    refs: dict[str, float] = {}
    for row in rows:
        if row["method"] == "no_cache_50" and row["latency_sec"]:
            refs[str(row["model"])] = float(row["latency_sec"])
    for row in rows:
        reference = refs.get(str(row["model"]))
        latency = row.get("latency_sec")
        row["speedup_vs_no_cache"] = reference / float(latency) if reference and latency else None


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
        "| model | method | images | steps | speedup | FID | IS | cache hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {method} | {num_images} | {steps} | {speedup_vs_no_cache} | {FID} | {IS} | {cache_hit_rate} |".format(
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
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out_dir or ROOT / "logs/stage4a/summary" / run_id).resolve()
    rows = collect_results(args.root.resolve(), args.fid_root.resolve(), [path.resolve() for path in args.result_json])
    _write_csv(out_dir / "stage4a_results.csv", rows)
    (out_dir / "stage4a_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary(out_dir / "summary.md", rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

