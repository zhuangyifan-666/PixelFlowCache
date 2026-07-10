#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.risk.io import load_jsonl, write_json_atomic, write_jsonl, write_text_atomic  # noqa: E402
from pfc.risk.schema import PIXARC_STAGE1_SCHEMA_VERSION, ensure_strict_json_value  # noqa: E402


RISK_METRICS = (
    "risk_scaled_rms",
    "risk_rel_l2",
    "risk_low",
    "risk_high",
    "velocity_rel_l2",
    "cond_xpred_rel_l2",
    "uncond_xpred_rel_l2",
)
STAT_FIELDS = ("count", "mean", "std", "min", "p50", "p90", "p95", "max")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge JiT PixARC Stage-1 sample outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--strict", action="store_true")
    return parser


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile for an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary_stats(values: Iterable[float]) -> dict[str, float | int]:
    finite_values = [float(value) for value in values]
    if not finite_values:
        raise ValueError("cannot summarize an empty sequence")
    ensure_strict_json_value(finite_values)
    return {
        "count": len(finite_values),
        "mean": statistics.fmean(finite_values),
        "std": statistics.pstdev(finite_values),
        "min": min(finite_values),
        "p50": percentile(finite_values, 0.50),
        "p90": percentile(finite_values, 0.90),
        "p95": percentile(finite_values, 0.95),
        "max": max(finite_values),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    ensure_strict_json_value(payload, str(path))
    return payload


def collect_stage1_records(
    run_dir: Path | str,
    expected_images: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(run_dir)
    config = _read_json(root / "run_config.json")
    configured_images = int(config.get("num_images", -1))
    expected_count = configured_images if expected_images is None else int(expected_images)
    if expected_count < 0:
        raise ValueError("expected image count is unavailable or invalid")
    if configured_images >= 0 and expected_count != configured_images:
        raise ValueError(
            f"expected-images={expected_count} disagrees with run config num_images={configured_images}"
        )
    sample_dirs: dict[int, Path] = {}
    samples_root = root / "samples"
    for sample_dir in sorted(samples_root.glob("sample_*")) if samples_root.is_dir() else ():
        suffix = sample_dir.name.removeprefix("sample_")
        if not suffix.isdigit():
            continue
        global_index = int(suffix)
        if global_index in sample_dirs:
            raise ValueError(f"duplicate Stage-1 sample index: {global_index}")
        sample_dirs[global_index] = sample_dir
    expected = set(range(expected_count))
    actual = set(sample_dirs)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Stage-1 sample set mismatch: missing={missing}, extra={extra}")

    risk_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    expected_signature = config.get("config_signature")
    for global_index in range(expected_count):
        sample_dir = sample_dirs[global_index]
        done = _read_json(sample_dir / "DONE.json")
        if int(done.get("schema_version", -1)) != PIXARC_STAGE1_SCHEMA_VERSION:
            raise ValueError(f"DONE schema mismatch for sample {global_index}")
        if int(done.get("global_index", -1)) != global_index:
            raise ValueError(f"DONE global index mismatch for sample {global_index}")
        if done.get("config_signature") != expected_signature:
            raise ValueError(f"DONE config signature mismatch for sample {global_index}")
        sample_summary = _read_json(sample_dir / "sample_summary.json")
        if int(sample_summary.get("global_index", -1)) != global_index:
            raise ValueError(f"sample summary index mismatch for sample {global_index}")
        sample_summaries.append(sample_summary)
        sample_risk = load_jsonl(sample_dir / "risk_records.jsonl")
        sample_correctness = load_jsonl(sample_dir / "correctness_records.jsonl")
        if len(sample_risk) != int(done.get("risk_record_count", -1)):
            raise ValueError(f"risk record count mismatch for sample {global_index}")
        if len(sample_correctness) != int(done.get("correctness_record_count", -1)):
            raise ValueError(f"correctness record count mismatch for sample {global_index}")
        for row in (*sample_risk, *sample_correctness):
            if int(row.get("global_index", -1)) != global_index:
                raise ValueError(f"record global index mismatch for sample {global_index}")
        risk_records.extend(sample_risk)
        correctness_records.extend(sample_correctness)

    risk_records.sort(
        key=lambda row: (
            int(row["global_index"]),
            int(row["step_idx"]),
            str(row.get("boundary_plan") or ""),
            str(row.get("action") or ""),
        )
    )
    correctness_records.sort(
        key=lambda row: (
            int(row["global_index"]),
            int(row["step_idx"]),
            str(row.get("boundary_plan") or ""),
            str(row.get("check_name") or ""),
            str(row.get("branch") or ""),
        )
    )
    return config, risk_records, correctness_records, sample_summaries


def _csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_latency_rows(risk_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in risk_records:
        latency = row.get("diagnostic_action_latency_ms")
        if latency is not None:
            grouped[(str(row["boundary_plan"]), str(row["action"]))].append(float(latency))
    return [
        {"boundary_plan": plan, "action": action, **summary_stats(values)}
        for (plan, action), values in sorted(grouped.items())
    ]


def build_risk_summary_rows(risk_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in risk_records:
        if not bool(row.get("action_ready")):
            continue
        for metric in RISK_METRICS:
            value = row.get(metric)
            if value is None:
                continue
            plan = str(row["boundary_plan"])
            action = str(row["action"])
            step = str(int(row["step_idx"]))
            grouped[("overall", metric, "*", "*", "*")].append(float(value))
            grouped[("plan_action", metric, plan, action, "*")].append(float(value))
            grouped[("plan_action_step", metric, plan, action, step)].append(float(value))
    return [
        {
            "scope": scope,
            "metric": metric,
            "boundary_plan": plan,
            "action": action,
            "step_idx": step,
            **summary_stats(values),
        }
        for (scope, metric, plan, action, step), values in sorted(grouped.items())
    ]


def merge_stage1(
    run_dir: Path | str,
    *,
    expected_images: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    config, risk_records, correctness_records, sample_summaries = collect_stage1_records(
        root, expected_images
    )
    merged = root / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    write_jsonl(merged / "risk_records.jsonl", risk_records)
    write_jsonl(merged / "correctness_records.jsonl", correctness_records)

    latency_rows = build_latency_rows(risk_records)
    risk_summary_rows = build_risk_summary_rows(risk_records)
    write_text_atomic(
        merged / "action_latency.csv",
        _csv_text(latency_rows, ["boundary_plan", "action", *STAT_FIELDS]),
    )
    write_text_atomic(
        merged / "risk_summary.csv",
        _csv_text(
            risk_summary_rows,
            ["scope", "metric", "boundary_plan", "action", "step_idx", *STAT_FIELDS],
        ),
    )
    correctness_summary = {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "record_count": len(correctness_records),
        "allclose_failure_count": sum(not bool(row.get("allclose")) for row in correctness_records),
        "shape_failure_count": sum(not bool(row.get("shape_match")) for row in correctness_records),
        "dtype_failure_count": sum(not bool(row.get("dtype_match")) for row in correctness_records),
        "future_leakage_count": sum(int(row.get("future_leakage_count", 0)) for row in correctness_records),
    }
    write_json_atomic(merged / "correctness_summary.json", correctness_summary)
    summary = {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "run_id": config.get("run_id"),
        "sample_count": len(sample_summaries),
        "risk_record_count": len(risk_records),
        "correctness_record_count": len(correctness_records),
        "latency_group_count": len(latency_rows),
        "risk_summary_row_count": len(risk_summary_rows),
        "parallel_wall_time_is_algorithm_speedup": False,
    }
    write_json_atomic(merged / "merge_summary.json", summary)
    markdown = "\n".join(
        [
            "# JiT PixARC Stage-1 Summary",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Completed samples: {summary['sample_count']}",
            f"- Risk records: {summary['risk_record_count']}",
            f"- Correctness records: {summary['correctness_record_count']}",
            f"- Correctness allclose failures: {correctness_summary['allclose_failure_count']}",
            "- Candidate actions are evaluated from fresh states; this is not sequential rollout.",
            "- Action latency is isolated diagnostic timing, not end-to-end algorithm speedup.",
            "- Four-GPU wall time only reflects independent sample sharding.",
            "",
        ]
    )
    write_text_atomic(merged / "stage1_summary.md", markdown)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = merge_stage1(args.run_dir, expected_images=args.expected_images)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
