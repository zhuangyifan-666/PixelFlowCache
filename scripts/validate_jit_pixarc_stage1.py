#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from merge_jit_pixarc_stage1 import collect_stage1_records  # noqa: E402
from pfc.risk.io import write_json_atomic  # noqa: E402
from pfc.risk.schema import PIXARC_STAGE1_SCHEMA_VERSION  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate JiT PixARC Stage-1 outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--age0-max-abs", type=float, default=1e-5)
    parser.add_argument("--age0-rel-l2", type=float, default=1e-6)
    parser.add_argument("--strict", action="store_true")
    return parser


def _issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, **details: Any) -> None:
    issues.append({"severity": severity, "code": code, "message": message, "details": details})


def _find_nonfinite(value: Any, path: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_find_nonfinite(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_find_nonfinite(item, f"{path}[{index}]"))
    return paths


def _plan_names(config: dict[str, Any]) -> list[str]:
    plans = config.get("plans", [])
    names = [str(plan["name"]) for plan in plans if isinstance(plan, dict) and "name" in plan]
    if not names:
        raise ValueError("run config does not contain resolved boundary plans")
    return names


def validate_stage1(
    run_dir: Path | str,
    *,
    expected_images: int | None = None,
    age0_max_abs: float = 1e-5,
    age0_rel_l2: float = 1e-6,
    write_report: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir)
    issues: list[dict[str, Any]] = []
    try:
        config, risk_records, correctness_records, sample_summaries = collect_stage1_records(
            root, expected_images
        )
    except Exception as exc:
        _issue(issues, "BLOCK", "record_collection", str(exc))
        report = {
            "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
            "status": "BLOCK",
            "issues": issues,
            "checks": {},
        }
        if write_report and root.exists():
            write_json_atomic(root / "merged" / "validation_report.json", report)
        return report

    steps = int(config["steps"])
    num_images = int(config["num_images"] if expected_images is None else expected_images)
    plans = _plan_names(config)
    actions = [str(action) for action in config.get("actions", [])]
    expected_samples = set(range(num_images))

    nonfinite_paths: list[str] = []
    for collection_name, records in (
        ("risk", risk_records),
        ("correctness", correctness_records),
        ("sample_summary", sample_summaries),
    ):
        for index, record in enumerate(records):
            nonfinite_paths.extend(
                _find_nonfinite(record, f"{collection_name}[{index}]")
            )
    if nonfinite_paths:
        _issue(
            issues,
            "BLOCK",
            "nonfinite",
            "Stage-1 output contains non-finite values",
            paths=nonfinite_paths[:20],
            total=len(nonfinite_paths),
        )

    fresh_checks = [
        row for row in correctness_records if row.get("check_name") == "fresh_split_equivalence"
    ]
    failed_fresh = [row for row in fresh_checks if not bool(row.get("allclose"))]
    equivalence_steps = [int(step) for step in config.get("correctness", {}).get("equivalence_steps", [])]
    fresh_check_keys = Counter(
        (int(row["global_index"]), int(row["step_idx"]), str(row["branch"]))
        for row in fresh_checks
    )
    expected_fresh_check_keys = {
        (sample, step, branch)
        for sample in expected_samples
        for step in equivalence_steps
        for branch in ("cond", "uncond")
    }
    missing_fresh_checks = sorted(expected_fresh_check_keys - set(fresh_check_keys))
    duplicate_fresh_checks = sorted(
        key for key, count in fresh_check_keys.items() if count != 1
    )
    if missing_fresh_checks or duplicate_fresh_checks:
        _issue(
            issues,
            "BLOCK",
            "fresh_equivalence_completeness",
            "fresh split-forward equivalence records are incomplete or duplicated",
            missing=missing_fresh_checks[:20],
            duplicate=duplicate_fresh_checks[:20],
        )
    if failed_fresh:
        _issue(
            issues,
            "BLOCK",
            "fresh_equivalence",
            "fresh split-forward equivalence failed",
            failures=len(failed_fresh),
        )

    replay_checks = [row for row in correctness_records if row.get("check_name") == "replay_age_0"]
    replay_keys = Counter(
        (int(row["global_index"]), int(row["step_idx"]), str(row["boundary_plan"]))
        for row in replay_checks
    )
    expected_replay_keys = {
        (sample, step, plan)
        for sample in expected_samples
        for step in range(steps)
        for plan in plans
    }
    missing_replay = sorted(expected_replay_keys - set(replay_keys))
    duplicate_replay = sorted(key for key, count in replay_keys.items() if count != 1)
    if missing_replay or duplicate_replay:
        _issue(
            issues,
            "BLOCK",
            "replay_completeness",
            "replay age-0 correctness records are incomplete or duplicated",
            missing=missing_replay[:20],
            duplicate=duplicate_replay[:20],
        )
    failed_age0 = [
        row
        for row in replay_checks
        if not bool(row.get("allclose"))
        or float(row.get("max_abs", math.inf)) > age0_max_abs
        or float(row.get("relative_l2", math.inf)) > age0_rel_l2
    ]
    if failed_age0:
        _issue(
            issues,
            "BLOCK",
            "replay_age0",
            "replay age-0 exceeds correctness thresholds",
            failures=len(failed_age0),
            max_abs_threshold=age0_max_abs,
            relative_l2_threshold=age0_rel_l2,
        )

    future_leakage = sum(
        int(row.get("future_leakage_count", 0)) for row in correctness_records
    ) + sum(int(row.get("future_leakage_count", 0)) for row in sample_summaries)
    if future_leakage:
        _issue(
            issues,
            "BLOCK",
            "future_leakage",
            "history records indicate future-step leakage",
            count=future_leakage,
        )

    fresh_risk = [row for row in risk_records if row.get("action") == "fresh"]
    fresh_keys = Counter((int(row["global_index"]), int(row["step_idx"])) for row in fresh_risk)
    expected_fresh_keys = {(sample, step) for sample in expected_samples for step in range(steps)}
    missing_fresh = sorted(expected_fresh_keys - set(fresh_keys))
    duplicate_fresh = sorted(key for key, count in fresh_keys.items() if count != 1)
    nonzero_fresh = [
        row
        for row in fresh_risk
        if any(float(row.get(metric, 0.0) or 0.0) != 0.0 for metric in ("risk_scaled_rms", "risk_rel_l2"))
    ]
    if missing_fresh or duplicate_fresh or nonzero_fresh:
        _issue(
            issues,
            "BLOCK",
            "fresh_record_completeness",
            "fresh risk records must occur once per sample/step with zero risk",
            missing=missing_fresh[:20],
            duplicate=duplicate_fresh[:20],
            nonzero=len(nonzero_fresh),
        )

    risk_by_key = Counter(
        (
            int(row["global_index"]),
            int(row["step_idx"]),
            str(row["boundary_plan"]),
            str(row["action"]),
        )
        for row in risk_records
        if row.get("action") != "fresh"
    )
    for action in [action for action in actions if action != "fresh"]:
        expected_action_keys = {
            (sample, step, plan, action)
            for sample in expected_samples
            for step in range(steps)
            for plan in plans
        }
        actual_action_keys = {key for key in risk_by_key if key[3] == action}
        missing = sorted(expected_action_keys - actual_action_keys)
        duplicate = sorted(key for key in expected_action_keys if risk_by_key[key] != 1)
        if missing or duplicate:
            _issue(
                issues,
                "BLOCK",
                "action_completeness",
                f"risk matrix is incomplete for {action}",
                action=action,
                missing=missing[:20],
                duplicate=duplicate[:20],
            )

    availability_failures: list[dict[str, Any]] = []
    unavailable_until = {"reuse_age_1": 1, "reuse_age_2": 2, "taylor_order_1": 2}
    for row in risk_records:
        action = str(row.get("action"))
        if action not in unavailable_until:
            continue
        should_be_ready = int(row["step_idx"]) >= unavailable_until[action]
        ready = bool(row.get("action_ready"))
        valid_reason = row.get("skip_reason") == "insufficient_history"
        if ready != should_be_ready or (not ready and not valid_reason):
            availability_failures.append(row)
    if availability_failures:
        _issue(
            issues,
            "BLOCK",
            "action_availability",
            "history-dependent action readiness is incorrect",
            failures=len(availability_failures),
        )

    candidate_shape_failures = [
        row
        for row in risk_records
        if bool(row.get("action_ready"))
        and (
            row.get("signals", {}).get("candidate_shape_match") is not True
            or row.get("signals", {}).get("candidate_dtype_match") is not True
        )
    ]
    correctness_shape_failures = [
        row
        for row in correctness_records
        if not bool(row.get("shape_match")) or not bool(row.get("dtype_match"))
    ]
    if candidate_shape_failures or correctness_shape_failures:
        _issue(
            issues,
            "BLOCK",
            "shape_dtype",
            "candidate or correctness output shape/dtype differs from fresh output",
            candidate_failures=len(candidate_shape_failures),
            correctness_failures=len(correctness_shape_failures),
        )

    reuse_rows = [
        row
        for row in risk_records
        if row.get("action") in {"reuse_age_1", "reuse_age_2"} and bool(row.get("action_ready"))
    ]
    if any(action in actions for action in ("reuse_age_1", "reuse_age_2")):
        if not reuse_rows or not any(float(row.get("risk_scaled_rms") or 0.0) > 0.0 for row in reuse_rows):
            _issue(
                issues,
                "BLOCK",
                "candidate_sanity",
                "all ready reuse candidates are identical to fresh",
            )

    uncleared_history = [
        row for row in sample_summaries if int(row.get("history_items_after_clear", -1)) != 0
    ]
    if uncleared_history:
        _issue(
            issues,
            "BLOCK",
            "history_cleanup",
            "sample history was not empty after instrumentation",
            failures=len(uncleared_history),
        )

    status = "BLOCK" if any(issue["severity"] == "BLOCK" for issue in issues) else (
        "WARN" if issues else "PASS"
    )
    report = {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "status": status,
        "run_id": config.get("run_id"),
        "thresholds": {
            "age0_max_abs": age0_max_abs,
            "age0_relative_l2": age0_rel_l2,
        },
        "checks": {
            "sample_count": len(sample_summaries),
            "risk_record_count": len(risk_records),
            "correctness_record_count": len(correctness_records),
            "fresh_equivalence_check_count": len(fresh_checks),
            "replay_age0_check_count": len(replay_checks),
            "future_leakage_count": future_leakage,
            "positive_reuse_risk_found": any(
                float(row.get("risk_scaled_rms") or 0.0) > 0.0 for row in reuse_rows
            ),
            "nonfinite_count": len(nonfinite_paths),
        },
        "issues": issues,
    }
    if write_report:
        write_json_atomic(root / "merged" / "validation_report.json", report)
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = validate_stage1(
        args.run_dir,
        expected_images=args.expected_images,
        age0_max_abs=args.age0_max_abs,
        age0_rel_l2=args.age0_rel_l2,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if args.strict and report["status"] == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
