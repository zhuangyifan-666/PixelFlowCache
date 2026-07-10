import csv

import pytest

from pfc.risk.io import load_jsonl
from pixarc_stage1_test_utils import make_stage1_run
from scripts.merge_jit_pixarc_stage1 import merge_stage1


def test_merge_sorts_records_and_writes_aggregates(tmp_path):
    run_dir = make_stage1_run(tmp_path / "run")
    summary = merge_stage1(run_dir, expected_images=2)
    assert summary["sample_count"] == 2
    merged = run_dir / "merged"
    risk = load_jsonl(merged / "risk_records.jsonl")
    keys = [
        (row["global_index"], row["step_idx"], row["boundary_plan"], row["action"])
        for row in risk
    ]
    assert keys == sorted(keys)
    with (merged / "action_latency.csv").open(encoding="utf-8", newline="") as handle:
        latency = list(csv.DictReader(handle))
    assert latency
    assert {"count", "mean", "std", "p50", "p90", "p95", "max"} <= set(latency[0])
    with (merged / "risk_summary.csv").open(encoding="utf-8", newline="") as handle:
        risk_summary = list(csv.DictReader(handle))
    assert {row["scope"] for row in risk_summary} == {
        "overall",
        "plan_action",
        "plan_action_step",
    }
    assert "not end-to-end algorithm speedup" in (merged / "stage1_summary.md").read_text(
        encoding="utf-8"
    )


def test_merge_rejects_missing_sample(tmp_path):
    run_dir = make_stage1_run(tmp_path / "run", num_images=2, sample_indices=[0])
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        merge_stage1(run_dir, expected_images=2)
