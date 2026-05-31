from __future__ import annotations

import json
import math

from scripts.run_jit_stage2c_probe import pearson_correlation, summarize_probe_records


def test_pearson_correlation_handles_constant_inputs() -> None:
    assert pearson_correlation([1.0, 1.0], [0.1, 0.2]) is None
    assert pearson_correlation([1.0], [0.1]) is None


def test_probe_summary_tracks_local_and_trajectory_error() -> None:
    records = [
        {
            "step_idx": 0,
            "trajectory_error": {"rel_l2": 0.10},
            "probe_error": {"rel_l2": 0.05},
            "amplification": 1.0,
        },
        {
            "step_idx": 0,
            "trajectory_error": {"rel_l2": 0.20},
            "probe_error": {"rel_l2": 0.10},
            "amplification": 1.0,
        },
        {
            "step_idx": 1,
            "trajectory_error": {"rel_l2": 0.40},
            "probe_error": {"rel_l2": 0.30},
            "amplification": 2.0,
        },
    ]
    summary = summarize_probe_records(records)
    assert summary["record_count"] == 3
    assert summary["probe_record_count"] == 3
    assert math.isclose(summary["step_means"][0]["trajectory_rel_l2_mean"], 0.15)
    assert math.isclose(summary["step_means"][0]["probe_rel_l2_mean"], 0.075)
    assert summary["max_probe_rel_l2"] == 0.30
    assert summary["correlation_amplification_probe_rel_l2"] is not None
    assert summary["dominance"] == "accumulated_trajectory_drift_dominates"
    json.dumps(summary)


def test_probe_summary_handles_missing_probe_records() -> None:
    summary = summarize_probe_records([{"step_idx": 0, "trajectory_error": {"rel_l2": 0.1}, "amplification": 1.0}])
    assert summary["probe_record_count"] == 0
    assert summary["dominance"] == "insufficient_probe_data"
    json.dumps(summary)
