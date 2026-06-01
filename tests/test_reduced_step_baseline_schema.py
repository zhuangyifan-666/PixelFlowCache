from __future__ import annotations

import json

from scripts.run_jit_stage3a_backbone_benchmark import compute_speedup
from scripts.run_jit_stage3a_reduced_steps import REDUCED_STEP_FIELDNAMES, summarize_reduced_rows


def test_reduced_step_schema_contains_expected_columns() -> None:
    assert {
        "method",
        "reference_steps",
        "eval_steps",
        "num_samples",
        "seed",
        "latency_median_sec",
        "reference_latency_median_sec",
        "speedup_vs_reference",
        "same_seed_rel_l2",
        "same_seed_mse",
        "same_seed_psnr",
        "low_freq_delta_ratio",
        "mid_freq_delta_ratio",
        "high_freq_delta_ratio",
    }.issubset(REDUCED_STEP_FIELDNAMES)


def test_compute_speedup_uses_reference_latency_over_method_latency() -> None:
    assert compute_speedup(10.0, 5.0) == 2.0


def test_reduced_step_summary_is_json_serializable() -> None:
    rows = [
        {
            "method": "reduced_steps",
            "reference_steps": 50,
            "eval_steps": 35,
            "num_samples": 4,
            "seed": 0,
            "latency_median_sec": 5.0,
            "reference_latency_median_sec": 10.0,
            "speedup_vs_reference": 2.0,
            "same_seed_rel_l2": 0.08,
            "same_seed_mse": 0.01,
            "same_seed_psnr": 30.0,
            "low_freq_delta_ratio": 0.1,
            "mid_freq_delta_ratio": 0.2,
            "high_freq_delta_ratio": 0.7,
        }
    ]
    summary = summarize_reduced_rows(rows)
    assert summary["best_quality_row"]["eval_steps"] == 35
    json.dumps(summary)
