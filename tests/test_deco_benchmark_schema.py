from __future__ import annotations

import json
import math

from scripts.run_deco_stage3b_benchmark import aggregate_benchmark_rows


def _row(method_type: str, method_name: str, seed: int, speedup: float, rel_l2: float) -> dict:
    return {
        "method_type": method_type,
        "method_name": method_name,
        "seed": seed,
        "num_samples": 8,
        "reference_steps": 20,
        "eval_steps": 20 if method_type != "reduced_steps" else 15,
        "cache_units": "backbone_blocks" if method_type == "cache" else "none",
        "cache_interval": 2 if method_type == "cache" else 1,
        "active_t_min": 0.2 if method_type == "cache" else None,
        "active_t_max": 1.0 if method_type == "cache" else None,
        "latency_median_sec": 1.0,
        "speedup_vs_reference": speedup,
        "cache_hit_rate": 0.4 if method_type == "cache" else 0.0,
        "same_seed_rel_l2": rel_l2,
        "same_seed_mse": rel_l2 * rel_l2,
        "same_seed_mae": rel_l2,
        "same_seed_psnr": 40.0,
        "low_freq_delta_ratio": 0.1,
        "mid_freq_delta_ratio": 0.2,
        "high_freq_delta_ratio": 0.7,
        "wrapped_module_count": 28 if method_type == "cache" else 0,
        "run_dir": "logs/test",
    }


def test_stage3b_aggregate_rows_are_json_serializable() -> None:
    rows = [
        _row("cache", "backbone_i2_t02_10", 0, 1.4, 0.03),
        _row("cache", "backbone_i2_t02_10", 1, 1.6, 0.05),
        _row("reduced_steps", "reduced_steps_15", 0, 1.5, 0.14),
    ]
    aggregate = aggregate_benchmark_rows(rows)
    cache_row = next(row for row in aggregate if row["method_name"] == "backbone_i2_t02_10")
    assert math.isclose(cache_row["speedup_mean"], 1.5)
    assert math.isclose(cache_row["rel_l2_mean"], 0.04)
    assert cache_row["seed_count"] == 2
    json.dumps(aggregate)
