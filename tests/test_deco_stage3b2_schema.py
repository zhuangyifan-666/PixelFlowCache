from __future__ import annotations

import json
import math

from scripts.deco_stage3b2_common import STAGE3B2_FIELDNAMES, aggregate_rows


def _row(method_type: str, method_name: str, seed: int, speedup: float, rel_l2: float) -> dict:
    return {
        "method_type": method_type,
        "method_name": method_name,
        "seed": seed,
        "num_samples": 8,
        "reference_steps": 20,
        "eval_steps": 20 if method_type != "reduced_steps" else 12,
        "cache_units": method_name if method_type == "cache" else "none",
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
        "wrapped_module_count": 1 if method_type == "cache" else 0,
        "wrapped_modules": "dec_net.final_layer" if method_type == "cache" else "",
        "has_final_cache": method_type == "cache",
        "has_backbone_cache": False,
        "has_decoder_cache": False,
        "run_dir": "logs/test",
    }


def test_stage3b2_required_columns_exist() -> None:
    row = _row("cache", "final_only", 0, 1.1, 0.04)
    assert set(STAGE3B2_FIELDNAMES).issubset(row.keys())


def test_stage3b2_aggregate_rows_are_json_serializable() -> None:
    rows = [
        _row("cache", "final_only", 0, 1.1, 0.04),
        _row("cache", "final_only", 1, 1.3, 0.06),
        _row("reduced_steps", "reduced_steps_12", 0, 1.6, 0.2),
    ]
    aggregate = aggregate_rows(rows)
    final = next(row for row in aggregate if row["method_name"] == "final_only")
    assert math.isclose(final["speedup_mean"], 1.2)
    assert math.isclose(final["rel_l2_mean"], 0.05)
    assert final["seed_count"] == 2
    assert final["has_final_cache"] is True
    json.dumps(aggregate)
