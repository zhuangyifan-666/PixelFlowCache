from __future__ import annotations

import json
import math

from scripts.make_stage3a_report_tables import make_paper_table_rows
from scripts.run_jit_stage3a_backbone_benchmark import aggregate_benchmark_rows


def _row(method_type: str, method_name: str, seed: int, speedup: float, rel_l2: float) -> dict:
    return {
        "method_type": method_type,
        "method_name": method_name,
        "seed": seed,
        "num_samples": 16,
        "reference_steps": 50,
        "eval_steps": 50 if method_type != "reduced_steps" else 35,
        "cache_layers": "all" if method_type == "cache" else "none",
        "cache_interval": 2 if method_type == "cache" else 1,
        "active_t_min": 0.2 if method_type == "cache" else None,
        "active_t_max": 0.8 if method_type == "cache" else None,
        "active_window_warmup_refreshes": 0,
        "latency_median_sec": 1.0,
        "speedup_vs_reference": speedup,
        "cache_hit_rate": 0.3 if method_type == "cache" else 0.0,
        "same_seed_rel_l2": rel_l2,
        "same_seed_mse": rel_l2 * rel_l2,
        "same_seed_psnr": 40.0,
        "low_freq_delta_ratio": 0.1,
        "mid_freq_delta_ratio": 0.2,
        "high_freq_delta_ratio": 0.7,
        "run_dir": "logs/test",
    }


def test_stage3a_aggregate_mean_std_over_seeds() -> None:
    rows = [
        _row("cache", "quality_t02_08", 0, 1.4, 0.02),
        _row("cache", "quality_t02_08", 1, 1.6, 0.04),
        _row("reduced_steps", "reduced_steps_35", 0, 1.5, 0.08),
    ]
    aggregate = aggregate_benchmark_rows(rows)
    quality = next(row for row in aggregate if row["method_name"] == "quality_t02_08")
    assert math.isclose(quality["speedup_mean"], 1.5)
    assert math.isclose(quality["rel_l2_mean"], 0.03)
    assert quality["seed_count"] == 2
    json.dumps(aggregate)


def test_stage3a_paper_table_fields_are_generated() -> None:
    aggregate = aggregate_benchmark_rows(
        [
            _row("cache", "quality_t02_08", 0, 1.4, 0.02),
            _row("reduced_steps", "reduced_steps_35", 0, 1.5, 0.08),
        ]
    )
    table = make_paper_table_rows(aggregate)
    quality = next(row for row in table if row["method"] == "quality_t02_08")
    assert quality["type"] == "cache"
    assert "Quality-first" in quality["notes"]
    assert "closest reduced-step" in quality["notes"]
    json.dumps(table)
