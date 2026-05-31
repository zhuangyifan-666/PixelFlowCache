from __future__ import annotations

import json
from pathlib import Path

from scripts.run_jit_stage2b_cache import Stage2BConfig
from scripts.run_jit_stage2c_window_ablation import (
    choose_best_window,
    result_row_from_stage2b,
    summarize_window_results,
)


def _config(t_min: float, t_max: float, interval: int = 2) -> Stage2BConfig:
    return Stage2BConfig(
        jit_dir=Path("third_party/JiT"),
        ckpt_dir=Path("ckpts/JiT/JiT-B-16-256"),
        run_id="test",
        run_dir=Path("logs/test"),
        preview_dir=Path("outputs/test"),
        cache_interval=interval,
        cache_layers="all",
        active_t_min=t_min,
        active_t_max=t_max,
    )


def _result(rel_l2: float, speedup: float) -> dict:
    return {
        "comparison": {
            "speedup_median": speedup,
            "same_seed_rel_l2": rel_l2,
            "same_seed_mse": rel_l2 * rel_l2,
            "same_seed_psnr": 30.0,
            "frequency_delta": {"low_ratio": 0.1, "mid_ratio": 0.2, "high_ratio": 0.7},
        },
        "cache_stats": {"hit_rate": 0.5},
        "selected_layer_ids": [0, 1],
        "run_dir": "logs/test/run",
    }


def test_result_row_from_stage2b_is_json_serializable() -> None:
    row = result_row_from_stage2b(_result(0.1, 1.5), _config(0.1, 0.8), "t_max")
    assert row["active_t_min"] == 0.1
    assert row["active_t_max"] == 0.8
    assert row["cache_interval"] == 2
    assert row["low_freq_delta_ratio"] == 0.1
    json.dumps(row)


def test_choose_best_window_prefers_low_rel_l2_then_speed() -> None:
    rows = [
        result_row_from_stage2b(_result(0.2, 2.0), _config(0.0, 0.8), "t_min"),
        result_row_from_stage2b(_result(0.1, 1.2), _config(0.1, 0.8), "t_min"),
        result_row_from_stage2b(_result(0.1, 1.4), _config(0.1, 0.9), "t_max"),
    ]
    assert choose_best_window(rows) == (0.1, 0.9)


def test_window_summary_is_json_serializable() -> None:
    rows = [
        result_row_from_stage2b(_result(0.2, 2.0), _config(0.0, 0.8), "t_min"),
        result_row_from_stage2b(_result(0.1, 1.2), _config(0.1, 0.8), "t_min"),
    ]
    summary = summarize_window_results(rows)
    assert summary["record_count"] == 2
    assert summary["best_quality_row"]["same_seed_rel_l2"] == 0.1
    assert summary["fastest_row"]["speedup_median"] == 2.0
    json.dumps(summary)
