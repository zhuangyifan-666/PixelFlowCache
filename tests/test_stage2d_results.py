from __future__ import annotations

import json

from scripts.run_jit_stage2d_first_hit_delay import summarize_first_hit_rows
from scripts.run_jit_stage2d_seed_sweep import summarize_seed_rows
from scripts.run_jit_stage2d_validate_best_windows import summarize_validation_rows


def _row(label: str, seed: int, rel_l2: float, speedup: float, warmup: int = 0) -> dict:
    return {
        "config_label": label,
        "cache_layers": "all",
        "cache_interval": 2,
        "active_t_min": 0.1,
        "active_t_max": 0.8,
        "num_samples": 8,
        "steps": 20,
        "seed": seed,
        "speedup_median": speedup,
        "cache_hit_rate": 0.3,
        "same_seed_rel_l2": rel_l2,
        "same_seed_mse": rel_l2 * rel_l2,
        "same_seed_psnr": 30.0,
        "low_freq_delta_ratio": 0.1,
        "mid_freq_delta_ratio": 0.2,
        "high_freq_delta_ratio": 0.7,
        "run_dir": "logs/test",
        "active_window_warmup_refreshes": warmup,
    }


def test_stage2d_validation_summary_is_json_serializable() -> None:
    rows = [
        _row("a", 0, 0.04, 1.4),
        _row("b", 0, 0.03, 1.2),
        {**_row("none", 0, 0.0, 1.0), "cache_layers": "none"},
    ]
    summary = summarize_validation_rows(rows)
    assert summary["best_quality_row"]["same_seed_rel_l2"] == 0.03
    assert summary["fastest_row"]["speedup_median"] == 1.4
    json.dumps(summary)


def test_stage2d_seed_summary_mean_std_is_json_serializable() -> None:
    rows = [_row("a", 0, 0.04, 1.4), _row("a", 1, 0.06, 1.6), _row("b", 0, 0.03, 1.2)]
    summary = summarize_seed_rows(rows)
    assert summary["config_count"] == 2
    group_a = next(row for row in summary["summaries"] if row["config_label"] == "a")
    assert group_a["same_seed_rel_l2_mean"] == 0.05
    assert group_a["speedup_median_mean"] == 1.5
    json.dumps(summary)


def test_stage2d_first_hit_summary_tracks_delta_vs_zero() -> None:
    rows = [_row("a", 0, 0.08, 1.5, warmup=0), _row("a", 0, 0.05, 1.3, warmup=1)]
    summary = summarize_first_hit_rows(rows)
    assert summary["best_quality_row"]["active_window_warmup_refreshes"] == 1
    assert summary["fastest_row"]["active_window_warmup_refreshes"] == 0
    assert summary["improvements_vs_zero"][1]["rel_l2_delta_vs_zero"] < 0
    json.dumps(summary)
