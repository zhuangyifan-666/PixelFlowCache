from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.collect_stage3c_unified_results import UNIFIED_FIELDNAMES, collect_stage3c_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_stage3c_unified_schema_from_fake_jit_and_deco(tmp_path: Path) -> None:
    jit_dir = tmp_path / "jit"
    deco_dir = tmp_path / "deco_validate"
    _write_csv(
        jit_dir / "benchmark_aggregate.csv",
        [
            {
                "method_type": "cache",
                "method_name": "quality_t02_08",
                "seed_count": 3,
                "num_samples": 16,
                "reference_steps": 50,
                "eval_steps": 50,
                "cache_layers": "all",
                "cache_interval": 2,
                "active_t_min": 0.2,
                "active_t_max": 0.8,
                "active_window_warmup_refreshes": 0,
                "speedup_mean": 1.4,
                "speedup_std": 0.01,
                "rel_l2_mean": 0.02,
                "rel_l2_std": 0.001,
                "psnr_mean": 46.0,
                "psnr_std": 1.0,
                "hit_rate_mean": 0.3,
            },
            {
                "method_type": "reduced_steps",
                "method_name": "reduced_steps_35",
                "seed_count": 3,
                "num_samples": 16,
                "reference_steps": 50,
                "eval_steps": 35,
                "cache_layers": "none",
                "cache_interval": 1,
                "active_t_min": "",
                "active_t_max": "",
                "active_window_warmup_refreshes": 0,
                "speedup_mean": 1.42,
                "speedup_std": 0.01,
                "rel_l2_mean": 0.13,
                "rel_l2_std": 0.01,
                "psnr_mean": 30.0,
                "psnr_std": 1.0,
                "hit_rate_mean": 0.0,
            },
        ],
    )
    _write_csv(
        deco_dir / "validation_aggregate.csv",
        [
            {
                "method_type": "cache",
                "method_name": "all_candidates",
                "seed_count": 1,
                "num_samples": 16,
                "reference_steps": 50,
                "eval_steps": 50,
                "cache_units": "all_candidates",
                "cache_interval": 2,
                "active_t_min": 0.2,
                "active_t_max": 1.0,
                "speedup_mean": 1.6,
                "speedup_std": 0.0,
                "rel_l2_mean": 0.046,
                "rel_l2_std": 0.0,
                "psnr_mean": 39.6,
                "psnr_std": 0.0,
                "hit_rate_mean": 0.4,
                "has_final_cache": True,
                "has_backbone_cache": True,
                "has_decoder_cache": True,
            },
            {
                "method_type": "reduced_steps",
                "method_name": "reduced_steps_30",
                "seed_count": 1,
                "num_samples": 16,
                "reference_steps": 50,
                "eval_steps": 30,
                "cache_units": "none",
                "cache_interval": 1,
                "active_t_min": "",
                "active_t_max": "",
                "speedup_mean": 1.65,
                "speedup_std": 0.0,
                "rel_l2_mean": 0.25,
                "rel_l2_std": 0.0,
                "psnr_mean": 24.8,
                "psnr_std": 0.0,
                "hit_rate_mean": 0.0,
                "has_final_cache": False,
                "has_backbone_cache": False,
                "has_decoder_cache": False,
            },
        ],
    )
    out_dir = collect_stage3c_results(
        jit_benchmark_dir=jit_dir,
        deco_validation_dir=deco_dir,
        deco_seed_dir=None,
        deco_decomposition_dir=None,
        output_dir=tmp_path / "unified",
        auto_detect=False,
    )
    with (out_dir / "unified_results.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert set(UNIFIED_FIELDNAMES).issubset(rows[0].keys())
    jit = next(row for row in rows if row["model"] == "JiT" and row["method_name"] == "quality_t02_08")
    deco = next(row for row in rows if row["model"] == "DeCo" and row["method_name"] == "all_candidates")
    assert jit["prediction_type"] == "xpred"
    assert jit["boundary_type"] == "backbone"
    assert jit["reduced_step_reference_match"] == "reduced_steps_35"
    assert deco["prediction_type"] == "vpred"
    assert deco["boundary_type"] == "all_candidates"
    assert deco["reduced_step_reference_match"] == "reduced_steps_30"
    json.loads((out_dir / "unified_results.json").read_text(encoding="utf-8"))
