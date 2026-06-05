from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.collect_stage3c_unified_results import UNIFIED_FIELDNAMES
from scripts.plot_stage3c_unified import plot_stage3c


def _write_unified(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "JiT",
            "prediction_type": "xpred",
            "method_type": "cache",
            "method_name": "quality_t02_08",
            "boundary_type": "backbone",
            "reference_steps": 50,
            "eval_steps": 50,
            "num_samples": 16,
            "seed_count": 3,
            "speedup_mean": 1.4,
            "speedup_std": 0.01,
            "rel_l2_mean": 0.02,
            "rel_l2_std": 0.001,
            "psnr_mean": 46.0,
            "psnr_std": 1.0,
            "hit_rate_mean": 0.3,
            "reduced_step_reference_match": "reduced_steps_35",
            "notes": "source=jit_stage3a",
        },
        {
            "model": "JiT",
            "prediction_type": "xpred",
            "method_type": "reduced_steps",
            "method_name": "reduced_steps_35",
            "boundary_type": "none",
            "reference_steps": 50,
            "eval_steps": 35,
            "num_samples": 16,
            "seed_count": 3,
            "speedup_mean": 1.42,
            "speedup_std": 0.01,
            "rel_l2_mean": 0.13,
            "rel_l2_std": 0.01,
            "psnr_mean": 30.0,
            "psnr_std": 1.0,
            "hit_rate_mean": 0.0,
            "reduced_step_reference_match": "",
            "notes": "source=jit_stage3a",
        },
        {
            "model": "DeCo",
            "prediction_type": "vpred",
            "method_type": "cache",
            "method_name": "all_candidates",
            "boundary_type": "all_candidates",
            "reference_steps": 50,
            "eval_steps": 50,
            "num_samples": 16,
            "seed_count": 1,
            "speedup_mean": 1.6,
            "speedup_std": 0.0,
            "rel_l2_mean": 0.046,
            "rel_l2_std": 0.0,
            "psnr_mean": 39.6,
            "psnr_std": 0.0,
            "hit_rate_mean": 0.4,
            "reduced_step_reference_match": "reduced_steps_30",
            "notes": "source=deco_validation",
        },
        {
            "model": "DeCo",
            "prediction_type": "vpred",
            "method_type": "reduced_steps",
            "method_name": "reduced_steps_30",
            "boundary_type": "none",
            "reference_steps": 50,
            "eval_steps": 30,
            "num_samples": 16,
            "seed_count": 1,
            "speedup_mean": 1.65,
            "speedup_std": 0.0,
            "rel_l2_mean": 0.25,
            "rel_l2_std": 0.0,
            "psnr_mean": 24.8,
            "psnr_std": 0.0,
            "hit_rate_mean": 0.0,
            "reduced_step_reference_match": "",
            "notes": "source=deco_validation",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_stage3c_plot_reads_fake_unified_csv(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    unified_dir = tmp_path / "unified"
    _write_unified(unified_dir / "unified_results.csv")
    paths = plot_stage3c(unified_dir, tmp_path / "figures")
    assert len(paths) == 5
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
