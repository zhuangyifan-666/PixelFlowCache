from __future__ import annotations

import csv
from pathlib import Path

from scripts.collect_stage3c_unified_results import UNIFIED_FIELDNAMES
from scripts.make_stage3c_paper_tables import closest_reduced_step, make_main_table, write_stage3c_tables


def _write_unified(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in UNIFIED_FIELDNAMES})


def _row(
    model: str,
    method_type: str,
    method_name: str,
    speedup: float,
    rel_l2: float,
    source: str,
    prediction_type: str = "xpred",
) -> dict[str, object]:
    return {
        "model": model,
        "prediction_type": prediction_type,
        "method_type": method_type,
        "method_name": method_name,
        "boundary_type": "backbone" if model == "JiT" and method_type == "cache" else "none",
        "reference_steps": 50,
        "eval_steps": 50 if method_type != "reduced_steps" else 35,
        "num_samples": 16,
        "seed_count": 3 if model == "JiT" else 1,
        "speedup_mean": speedup,
        "speedup_std": 0.01,
        "rel_l2_mean": rel_l2,
        "rel_l2_std": 0.001,
        "psnr_mean": 40.0,
        "psnr_std": 0.1,
        "hit_rate_mean": 0.3 if method_type == "cache" else 0.0,
        "reduced_step_reference_match": "",
        "notes": f"source={source}",
    }


def test_stage3c_main_table_and_closest_reduced_match(tmp_path: Path) -> None:
    rows = [
        _row("JiT", "cache", "quality_t02_08", 1.4, 0.02, "jit_stage3a"),
        _row("JiT", "cache", "speed_t02_10", 1.6, 0.03, "jit_stage3a"),
        _row("JiT", "reduced_steps", "reduced_steps_35", 1.42, 0.13, "jit_stage3a"),
        _row("JiT", "reduced_steps", "reduced_steps_30", 1.66, 0.18, "jit_stage3a"),
        _row("DeCo", "cache", "all_candidates", 1.6, 0.046, "deco_validation", "vpred"),
        _row("DeCo", "cache", "backbone_plus_final", 1.48, 0.046, "deco_validation", "vpred"),
        _row("DeCo", "cache", "final_only", 1.0, 0.046, "deco_validation", "vpred"),
        _row("DeCo", "cache", "backbone_only", 1.46, 0.069, "deco_validation", "vpred"),
        _row("DeCo", "reduced_steps", "reduced_steps_30", 1.65, 0.25, "deco_validation", "vpred"),
    ]
    unified_dir = tmp_path / "unified"
    _write_unified(unified_dir / "unified_results.csv", rows)

    reduced = closest_reduced_step(rows[0], rows)  # type: ignore[arg-type]
    assert reduced is not None
    assert reduced["method_name"] == "reduced_steps_35"

    main_rows = make_main_table(rows)  # type: ignore[arg-type]
    methods = [row["method"] for row in main_rows]
    assert "quality_t02_08" in methods
    assert "reduced_steps_35" in methods
    assert "all_candidates" in methods
    assert "reduced_steps_30" in methods

    outputs = write_stage3c_tables(unified_dir)
    assert unified_dir / "paper_table_main_cache_vs_reduced.md" in outputs
    assert (unified_dir / "paper_table_boundary_ablation.csv").exists()
    assert (unified_dir / "paper_table_seed_stability.csv").exists()
