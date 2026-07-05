from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_jit_safe_1000_results_writes_summary(tmp_path: Path) -> None:
    run_id = "unit_run"
    output_root = tmp_path / "outputs"
    fid_root = tmp_path / "fid"
    pair_root = tmp_path / "pair"
    out_dir = tmp_path / "summary"
    methods = ["no_cache_50", "safe_bfc_quality"]

    for method, ips in [("no_cache_50", 5.0), ("safe_bfc_quality", 10.0)]:
        run_dir = output_root / "jit" / run_id / method
        _write_json(run_dir / "generation_meta.json", {"num_images": 1000, "eval_steps": 50})
        _write_json(run_dir / "latency.json", {"latency_sec": 200.0 / ips, "images_per_sec": ips, "generated_images": 1000})
        cache_payload = {"hit_rate": 0.0, "total_calls": 0, "hits": 0, "refreshes": 0}
        if method == "safe_bfc_quality":
            cache_payload["safe_policy"] = {
                "config": {"max_age": 2, "safe_lambda": 0.5, "quantile": 0.95},
                "stats": {"safe_reuse": 7, "unsafe_refresh": 3, "mean_age": 1.5},
            }
        _write_json(run_dir / "cache_stats.json", cache_payload)
        _write_json(fid_root / run_id / "jit" / method / "fid_results.json", {"fid": 12.3, "is": 45.6})

    _write_json(
        pair_root / run_id / "jit" / "safe_bfc_quality" / "pair_metrics.json",
        {
            "summary": {
                "psnr": {"mean": 30.0},
                "ssim": {"mean": 0.9},
                "lpips": {"mean": 0.1},
                "rel_l2": {"mean": 0.02},
            }
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_jit_safe_1000_results.py",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--fid-root",
            str(fid_root),
            "--pair-root",
            str(pair_root),
            "--methods",
            ",".join(methods),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    rows = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    safe = next(row for row in rows if row["method"] == "safe_bfc_quality")
    assert float(safe["speedup_vs_no_cache"]) == 2.0
    assert safe["safe_reuse"] == "7"
    assert "1000-image proxy results" in (out_dir / "summary.md").read_text(encoding="utf-8")
