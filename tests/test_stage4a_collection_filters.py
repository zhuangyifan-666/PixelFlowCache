from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_stage4a_fid_results import collect_results
from scripts.plot_stage4a_full_eval import filter_rows_for_plot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_method(
    root: Path,
    fid_root: Path,
    *,
    model: str = "jit",
    run_id: str,
    method: str,
    num_images: int,
    latency_sec: float,
) -> Path:
    method_dir = root / model / run_id / method
    (method_dir / "images").mkdir(parents=True)
    _write_json(
        method_dir / "generation_meta.json",
        {
            "model": model,
            "run_id": run_id,
            "num_images": num_images,
            "method": {
                "method_name": method,
                "eval_steps": 50,
            },
        },
    )
    _write_json(method_dir / "latency.json", {"latency_sec": latency_sec, "images_per_sec": num_images / latency_sec})
    _write_json(method_dir / "cache_stats.json", {"hit_rate": 0.0 if method == "no_cache_50" else 0.4})
    _write_json(
        fid_root / run_id / model / method / "fid_results.json",
        {
            "fake_dir": str((method_dir / "images").resolve()),
            "frechet_inception_distance": 1.0,
            "inception_score_mean": 2.0,
            "backend": "torch_fidelity",
        },
    )
    return method_dir


def test_stage4a_speedup_references_do_not_mix_num_images_or_run_id(tmp_path: Path) -> None:
    root = tmp_path / "full_generation"
    fid_root = tmp_path / "fid"
    _write_method(root, fid_root, run_id="stage4a_n100_seed0", method="no_cache_50", num_images=100, latency_sec=10)
    _write_method(root, fid_root, run_id="stage4a_n100_seed0", method="bfc", num_images=100, latency_sec=5)
    _write_method(
        root, fid_root, run_id="stage4a_n50000_seed0", method="no_cache_50", num_images=50000, latency_sec=1000
    )
    _write_method(root, fid_root, run_id="stage4a_n50000_seed0", method="bfc", num_images=50000, latency_sec=250)

    rows = collect_results(root, fid_root, [])
    by_key = {(row["run_id"], row["num_images"], row["method"]): row for row in rows}

    assert by_key[("stage4a_n100_seed0", 100, "no_cache_50")]["speedup_vs_no_cache"] == 1.0
    assert by_key[("stage4a_n100_seed0", 100, "bfc")]["speedup_vs_no_cache"] == 2.0
    assert by_key[("stage4a_n50000_seed0", 50000, "no_cache_50")]["speedup_vs_no_cache"] == 1.0
    assert by_key[("stage4a_n50000_seed0", 50000, "bfc")]["speedup_vs_no_cache"] == 4.0
    assert by_key[("stage4a_n100_seed0", 100, "bfc")]["reference_key"] != by_key[
        ("stage4a_n50000_seed0", 50000, "bfc")
    ]["reference_key"]


def test_stage4a_collect_filters_run_id_and_num_images(tmp_path: Path) -> None:
    root = tmp_path / "full_generation"
    fid_root = tmp_path / "fid"
    _write_method(root, fid_root, run_id="stage4a_n100_seed0", method="no_cache_50", num_images=100, latency_sec=10)
    _write_method(root, fid_root, run_id="stage4a_n50000_seed0", method="no_cache_50", num_images=50000, latency_sec=1000)
    _write_method(root, fid_root, run_id="stage4a_n50000_seed0", method="bfc", num_images=50000, latency_sec=250)

    rows = collect_results(root, fid_root, [], run_id="stage4a_n50000_seed0", num_images=50000)

    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"no_cache_50", "bfc"}
    assert {row["num_images"] for row in rows} == {50000}
    assert {row["run_id"] for row in rows} == {"stage4a_n50000_seed0"}


def test_stage4a_plot_filter_selects_50k_rows() -> None:
    rows = [
        {"model": "jit", "run_id": "stage4a_n100_seed0", "method": "no_cache_50", "num_images": "100"},
        {"model": "jit", "run_id": "stage4a_n100_seed0", "method": "bfc", "num_images": "100"},
        {"model": "jit", "run_id": "stage4a_n50000_seed0", "method": "no_cache_50", "num_images": "50000"},
        {"model": "jit", "run_id": "stage4a_n50000_seed0", "method": "bfc", "num_images": "50000"},
    ]

    explicit = filter_rows_for_plot(rows, num_images=50000)
    default_largest = filter_rows_for_plot(rows, warn=False)

    assert [row["method"] for row in explicit] == ["no_cache_50", "bfc"]
    assert [row["method"] for row in default_largest] == ["no_cache_50", "bfc"]
    assert all(row["num_images"] == "50000" for row in explicit)
    assert all(row["num_images"] == "50000" for row in default_largest)
