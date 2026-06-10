from __future__ import annotations

from scripts.plot_stage4a_full_eval import filter_rows_for_plot


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
