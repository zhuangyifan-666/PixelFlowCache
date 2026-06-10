from __future__ import annotations

import torch

from pfc.diagnostics.frequency import fft_frequency_bands, frequency_delta_bands


def test_fft_frequency_bands_bchw() -> None:
    x = torch.randn(2, 3, 16, 16)
    record = fft_frequency_bands(x)
    assert record["shape"] == [2, 3, 16, 16]
    ratio_sum = record["low_ratio"] + record["mid_ratio"] + record["high_ratio"]
    assert abs(ratio_sum - 1.0) < 1e-5
    assert record["total_energy"] > 0


def test_frequency_delta_bands() -> None:
    previous = torch.zeros(1, 1, 8, 8)
    current = torch.ones(1, 1, 8, 8)
    record = frequency_delta_bands(current, previous)
    assert record["total_energy"] > 0
    assert "rel_l2_delta" in record
