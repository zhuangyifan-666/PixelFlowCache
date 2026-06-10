"""Lightweight diagnostics used by final BoundaryFlowCache runtime code."""

from pfc.diagnostics.frequency import fft_frequency_bands, frequency_delta_bands
from pfc.diagnostics.tensor_stats import l2_norm, relative_l2_delta, summarize_tensor

__all__ = [
    "fft_frequency_bands",
    "frequency_delta_bands",
    "l2_norm",
    "relative_l2_delta",
    "summarize_tensor",
]
