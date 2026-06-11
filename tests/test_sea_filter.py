from __future__ import annotations

import torch

from pfc.cache.spectral_dynamic_policy import apply_sea_filter, make_sea_filter


def test_make_sea_filter_bchw_broadcastable() -> None:
    filt = make_sea_filter(
        16,
        12,
        t=0.5,
        beta=2.0,
        eps=1e-6,
        normalize_filter=True,
        device="cpu",
        dtype=torch.float32,
    )
    assert filt.shape == (1, 1, 16, 12)


def test_apply_sea_filter_preserves_shape() -> None:
    x = torch.randn(2, 3, 16, 16)
    y = apply_sea_filter(x, t=0.5, beta=2.0, eps=1e-6, normalize_filter=True)
    assert y.shape == x.shape


def test_sea_filter_no_nan_inf_at_boundary_times() -> None:
    x = torch.randn(1, 3, 8, 8)
    for t in (0.0, 0.5, 1.0):
        y = apply_sea_filter(x, t=t, beta=2.0, eps=1e-6, normalize_filter=True)
        assert torch.isfinite(y).all()


def test_normalized_filter_has_finite_mean() -> None:
    filt = make_sea_filter(
        32,
        32,
        t=0.25,
        beta=2.0,
        eps=1e-6,
        normalize_filter=True,
        device="cpu",
        dtype=torch.float32,
    )
    assert torch.isfinite(filt).all()
    assert torch.isclose(filt.mean(), torch.tensor(1.0), atol=1e-5)
