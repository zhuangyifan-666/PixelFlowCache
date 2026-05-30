from __future__ import annotations

import torch

from pfc.adapters.base import ModelAdapter


class DummyAdapter(ModelAdapter):
    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond, **kwargs):
        return x


def test_xpred_conversion_scalar_t() -> None:
    adapter = DummyAdapter("dummy_x", "xpred")
    x = torch.ones(2, 3, 4, 4)
    raw = x + 2.0
    velocity = adapter.raw_to_velocity(raw, x, 0.5)
    assert torch.allclose(velocity, torch.full_like(x, 4.0))


def test_xpred_conversion_vector_t_image() -> None:
    adapter = DummyAdapter("dummy_x", "xpred")
    x = torch.ones(2, 3, 4, 4)
    raw = x + 2.0
    t = torch.tensor([0.5, 0.75])
    velocity = adapter.raw_to_velocity(raw, x, t)
    expected = torch.stack([torch.full((3, 4, 4), 4.0), torch.full((3, 4, 4), 8.0)], dim=0)
    assert torch.allclose(velocity, expected)


def test_xpred_conversion_vector_t_tokens() -> None:
    adapter = DummyAdapter("dummy_x", "xpred")
    x = torch.ones(2, 5, 3)
    raw = x + 1.0
    t = torch.tensor([0.5, 0.75])
    velocity = adapter.raw_to_velocity(raw, x, t)
    expected = torch.stack([torch.full((5, 3), 2.0), torch.full((5, 3), 4.0)], dim=0)
    assert torch.allclose(velocity, expected)


def test_vpred_returns_raw() -> None:
    adapter = DummyAdapter("dummy_v", "vpred")
    x = torch.zeros(2, 3, 4, 4)
    raw = torch.randn_like(x)
    assert adapter.raw_to_velocity(raw, x, 0.9) is raw

