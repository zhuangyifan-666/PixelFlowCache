from __future__ import annotations

import torch

from pfc.adapters.base import ModelAdapter
from pfc.samplers.unified_sampler import UnifiedPixelFlowSampler


class CondDependentAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__("cond_dependent", "vpred")

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond, **kwargs):
        return torch.full_like(x, float(cond))


def test_cfg_formula() -> None:
    adapter = CondDependentAdapter()
    sampler = UnifiedPixelFlowSampler(adapter, solver="euler", steps=1, cfg_scale=3.0)
    x = torch.zeros(1, 1, 2, 2)
    velocity, diagnostics = sampler.predict_velocity(x, 0.5, cond=2.0, uncond=1.0)
    expected = torch.full_like(x, 1.0 + 3.0 * (2.0 - 1.0))
    assert torch.allclose(velocity, expected)
    assert diagnostics["cfg_enabled"] is True


def test_cfg_interval_disables_guidance_outside_interval() -> None:
    adapter = CondDependentAdapter()
    sampler = UnifiedPixelFlowSampler(adapter, solver="euler", steps=1, cfg_scale=3.0, cfg_interval=(0.1, 0.9))
    x = torch.zeros(1, 1, 2, 2)
    velocity, diagnostics = sampler.predict_velocity(x, 0.95, cond=2.0, uncond=1.0)
    assert torch.allclose(velocity, torch.full_like(x, 2.0))
    assert diagnostics["cfg_enabled"] is False

