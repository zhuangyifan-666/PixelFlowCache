from __future__ import annotations

import torch

from pfc.adapters.base import ModelAdapter
from pfc.cache.base_policy import NoCachePolicy
from pfc.samplers.unified_sampler import UnifiedPixelFlowSampler


class ConstantVAdapter(ModelAdapter):
    def __init__(self, value: float) -> None:
        super().__init__("constant_v", "vpred")
        self.value = value

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond, **kwargs):
        return torch.full_like(x, self.value)


def test_euler_sampler_updates_predictably() -> None:
    adapter = ConstantVAdapter(2.0)
    sampler = UnifiedPixelFlowSampler(adapter, solver="euler", steps=4)
    noise = torch.zeros(2, 3, 4, 4)
    sample, diagnostics = sampler.sample(noise, cond=None)
    assert torch.allclose(sample, torch.full_like(noise, 2.0))
    assert sample.shape == noise.shape
    assert diagnostics["num_steps"] == 4


def test_no_cache_policy_path_runs() -> None:
    adapter = ConstantVAdapter(1.5)
    sampler = UnifiedPixelFlowSampler(adapter, solver="euler", steps=3)
    noise = torch.zeros(1, 1, 2, 2)
    sample, _ = sampler.sample(noise, cond=None, cache_policy=NoCachePolicy())
    assert torch.allclose(sample, torch.full_like(noise, 1.5))

