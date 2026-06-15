from __future__ import annotations

import json

import torch
import torch.nn as nn

from pfc.adapters import PixelGenBoundaryAdapter
from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
from pfc.core.registry import get_adapter


class FakePixelGenDenoiser(nn.Module):
    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(2, 2)


class FakePixelGenWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.denoiser = FakePixelGenDenoiser(num_blocks=2)
        self.ema_denoiser = FakePixelGenDenoiser(num_blocks=3)


def test_pixelgen_adapter_registry() -> None:
    assert get_adapter("pixelgen") is PixelGenBoundaryAdapter


def test_pixelgen_adapter_lists_jit_style_blocks() -> None:
    adapter = PixelGenBoundaryAdapter()
    model = FakePixelGenDenoiser(num_blocks=4)
    candidates = {candidate.name: candidate for candidate in adapter.list_boundary_candidates(model)}
    assert candidates["jit_style_blocks"].module_names == ("blocks.0", "blocks.1", "blocks.2", "blocks.3")
    assert candidates["jit_style_blocks"].is_quality_critical
    assert candidates["jit_style_blocks"].is_speed_critical


def test_pixelgen_default_boundary_set_uses_ema_wrapper_blocks() -> None:
    adapter = PixelGenBoundaryAdapter()
    wrapper = FakePixelGenWrapper()
    boundary_set = adapter.default_boundary_set(wrapper, "bfc_quality_t02_08")
    assert boundary_set.name == "pixelgen_jit_style_blocks"
    assert boundary_set.module_names() == ("blocks.0", "blocks.1", "blocks.2")
    json.dumps(adapter.describe())
    json.dumps(boundary_set.to_dict())


def test_pixelgen_wrap_boundary_set_installs_cached_modules() -> None:
    adapter = PixelGenBoundaryAdapter()
    model = FakePixelGenDenoiser(num_blocks=3)
    boundary_set = adapter.default_boundary_set(model)
    cache_state = RuntimeCacheState(model_name="PixelGen")
    policy = FixedIntervalCachePolicy(
        interval=2,
        cache_modules=set(boundary_set.module_names()),
        solver_stages={"heun_predictor", "heun_corrector"},
    )
    wrapped = adapter.wrap_boundary_set(model, boundary_set, cache_state, policy)
    assert wrapped == ["blocks.0", "blocks.1", "blocks.2"]
    assert all(isinstance(block, CachedModule) for block in model.blocks)
    assert not isinstance(model.final_layer, CachedModule)


def test_pixelgen_output_to_velocity_uses_pixelgen_eps() -> None:
    adapter = PixelGenBoundaryAdapter()
    x = torch.zeros(1, 1, 1, 1)
    out = torch.ones(1, 1, 1, 1)
    velocity = adapter.output_to_velocity(out, x, torch.tensor([0.99]))
    assert torch.allclose(velocity, torch.full_like(velocity, 20.0))
