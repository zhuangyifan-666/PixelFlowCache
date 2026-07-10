from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

from pfc.cache.cache_state import RuntimeCacheState


ROOT = Path(__file__).resolve().parents[1]


class _Scheduler:
    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    @staticmethod
    def dalpha_over_alpha(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    @staticmethod
    def dsigma_mul_sigma(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


class _FakeEulerSampler:
    def __init__(self, num_steps: int = 4, **_kwargs: object) -> None:
        self.num_steps = num_steps
        self.timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
        self.scheduler = _Scheduler()
        self.w_scheduler = None
        self.guidance = 2.0
        self.guidance_interval_min = 0.1
        self.guidance_interval_max = 1.0

    @staticmethod
    def guidance_fn(out: torch.Tensor, guidance: float) -> torch.Tensor:
        uncondition, condition = out.chunk(2)
        return uncondition + guidance * (condition - uncondition)

    @staticmethod
    def step_fn(x: torch.Tensor, v: torch.Tensor, dt: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return x + v * dt

    last_step_fn = step_fn

    def __call__(self, net: nn.Module, noise: torch.Tensor, condition: torch.Tensor, uncondition: torch.Tensor) -> torch.Tensor:
        trajectories, _ = self._impl_sampling(net, noise, condition, uncondition)
        return trajectories[-1]


class _FakeNet(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return x * 0.25 + labels.to(x.dtype).view(-1, 1, 1, 1) * 1e-3


def _load_sampler_class():
    sampling = types.ModuleType("src.diffusion.flow_matching.sampling")
    sampling.EulerSampler = _FakeEulerSampler
    modules = {
        "src": types.ModuleType("src"),
        "src.diffusion": types.ModuleType("src.diffusion"),
        "src.diffusion.flow_matching": types.ModuleType("src.diffusion.flow_matching"),
        "src.diffusion.flow_matching.sampling": sampling,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "_pfc_test_deco_cached_sampler", ROOT / "pfc/cache/deco_cached_sampler.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CachedDeCoEulerSampler
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_deco_cached_sampler_disabled_state_matches_no_cache() -> None:
    sampler_class = _load_sampler_class()
    labels = torch.tensor([3, 4])
    uncondition = torch.tensor([1000, 1000])
    noise = torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(5))
    reference = sampler_class(cache_state=None, log_diagnostics=False, num_steps=4)(
        _FakeNet(), noise, labels, uncondition
    )
    disabled = sampler_class(
        cache_state=RuntimeCacheState(model_name="DeCo", enabled=False),
        log_diagnostics=False,
        num_steps=4,
    )(_FakeNet(), noise, labels, uncondition)
    assert torch.equal(reference, disabled)


def test_deco_hot_path_has_no_per_step_item_or_cpu_scalar_conversion() -> None:
    source = (ROOT / "pfc/cache/deco_cached_sampler.py").read_text(encoding="utf-8")
    assert ".item(" not in source
    assert "float(t_cur" not in source
