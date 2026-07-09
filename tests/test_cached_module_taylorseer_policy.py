from __future__ import annotations

import torch
from torch import nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.taylorseer_policy import TaylorSeerCachePolicy


class CountingAdd(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + float(self.calls)


def test_cached_module_taylorseer_forecast_skips_original_module() -> None:
    module = CountingAdd()
    state = RuntimeCacheState(model_name="JiT")
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=4, max_order=1, min_history=2)
    wrapped = CachedModule(module, "blocks.0", state, policy)
    x = torch.zeros(1, 2)

    state.set_context(0, 0.0, "cond")
    first = wrapped(x)
    state.set_context(1, 0.1, "cond")
    second = wrapped(x)
    state.set_context(2, 0.2, "cond")
    third = wrapped(x)

    assert module.calls == 2
    assert torch.allclose(first, torch.ones_like(first))
    assert torch.allclose(second, torch.full_like(second, 2.0))
    assert torch.allclose(third, torch.full_like(third, 3.0))
    assert state.summary()["hits"] == 1
    assert state.summary()["misses"] == 2
    summary = policy.summary()["stats"]
    assert summary["forecast_decisions"] == 1
    assert summary["forecast_committed"] == 1
    assert summary["history_appends"] == 2
