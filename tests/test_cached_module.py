from __future__ import annotations

import torch
from torch import nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


class CountingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + float(self.calls)


def test_disabled_cache_calls_module_every_time() -> None:
    module = CountingModule()
    state = RuntimeCacheState(enabled=True)
    policy = FixedIntervalCachePolicy(enabled=False, interval=2)
    wrapped = CachedModule(module, "blocks.0", state, policy)

    for step in range(3):
        state.set_context(step, step / 10, "cond")
        wrapped(torch.zeros(1, 2))

    assert module.calls == 3
    assert state.summary()["disabled"] == 3


def test_interval_two_reuses_without_calling_wrapped_module() -> None:
    module = CountingModule()
    state = RuntimeCacheState()
    policy = FixedIntervalCachePolicy(interval=2)
    wrapped = CachedModule(module, "blocks.0", state, policy)

    state.set_context(0, 0.0, "cond")
    first = wrapped(torch.zeros(1, 2))
    state.set_context(1, 0.1, "cond")
    second = wrapped(torch.zeros(1, 2))
    state.set_context(2, 0.2, "cond")
    third = wrapped(torch.zeros(1, 2))

    assert module.calls == 2
    assert torch.equal(second, first)
    assert not torch.equal(third, first)
    summary = state.summary()
    assert summary["hits"] == 1
    assert summary["misses"] == 2
    assert summary["refreshes"] == 2


def test_batch_size_change_uses_safe_compute_path() -> None:
    module = CountingModule()
    state = RuntimeCacheState()
    policy = FixedIntervalCachePolicy(interval=2)
    wrapped = CachedModule(module, "blocks.0", state, policy)

    state.set_context(0, 0.0, "cond")
    wrapped(torch.zeros(1, 2))
    state.set_context(1, 0.1, "cond")
    out = wrapped(torch.zeros(2, 2))

    assert module.calls == 2
    assert out.shape == (2, 2)
    assert state.summary()["hits"] == 0
    assert state.summary()["misses"] == 2
