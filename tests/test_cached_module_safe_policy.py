from __future__ import annotations

import torch
from torch import nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
from pfc.cache.safe_map_policy import SafeMapCachePolicy


class CountingLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + float(self.calls)


def _policy() -> SafeMapCachePolicy:
    return SafeMapCachePolicy(
        safe_map={
            "policy_name": "SafeMapCachePolicy",
            "model_name": "JiT",
            "solver_stages": ["euler"],
            "branches": ["global"],
            "boundary_groups": {"jit_whole_backbone": ["blocks.0"]},
            "module_to_boundary": {"blocks.0": "jit_whole_backbone"},
            "max_age": 1,
            "safe": {"euler": {"global": {"jit_whole_backbone": {"1": {"1": True}}}}},
        }
    )


def test_cached_module_uses_safe_policy_entry_age() -> None:
    module = CountingLinear()
    state = RuntimeCacheState()
    wrapped = CachedModule(module, "blocks.0", state, _policy())

    state.set_context(0, 0.0, "cond")
    first = wrapped(torch.zeros(1, 2))
    state.set_context(1, 0.1, "cond")
    second = wrapped(torch.zeros(1, 2))

    assert module.calls == 1
    assert torch.equal(first, second)
    summary = state.summary()
    assert summary["misses"] == 1
    assert summary["hits"] == 1


def test_cached_module_fixed_interval_behavior_is_unchanged() -> None:
    module = CountingLinear()
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
