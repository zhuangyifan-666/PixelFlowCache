from __future__ import annotations

import torch
import torch.nn as nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.dynamic_policy_adapter import DynamicPolicyAdapter
from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy


def test_adapter_exposes_fixed_interval_like_interface() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=0.5)
    adapter = DynamicPolicyAdapter(policy, cache_modules={"blocks.0"})
    assert adapter.enabled is True
    assert adapter.solver_stages == {"euler"}
    assert adapter.should_cache_module("blocks.0")
    assert not adapter.should_cache_module("blocks.1")
    assert adapter.is_branch_enabled("cond")


def test_adapter_refresh_and_reuse_follow_last_decision() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=1.0)
    adapter = DynamicPolicyAdapter(policy, cache_modules={"blocks.0"})
    policy.update(torch.ones(1, 1, 4, 4), step_idx=0, t=0.0)
    assert adapter.should_refresh(0, 0.0, "blocks.0", "cond", "euler")
    assert not adapter.should_reuse(0, 0.0, "blocks.0", "cond", "euler")
    policy.update(torch.ones(1, 1, 4, 4) * 1.01, step_idx=1, t=0.1)
    assert not adapter.should_refresh(1, 0.1, "blocks.0", "cond", "euler")
    assert adapter.should_reuse(1, 0.1, "blocks.0", "cond", "euler")


def test_adapter_module_filtering_disables_reuse_for_unselected_modules() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=1.0)
    adapter = DynamicPolicyAdapter(policy, cache_modules={"blocks.0"})
    policy.update(torch.ones(1, 1, 4, 4), step_idx=0, t=0.0)
    assert adapter.should_refresh(0, 0.0, "blocks.1", "cond", "euler")


def test_adapter_is_active_for_cached_module_dynamic_policy() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=1.0)
    adapter = DynamicPolicyAdapter(
        policy,
        cache_modules={"blocks.0"},
        solver_stages={"heun_predictor"},
    )
    cache_state = RuntimeCacheState(model_name="PixelGen", enabled=True)
    cached = CachedModule(nn.Identity(), "blocks.0", cache_state, adapter)
    x = torch.ones(1, 1, 4, 4)

    policy.update(x, step_idx=0, t=0.1, branch="cfg_cat")
    cache_state.set_context(0, 0.1, "cfg_cat", solver_stage="heun_predictor")
    cached(x)
    assert cache_state.summary()["misses"] == 1

    policy.update(x, step_idx=1, t=0.2, branch="cfg_cat")
    cache_state.set_context(1, 0.2, "cfg_cat", solver_stage="heun_predictor")
    cached(x)
    assert cache_state.summary()["hits"] == 1
