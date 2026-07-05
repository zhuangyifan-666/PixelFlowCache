from __future__ import annotations

import torch

from pfc.cache.cache_state import CacheEntry
from pfc.cache.safe_map_policy import SafeMapCachePolicy


def _safe_map(max_age: int = 2) -> dict:
    return {
        "policy_name": "SafeMapCachePolicy",
        "model_name": "JiT",
        "solver_stages": ["euler"],
        "branches": ["global"],
        "boundary_groups": {"jit_whole_backbone": ["blocks.0"]},
        "module_to_boundary": {"blocks.0": "jit_whole_backbone"},
        "max_age": max_age,
        "quantile": 0.95,
        "lambda": 0.5,
        "lte_floor": 1e-3,
        "safe": {
            "euler": {
                "global": {
                    "jit_whole_backbone": {
                        "1": {"1": True},
                        "2": {"2": False},
                    }
                }
            }
        },
    }


def test_safe_map_policy_reuses_safe_age_and_rejects_unsafe_age() -> None:
    policy = SafeMapCachePolicy(safe_map=_safe_map())
    entry = CacheEntry(tensor=torch.ones(1), step_idx=0, t=0.0)

    assert policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    assert not policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    summary = policy.summary()["stats"]
    assert summary["safe_reuse"] == 1
    assert summary["unsafe_refresh"] == 1


def test_safe_map_policy_missing_boundary_fallback_and_max_age() -> None:
    policy = SafeMapCachePolicy(safe_map=_safe_map(max_age=1))
    entry = CacheEntry(tensor=torch.ones(1), step_idx=0, t=0.0)

    assert policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="uncond",
        solver_stage="euler",
        entry=entry,
    )
    assert not policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.9",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    assert not policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    summary = policy.summary()["stats"]
    assert summary["safe_reuse"] == 1
    assert summary["over_age_refresh"] == 1


def test_safe_map_policy_missing_entry_refreshes() -> None:
    policy = SafeMapCachePolicy(safe_map=_safe_map())
    assert not policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=None,
    )
    assert policy.summary()["stats"]["missing_entry_refresh"] == 1
