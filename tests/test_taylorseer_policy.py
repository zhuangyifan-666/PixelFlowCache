from __future__ import annotations

import torch

from pfc.cache.cache_state import CacheEntry
from pfc.cache.taylorseer_policy import TaylorSeerCachePolicy


def _entry(value: float = 0.0, step_idx: int = 0, batch: int = 1) -> CacheEntry:
    return CacheEntry(tensor=torch.full((batch, 1), value), step_idx=step_idx, t=float(step_idx))


def _append(policy: TaylorSeerCachePolicy, step: int, value: float, branch: str = "cond") -> None:
    policy.on_refresh_committed(
        step_idx=step,
        t=float(step),
        module_name="blocks.0",
        cfg_branch=branch,
        solver_stage="euler",
        entry=None,
        tensor=torch.full((1, 1), value),
    )


def test_taylorseer_policy_refresh_first_and_insufficient_history() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=4, max_order=4)
    assert not policy.should_reuse_entry(
        step_idx=0,
        t=0.0,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(step_idx=-1),
    )
    _append(policy, 0, 1.0)
    assert not policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(step_idx=0),
    )
    stats = policy.summary()["stats"]
    assert stats["scheduled_refresh"] == 1
    assert stats["insufficient_history_refresh"] == 1


def test_taylorseer_policy_forecasts_when_history_is_ready_and_refreshes_on_interval() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=4, max_order=4)
    _append(policy, 0, 1.0)
    _append(policy, 1, 2.0)

    assert policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(step_idx=1),
    )
    assert not policy.should_reuse_entry(
        step_idx=4,
        t=0.4,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(step_idx=1),
    )
    stats = policy.summary()["stats"]
    assert stats["forecast_decisions"] == 1
    assert stats["interval_refresh"] == 1


def test_taylorseer_lagrange_forecast_linear_sequence() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=10, max_order=1, min_history=2)
    _append(policy, 0, 0.0)
    _append(policy, 1, 2.0)
    entry = _entry(value=2.0, step_idx=1)

    assert policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    predicted = policy.make_reuse_tensor(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
        current_input=torch.zeros(1, 1),
    )
    assert predicted is not None
    assert torch.allclose(predicted, torch.full((1, 1), 4.0))


def test_taylorseer_lagrange_forecast_second_order_sequence() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=10, max_order=2, min_history=3)
    _append(policy, 0, 0.0)
    _append(policy, 1, 1.0)
    _append(policy, 2, 4.0)
    entry = _entry(value=4.0, step_idx=2)

    assert policy.should_reuse_entry(
        step_idx=3,
        t=0.3,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    predicted = policy.make_reuse_tensor(
        step_idx=3,
        t=0.3,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
        current_input=torch.zeros(1, 1),
    )
    assert predicted is not None
    assert torch.allclose(predicted, torch.full((1, 1), 9.0))


def test_taylorseer_branch_history_is_independent() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=10, max_order=1)
    _append(policy, 0, 1.0, branch="cond")
    _append(policy, 1, 2.0, branch="cond")
    _append(policy, 0, 10.0, branch="uncond")

    assert policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(step_idx=1),
    )
    assert not policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="uncond",
        solver_stage="euler",
        entry=_entry(step_idx=0),
    )


def test_taylorseer_clear_batch_clears_history() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=10, max_order=1)
    _append(policy, 0, 1.0)
    _append(policy, 1, 2.0)
    policy.clear_batch()

    assert not policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(step_idx=1),
    )


def test_taylorseer_summary_contains_forecast_stats() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=10, max_order=1)
    _append(policy, 0, 1.0)
    _append(policy, 1, 2.0)
    entry = _entry(value=2.0, step_idx=1)
    assert policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    predicted = policy.make_reuse_tensor(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
        current_input=torch.zeros(1, 1),
    )
    policy.on_reuse_committed(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
        tensor=predicted,
    )

    summary = policy.summary()
    assert summary["config"]["interval"] == 10
    assert summary["config"]["max_order"] == 1
    assert summary["stats"]["forecast_committed"] == 1
    assert summary["stats"]["mean_effective_order"] == 1.0
