from __future__ import annotations

import json

from pfc.core.cache_scheduler import FixedWindowScheduler


def test_fixed_window_scheduler_refresh_reuse_pattern() -> None:
    scheduler = FixedWindowScheduler(interval=2, active_t_min=0.2, active_t_max=1.0)
    scheduler.set_context(step_idx=0, t=0.1, branch="cond")
    assert scheduler.should_refresh("blocks.0")
    assert not scheduler.should_reuse("blocks.0")

    scheduler.set_context(step_idx=0, t=0.2, branch="cond")
    assert scheduler.should_refresh("blocks.0")

    scheduler.set_context(step_idx=1, t=0.3, branch="cond")
    assert scheduler.should_reuse("blocks.0")

    scheduler.set_context(step_idx=2, t=0.4, branch="cond")
    assert scheduler.should_refresh("blocks.0")


def test_fixed_window_scheduler_to_dict_and_policy_adapter() -> None:
    scheduler = FixedWindowScheduler(interval=3, cache_cond=True, cache_uncond=False, solver_stages={"euler"})
    policy = scheduler.to_policy_adapter(cache_modules={"a"})
    assert policy.interval == 3
    assert policy.cache_modules == {"a"}
    assert policy.cache_cond
    assert not policy.cache_uncond
    json.dumps(scheduler.to_dict())
    json.dumps(scheduler.summary())
