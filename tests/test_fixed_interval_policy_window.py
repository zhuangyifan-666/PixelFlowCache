from __future__ import annotations

import json

from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


def test_outside_active_t_window_no_reuse() -> None:
    policy = FixedIntervalCachePolicy(interval=2, active_t_min=0.1, active_t_max=0.8)
    assert not policy.should_reuse(1, 0.06, "blocks.0", "cond", "euler")
    assert policy.should_refresh(1, 0.06, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(9, 0.8, "blocks.0", "cond", "euler")


def test_inside_active_t_window_interval_behavior_works() -> None:
    policy = FixedIntervalCachePolicy(interval=2, active_t_min=0.1, active_t_max=0.8)
    assert policy.should_refresh(2, 0.2, "blocks.0", "cond", "euler")
    assert policy.should_reuse(3, 0.3, "blocks.0", "cond", "euler")


def test_active_step_min_max() -> None:
    policy = FixedIntervalCachePolicy(interval=2, active_step_min=2, active_step_max=5)
    assert not policy.should_reuse(1, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_reuse(3, 0.3, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(5, 0.5, "blocks.0", "cond", "euler")


def test_to_dict_includes_window_fields() -> None:
    policy = FixedIntervalCachePolicy(interval=2, active_t_min=0.1, active_t_max=0.8, active_step_min=2)
    data = policy.to_dict()
    assert data["active_t_min"] == 0.1
    assert data["active_t_max"] == 0.8
    assert data["active_step_min"] == 2
    assert data["active_step_max"] is None
    assert data["active_window_warmup_refreshes"] == 0
    json.dumps(data)


def test_active_window_warmup_refreshes_delay_first_reuse() -> None:
    policy = FixedIntervalCachePolicy(
        interval=2,
        active_t_min=0.1,
        active_t_max=0.8,
        active_window_warmup_refreshes=1,
    )
    assert policy.should_refresh(1, 0.06, "blocks.0", "cond", "euler")
    assert policy.should_refresh(2, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.0", "cond", "euler")
    assert policy.should_refresh(4, 0.2, "blocks.0", "cond", "euler")
    assert policy.should_reuse(5, 0.25, "blocks.0", "cond", "euler")


def test_active_window_warmup_is_per_module_and_branch() -> None:
    policy = FixedIntervalCachePolicy(
        interval=2,
        active_t_min=0.1,
        active_t_max=0.8,
        active_window_warmup_refreshes=1,
    )
    assert policy.should_refresh(2, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.1", "cond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.0", "uncond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.0", "cond", "euler")
    assert policy.should_reuse(5, 0.25, "blocks.0", "cond", "euler")


def test_active_window_two_refresh_warmup_spans_candidate_reuse_step() -> None:
    policy = FixedIntervalCachePolicy(
        interval=2,
        active_t_min=0.1,
        active_t_max=0.8,
        active_window_warmup_refreshes=2,
    )
    assert policy.should_refresh(2, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.0", "cond", "euler")
    assert policy.should_refresh(4, 0.2, "blocks.0", "cond", "euler")
    assert policy.should_refresh(5, 0.25, "blocks.0", "cond", "euler")
    assert policy.should_refresh(6, 0.3, "blocks.0", "cond", "euler")
    assert policy.should_reuse(7, 0.35, "blocks.0", "cond", "euler")


def test_active_window_warmup_resets_after_leaving_window() -> None:
    policy = FixedIntervalCachePolicy(
        interval=2,
        active_t_min=0.1,
        active_t_max=0.8,
        active_window_warmup_refreshes=1,
    )
    assert policy.should_refresh(2, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(9, 0.8, "blocks.0", "cond", "euler")
    assert policy.should_refresh(2, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(3, 0.15, "blocks.0", "cond", "euler")
