from __future__ import annotations

import json

from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


def test_interval_one_never_reuses() -> None:
    policy = FixedIntervalCachePolicy(interval=1)
    assert policy.should_refresh(0, 0.0, "blocks.0", "cond", "euler")
    assert policy.should_refresh(1, 0.1, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(1, 0.1, "blocks.0", "cond", "euler")


def test_interval_two_reuses_odd_steps_and_refreshes_even_steps() -> None:
    policy = FixedIntervalCachePolicy(interval=2)
    assert policy.should_refresh(0, 0.0, "blocks.0", "cond", "euler")
    assert policy.should_reuse(1, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(2, 0.2, "blocks.0", "cond", "euler")


def test_warmup_steps_disable_reuse() -> None:
    policy = FixedIntervalCachePolicy(interval=2, warmup_steps=2)
    assert policy.should_refresh(1, 0.1, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(1, 0.1, "blocks.0", "cond", "euler")
    assert policy.should_refresh(2, 0.2, "blocks.0", "cond", "euler")
    assert policy.should_reuse(3, 0.3, "blocks.0", "cond", "euler")


def test_cache_modules_filter_modules() -> None:
    policy = FixedIntervalCachePolicy(interval=2, cache_modules={"blocks.1"})
    assert policy.should_cache_module("blocks.1")
    assert not policy.should_cache_module("blocks.0")
    assert policy.should_refresh(1, 0.1, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(1, 0.1, "blocks.0", "cond", "euler")


def test_branch_filtering() -> None:
    policy = FixedIntervalCachePolicy(interval=2, cache_cond=True, cache_uncond=False)
    assert policy.is_branch_enabled("cond")
    assert not policy.is_branch_enabled("uncond")
    assert policy.should_reuse(1, 0.1, "blocks.0", "cond", "euler")
    assert not policy.should_reuse(1, 0.1, "blocks.0", "uncond", "euler")


def test_to_dict_is_json_serializable() -> None:
    policy = FixedIntervalCachePolicy(interval=3, cache_modules={"blocks.2"}, solver_stages={"euler"})
    json.dumps(policy.to_dict())
