from __future__ import annotations

import json

import torch

from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, relative_l1


def test_relative_l1_identical_tensors_returns_zero() -> None:
    x = torch.ones(2, 3, 4, 4)
    assert float(relative_l1(x, x).item()) == 0.0


def test_raw_policy_refreshes_first_step() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=0.5)
    decision = policy.update(torch.ones(1, 1, 4, 4), step_idx=0, t=0.0)
    assert decision.should_refresh is True
    assert decision.reason == "first_observation"


def test_raw_policy_reuses_below_threshold() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=1.0)
    policy.update(torch.ones(1, 1, 4, 4), step_idx=0, t=0.0)
    decision = policy.update(torch.ones(1, 1, 4, 4) * 1.01, step_idx=1, t=0.1)
    assert decision.should_refresh is False
    assert policy.should_reuse()


def test_raw_policy_refreshes_above_threshold() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=0.05)
    policy.update(torch.ones(1, 1, 4, 4), step_idx=0, t=0.0)
    decision = policy.update(torch.ones(1, 1, 4, 4) * 2.0, step_idx=1, t=0.1)
    assert decision.should_refresh is True
    assert decision.reason == "threshold_exceeded"


def test_raw_policy_summary_json_serializable() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=0.5)
    policy.update(torch.ones(1, 1, 4, 4), step_idx=0, t=0.0)
    policy.update(torch.ones(1, 1, 4, 4) * 1.1, step_idx=1, t=0.1)
    json.dumps(policy.summary())
    assert policy.summary()["stats"]["total_steps"] == 2
