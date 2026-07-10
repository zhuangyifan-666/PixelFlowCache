from __future__ import annotations

import pytest
import torch

from pfc.cache.dicache_policy import DiCachePolicy


def _refresh(
    policy: DiCachePolicy,
    branch: str,
    probe: float,
    full: float,
    step: int,
) -> None:
    zeros = torch.zeros(1, 2, 2)
    policy.record_refresh(
        branch,
        input_feature=zeros,
        probe_feature=torch.full_like(zeros, probe),
        full_feature=torch.full_like(zeros, full),
        step_idx=step,
    )


def test_one_history_falls_back_to_latest_true_residual() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5)
    _refresh(policy, "cond", 1.0, 10.0, 0)
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full((1, 2, 2), 2.0),
    )
    assert not result.dcta_used
    assert result.fallback_used
    assert not result.degenerate
    assert result.gamma_tensor is None
    assert result.reason == "insufficient_history_fallback"
    assert torch.equal(result.residual, torch.full((1, 2, 2), 10.0))
    summary = policy.summary()
    assert summary["dcta_branch_calls"] == 1
    assert summary["dcta_branch_fallback_calls"] == 1
    assert summary["dcta_branch_insufficient_history_fallback_calls"] == 1
    assert summary["gamma_stats"]["count"] == 0


def test_disabled_dcta_falls_back_even_with_two_histories() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5, dcta_enabled=False)
    _refresh(policy, "cond", 1.0, 10.0, 0)
    _refresh(policy, "cond", 3.0, 14.0, 1)
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full((1, 2, 2), 4.0),
    )
    assert result.reason == "dcta_disabled_fallback"
    assert torch.equal(result.residual, torch.full((1, 2, 2), 14.0))


def test_two_histories_apply_first_order_dcta_formula_and_flush_gamma() -> None:
    policy = DiCachePolicy(
        total_blocks=4,
        total_steps=5,
        gamma_min=1.0,
        gamma_max=1.5,
    )
    _refresh(policy, "cond", 1.0, 10.0, 0)
    _refresh(policy, "cond", 3.0, 14.0, 1)
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full((1, 2, 2), 3.5),
    )
    assert result.dcta_used
    assert not result.fallback_used
    assert result.gamma_tensor is not None
    assert result.gamma_tensor.ndim == 0
    assert float(result.gamma_tensor) == pytest.approx(1.25)
    assert torch.allclose(result.residual, torch.full((1, 2, 2), 15.0))
    assert policy.summary()["gamma_stats"]["count"] == 0
    policy.finalize_batch_statistics()
    assert policy.summary()["gamma_stats"]["count"] == 1


@pytest.mark.parametrize(
    ("current", "expected", "counter"),
    [
        (1.5, 1.0, "gamma_clip_low_count"),
        (5.0, 1.5, "gamma_clip_high_count"),
    ],
)
def test_gamma_clamps(current: float, expected: float, counter: str) -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5)
    _refresh(policy, "cond", 1.0, 10.0, 0)
    _refresh(policy, "cond", 3.0, 14.0, 1)
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full((1, 2, 2), current),
    )
    assert result.gamma_tensor is not None
    assert float(result.gamma_tensor) == pytest.approx(expected)
    policy.finalize_batch_statistics()
    assert policy.summary()[counter] == 1


def test_zero_denominator_is_degenerate_fallback_not_dcta() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5)
    _refresh(policy, "cond", 1.0, 10.0, 0)
    _refresh(policy, "cond", 1.0, 14.0, 1)
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full((1, 2, 2), 4.0),
    )
    assert not result.dcta_used
    assert result.fallback_used
    assert result.degenerate
    assert result.gamma_tensor is None
    assert result.reason == "degenerate_probe_trajectory_fallback"
    assert torch.equal(result.residual, torch.full((1, 2, 2), 14.0))
    summary = policy.summary()
    assert summary["dcta_branch_degenerate_fallback_calls"] == 1
    assert summary["gamma_stats"]["count"] == 0


def test_nonfinite_denominator_is_degenerate_fallback() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5)
    _refresh(policy, "cond", 1.0, 10.0, 0)
    _refresh(policy, "cond", float("nan"), 14.0, 1)
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full((1, 2, 2), 4.0),
    )
    assert result.degenerate
    assert not result.dcta_used
    assert torch.equal(result.residual, torch.full((1, 2, 2), 14.0))


def test_branch_histories_are_independent_and_reuse_does_not_pollute_them() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5)
    for branch, offset in (("cond", 0.0), ("uncond", 10.0)):
        _refresh(policy, branch, 1.0, 10.0 + offset, 0)
        _refresh(policy, branch, 3.0, 14.0 + offset, 1)
    cond = policy.approximate_residual(
        "cond", current_probe_residual=torch.full((1, 2, 2), 3.5)
    )
    uncond = policy.approximate_residual(
        "uncond", current_probe_residual=torch.full((1, 2, 2), 5.0)
    )
    assert cond.gamma_tensor is not None and uncond.gamma_tensor is not None
    assert float(cond.gamma_tensor) == pytest.approx(1.25)
    assert float(uncond.gamma_tensor) == pytest.approx(1.5)
    assert len(policy.state.branch_histories["cond"].full_residual_history) == 2
    assert len(policy.state.branch_histories["uncond"].full_residual_history) == 2


def test_refresh_history_is_detached_and_bounded_to_two() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5, clone_history=True)
    for step in range(3):
        _refresh(policy, "cond", float(step + 1), float(step + 10), step)
    history = policy.state.branch_histories["cond"]
    assert list(history.refresh_step_history) == [1, 2]
    assert len(history.full_residual_history) == 2
    assert all(not value.requires_grad for value in history.full_residual_history)
