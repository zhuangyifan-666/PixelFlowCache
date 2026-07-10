from __future__ import annotations

import pytest
import torch

from pfc.cache.cache_state import CacheEntry
from pfc.cache.speca_policy import SpeCaCachePolicy, resolve_verifier_module
from pfc.cache.taylorseer_policy import TaylorSeerCachePolicy


MODULES = {"blocks.0", "blocks.1"}
BRANCHES = ("cond", "uncond")


def _entry(value: float = 0.0, step_idx: int = 0) -> CacheEntry:
    return CacheEntry(torch.full((1, 1), value), step_idx=step_idx, t=float(step_idx))


def _policy(**kwargs) -> SpeCaCachePolicy:
    return SpeCaCachePolicy(cache_modules=MODULES, total_steps=10, **kwargs)


def _append_full_history(policy: SpeCaCachePolicy, step_idx: int) -> None:
    for module_idx, module_name in enumerate(sorted(MODULES)):
        for branch_idx, branch in enumerate(BRANCHES):
            policy.on_refresh_committed(
                step_idx=step_idx,
                t=float(step_idx),
                module_name=module_name,
                cfg_branch=branch,
                solver_stage="euler",
                entry=None,
                tensor=torch.tensor([[float(step_idx + module_idx + branch_idx)]]),
            )


def _decide(policy: SpeCaCachePolicy, step_idx: int, branch: str = "cond"):
    policy.should_reuse_entry(
        step_idx=step_idx,
        t=step_idx / 10.0,
        module_name="blocks.0",
        cfg_branch=branch,
        solver_stage="euler",
        entry=_entry(step_idx=max(step_idx - 1, 0)),
    )
    decision = policy.current_step_decision()
    assert decision is not None
    return decision


def test_speca_is_a_taylorseer_policy_and_resolves_last_numeric_block() -> None:
    policy = _policy()
    assert isinstance(policy, TaylorSeerCachePolicy)
    assert policy.verifier_module == "blocks.1"
    assert policy.verifier_module_requested == "auto"
    assert policy.verifier_module_resolved == "blocks.1"
    assert resolve_verifier_module(["blocks.11", "blocks.2", "blocks.7"], "auto") == "blocks.11"
    assert resolve_verifier_module(["net.blocks.0", "net.blocks.3"], "net.blocks.0") == "blocks.0"
    with pytest.raises(ValueError, match="not in selected modules"):
        resolve_verifier_module(["blocks.0"], "blocks.1")
    explicit = _policy(verifier_module="blocks.0")
    assert explicit.verifier_module_requested == "blocks.0"
    assert explicit.verifier_module_resolved == "blocks.0"


def test_speca_threshold_schedule_is_noise_to_image_nonincreasing() -> None:
    policy = _policy(base_threshold=0.1, decay_rate=0.01, min_threshold=0.01)
    thresholds = [policy.threshold_for_step(step) for step in range(10)]
    assert thresholds[0] == pytest.approx(max(0.1 * 0.01 ** (1 / 10), 0.01))
    assert thresholds[-1] == pytest.approx(max(0.1 * 0.01, 0.01))
    assert all(value >= 0.01 for value in thresholds)
    assert thresholds == sorted(thresholds, reverse=True)
    with pytest.raises(ValueError, match="decay_rate"):
        _policy(decay_rate=0.0)
    with pytest.raises(ValueError, match="base_threshold"):
        _policy(base_threshold=0.0)
    with pytest.raises(ValueError, match="min_threshold"):
        _policy(min_threshold=0.0)
    with pytest.raises(ValueError, match="must not exceed"):
        _policy(base_threshold=0.1, min_threshold=0.2)
    with pytest.raises(ValueError, match="total_steps"):
        SpeCaCachePolicy(cache_modules=MODULES, total_steps=0)


def test_speca_schedule_initial_history_minimum_and_shared_step_decision() -> None:
    policy = _policy(first_full_steps=2, min_history=2)
    step0 = _decide(policy, 0)
    assert step0.mode == "full" and step0.reason == "initial_full_steps"
    _append_full_history(policy, 0)

    step1 = _decide(policy, 1)
    assert step1.mode == "full" and step1.reason == "initial_full_steps"
    _append_full_history(policy, 1)

    step2 = _decide(policy, 2)
    assert step2.mode == "speculative"
    assert step2.consecutive_speculative_steps == 1
    assert not step2.verification_enabled
    same_step = _decide(policy, 2, branch="uncond")
    assert same_step is step2

    step3 = _decide(policy, 3)
    assert step3.mode == "speculative"
    assert step3.consecutive_speculative_steps == 2
    assert not step3.verification_enabled

    step4 = _decide(policy, 4)
    assert step4.mode == "speculative"
    assert step4.consecutive_speculative_steps == 3
    assert step4.verification_enabled


@pytest.mark.parametrize(
    ("aggregation", "cond_error", "uncond_error", "expected"),
    [("mean", 0.02, 0.04, 0.03), ("max", 0.02, 0.04, 0.04)],
)
def test_speca_branch_errors_are_aggregated_for_the_next_step(
    aggregation: str,
    cond_error: float,
    uncond_error: float,
    expected: float,
) -> None:
    policy = _policy(
        first_full_steps=2,
        min_history=2,
        branch_aggregation=aggregation,
        base_threshold=0.1,
        decay_rate=1.0,
    )
    _decide(policy, 0)
    _append_full_history(policy, 0)
    _decide(policy, 1)
    _append_full_history(policy, 1)
    _decide(policy, 2)
    assert not _decide(policy, 3).verification_enabled
    decision4 = _decide(policy, 4)
    assert decision4.verification_enabled
    policy.record_verification_error(step_idx=4, cfg_branch="cond", solver_stage="euler", error=cond_error)
    policy.record_verification_error(step_idx=4, cfg_branch="uncond", solver_stage="euler", error=uncond_error)

    decision5 = _decide(policy, 5)
    assert decision5.mode == "speculative"
    assert decision5.reason == "verification_accept"
    assert decision5.previous_verification_error == pytest.approx(expected)


def test_speca_rejects_only_after_min_run_and_enforces_max_run() -> None:
    policy = _policy(
        first_full_steps=2,
        min_history=2,
        min_forecast_steps=2,
        max_forecast_steps=5,
        base_threshold=0.1,
        decay_rate=1.0,
    )
    for step in range(2):
        _decide(policy, step)
        _append_full_history(policy, step)
    assert _decide(policy, 2).mode == "speculative"
    assert not _decide(policy, 3).verification_enabled
    decision4 = _decide(policy, 4)
    assert decision4.verification_enabled
    policy.record_verification_error(step_idx=4, cfg_branch="cond", solver_stage="euler", error=0.2)
    policy.record_verification_error(step_idx=4, cfg_branch="uncond", solver_stage="euler", error=0.2)
    rejected = _decide(policy, 5)
    assert rejected.mode == "full"
    assert rejected.reason == "verification_reject"
    assert rejected.consecutive_speculative_steps == 0

    max_policy = _policy(
        first_full_steps=2,
        min_history=2,
        min_forecast_steps=2,
        max_forecast_steps=5,
        base_threshold=0.1,
        decay_rate=1.0,
    )
    for step in range(2):
        _decide(max_policy, step)
        _append_full_history(max_policy, step)
    for step in range(2, 7):
        decision = _decide(max_policy, step)
        assert decision.mode == "speculative"
        assert decision.consecutive_speculative_steps == step - 1
        if decision.verification_enabled:
            for branch in BRANCHES:
                max_policy.record_verification_error(
                    step_idx=step,
                    cfg_branch=branch,
                    solver_stage="euler",
                    error=0.01,
                )
    maximum = _decide(max_policy, 7)
    assert maximum.mode == "full"
    assert maximum.reason == "max_forecast_steps"


def test_speca_missing_verification_forces_next_step_full_and_clear_batch_resets_state() -> None:
    policy = _policy(first_full_steps=2, min_history=2)
    for step in range(2):
        _decide(policy, step)
        _append_full_history(policy, step)
    _decide(policy, 2)
    assert not _decide(policy, 3).verification_enabled
    assert _decide(policy, 4).verification_enabled
    missing = _decide(policy, 5)
    assert missing.mode == "full"
    assert missing.reason == "missing_verification_error"

    policy.clear_batch()
    assert policy.current_step_decision() is None
    after_clear = _decide(policy, 0)
    assert after_clear.mode == "full"
    assert not policy._history


def test_speca_single_branch_error_is_used_and_missing_branch_is_counted() -> None:
    policy = _policy(
        first_full_steps=2,
        min_history=2,
        base_threshold=0.1,
        decay_rate=1.0,
    )
    for step in range(2):
        _decide(policy, step)
        _append_full_history(policy, step)
    _decide(policy, 2)
    assert not _decide(policy, 3).verification_enabled
    assert _decide(policy, 4).verification_enabled
    policy.record_verification_error(
        step_idx=4,
        cfg_branch="cond",
        solver_stage="euler",
        error=0.02,
    )
    next_decision = _decide(policy, 5)
    assert next_decision.previous_verification_error == pytest.approx(0.02)
    assert next_decision.mode == "speculative"
    assert policy.summary()["missing_branch_verification"] == 1


def test_speca_history_readiness_requires_every_module_and_branch() -> None:
    policy = _policy(first_full_steps=0, min_history=2)
    for step in range(2):
        for module_name in MODULES:
            policy.on_refresh_committed(
                step_idx=step,
                t=float(step),
                module_name=module_name,
                cfg_branch="cond",
                solver_stage="euler",
                entry=None,
                tensor=torch.tensor([[float(step)]]),
            )
    assert not policy.histories_ready(solver_stage="euler", batch_signature="b:1")
    decision = _decide(policy, 2)
    assert decision.mode == "full"
    assert decision.reason == "insufficient_history"


def test_speca_summary_exposes_required_schedule_and_overhead_fields() -> None:
    policy = _policy()
    _decide(policy, 0)
    summary = policy.summary()
    assert summary["policy_name"] == "SpeCaCachePolicy"
    assert summary["baseline_name"] == "adapted SpeCa-style"
    assert summary["total_steps_seen"] == 1
    assert summary["verifier_module"] == "blocks.1"
    assert summary["verification_overhead_stats"]["estimated_verifier_block_fraction"] == 0.5
    assert summary["verification_acceptance_rate"] == 0.0
    assert summary["speculative_step_ratio"] == 0.0
    assert summary["timing_semantics"] == "host_dispatch_only"


def test_speca_acceptance_counters_and_step_ratio_use_consistent_units() -> None:
    policy = _policy(
        first_full_steps=2,
        min_history=2,
        base_threshold=0.1,
        decay_rate=1.0,
    )
    for step in range(2):
        _decide(policy, step)
        _append_full_history(policy, step)
    _decide(policy, 2)
    _decide(policy, 3)
    _decide(policy, 4)
    for branch in BRANCHES:
        policy.record_verification_error(
            step_idx=4,
            cfg_branch=branch,
            solver_stage="euler",
            error=0.01,
        )
    accepted = _decide(policy, 5)
    assert accepted.reason == "verification_accept"
    for branch in BRANCHES:
        policy.record_verification_error(
            step_idx=5,
            cfg_branch=branch,
            solver_stage="euler",
            error=0.2,
        )
    rejected = _decide(policy, 6)
    assert rejected.reason == "verification_reject"
    summary = policy.summary()
    assert summary["verification_accept_decisions"] == 1
    assert summary["verification_reject_decisions"] == 1
    assert summary["verification_acceptance_rate"] == pytest.approx(0.5)
    assert summary["speculative_step_ratio"] == pytest.approx(4 / 7)


def test_speca_verification_error_samples_are_bounded_but_moments_use_all_values() -> None:
    policy = _policy(
        first_full_steps=2,
        min_history=2,
        max_verification_error_samples=3,
    )
    for step in range(2):
        _decide(policy, step)
        _append_full_history(policy, step)
    _decide(policy, 2)
    _decide(policy, 3)
    assert _decide(policy, 4).verification_enabled
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        policy.record_verification_error(
            step_idx=4,
            cfg_branch="cond",
            solver_stage="euler",
            error=value,
        )
    errors = policy.summary()["verification_errors"]
    assert errors["count"] == 5
    assert errors["mean"] == pytest.approx(3.0)
    assert errors["sample_count"] == 3
    assert errors["p50"] == pytest.approx(4.0)
    assert errors["quantiles_approximate"] is True
    assert errors["max_samples"] == 3
    assert "values" not in errors
