from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch

from pfc.cache.dicache_policy import (
    DiCachePolicy,
    aggregate_branch_errors,
    delta_minus,
    is_retention_full_step,
    relative_l1,
)


def _features(value: float) -> dict[str, torch.Tensor]:
    return {"cond": torch.full((1, 2, 2), value), "uncond": torch.full((1, 2, 2), value)}


def _finish_full(policy: DiCachePolicy, decision, h0: torch.Tensor, probes: dict[str, torch.Tensor]) -> None:
    for branch, probe in probes.items():
        policy.record_refresh(
            branch,
            input_feature=h0,
            probe_feature=probe,
            full_feature=probe + 2.0,
            step_idx=decision.step_idx,
        )
    policy.finish_step(decision, input_feature=h0, probe_features=probes)


def test_relative_l1_metrics_are_finite_and_float32() -> None:
    assert relative_l1(torch.ones(4), torch.ones(4)) == 0.0
    assert relative_l1(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0])) == pytest.approx(1.0)
    assert delta_minus(0.4, 0.1) == pytest.approx(0.3)
    assert relative_l1(torch.zeros(4), torch.zeros(4)) == 0.0
    bf16 = relative_l1(torch.tensor([2.0, 4.0], dtype=torch.bfloat16), torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert bf16 == pytest.approx(1.0)
    assert math.isfinite(relative_l1(torch.tensor([float("nan")]), torch.ones(1)))


def test_branch_aggregation_mean_and_max() -> None:
    values = {"cond": 0.2, "uncond": 0.6}
    assert aggregate_branch_errors(values, "mean") == pytest.approx(0.4)
    assert aggregate_branch_errors(values, "max") == pytest.approx(0.6)


def test_retention_boundary_matches_released_flux_compatibility() -> None:
    assert [idx for idx in range(50) if is_retention_full_step(idx, 50, 0.2)] == list(range(11))
    assert is_retention_full_step(0, 50, 0.0)
    assert not is_retention_full_step(1, 50, 0.0)


def test_first_retention_last_and_force_full_reasons() -> None:
    h0 = torch.ones(1, 2, 2)
    policy = DiCachePolicy(total_blocks=4, total_steps=5, ret_ratio=0.2)
    first = policy.decide(step_idx=0, input_feature=h0, probe_features=_features(2.0))
    assert (first.decision, first.reason) == ("full", "first_step")
    _finish_full(policy, first, h0, _features(2.0))
    retention = policy.decide(step_idx=1, input_feature=h0, probe_features=_features(2.0))
    assert retention.reason == "retention_full"
    _finish_full(policy, retention, h0, _features(2.0))
    last = policy.decide(step_idx=4, input_feature=h0, probe_features=_features(2.0))
    assert last.reason == "last_step"

    forced = DiCachePolicy(total_blocks=4, total_steps=3, ret_ratio=0.0, force_full=True)
    for step in range(3):
        assert forced.decide(step_idx=step, input_feature=h0, probe_features=_features(2.0)).reason == "force_full"


def test_accumulated_threshold_and_reset() -> None:
    h0 = torch.ones(1, 2, 2)
    policy = DiCachePolicy(
        total_blocks=4,
        total_steps=5,
        ret_ratio=0.0,
        force_last_step_full=False,
        reuse_threshold=0.5,
    )
    probes = _features(2.0)
    first = policy.decide(step_idx=0, input_feature=h0, probe_features=probes)
    _finish_full(policy, first, h0, probes)
    reuse = policy.decide(step_idx=1, input_feature=h0, probe_features=_features(2.2))
    assert reuse.decision == "reuse"
    assert reuse.accumulated_error_after == pytest.approx(0.1)
    policy.finish_step(reuse, input_feature=h0, probe_features=_features(2.2))
    refresh = policy.decide(step_idx=2, input_feature=h0, probe_features=_features(3.2))
    assert (refresh.decision, refresh.reason) == ("full", "adaptive_threshold")
    assert refresh.accumulated_error_after == 0.0


def test_released_flux_variant_uses_strict_lt_at_threshold() -> None:
    h0 = torch.ones(1, 2, 2)
    policy = DiCachePolicy(
        total_blocks=4,
        total_steps=3,
        ret_ratio=0.0,
        force_last_step_full=False,
        reuse_threshold=0.5,
    )
    probes = _features(2.0)
    first = policy.decide(step_idx=0, input_feature=h0, probe_features=probes)
    _finish_full(policy, first, h0, probes)
    policy.state.accumulated_error = 0.5
    assert policy.decide(
        step_idx=1, input_feature=h0, probe_features=probes
    ).decision == "full"


def test_insufficient_history_is_conservative_and_clear_batch_clears_state() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=4, ret_ratio=0.0, force_last_step_full=False)
    h0 = torch.ones(1, 2, 2)
    policy.state.previous_input_feature = h0
    for history in policy.state.branch_histories.values():
        history.previous_probe_feature = torch.ones(1, 2, 2)
    decision = policy.decide(step_idx=1, input_feature=h0, probe_features=_features(1.0))
    assert (decision.decision, decision.reason) == ("full", "insufficient_history")
    policy.clear_batch()
    assert policy.state.previous_input_feature is None
    assert policy.state.accumulated_error == 0.0


def test_bounded_samples_and_running_moments_cover_all_values() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=6, ret_ratio=0.0, force_last_step_full=False, max_error_samples=2)
    h0 = torch.ones(1, 2, 2)
    probes = _features(2.0)
    first = policy.decide(step_idx=0, input_feature=h0, probe_features=probes)
    _finish_full(policy, first, h0, probes)
    for step, value in enumerate((2.1, 2.3, 2.6), start=1):
        decision = policy.decide(step_idx=step, input_feature=h0, probe_features=_features(value))
        policy.finish_step(decision, input_feature=h0, probe_features=_features(value))
    summary = policy.summary()
    stats = summary["decision_branch_error_stats"]
    assert stats["count"] == 6
    assert len(stats["bounded_samples"]) == 2
    assert stats["mean"] > 0.0


def test_observed_and_decision_error_stats_have_distinct_populations() -> None:
    policy = DiCachePolicy(
        total_blocks=4,
        total_steps=5,
        ret_ratio=0.2,
        force_last_step_full=False,
        reuse_threshold=10.0,
    )
    h0 = torch.ones(1, 2, 2)
    first_probes = _features(2.0)
    first = policy.decide(step_idx=0, input_feature=h0, probe_features=first_probes)
    _finish_full(policy, first, h0, first_probes)

    retention_probes = _features(2.2)
    retention = policy.decide(
        step_idx=1,
        input_feature=h0,
        probe_features=retention_probes,
    )
    assert retention.reason == "retention_full"
    _finish_full(policy, retention, h0, retention_probes)

    adaptive_probes = _features(2.4)
    adaptive = policy.decide(
        step_idx=2,
        input_feature=h0,
        probe_features=adaptive_probes,
    )
    assert adaptive.reason == "adaptive_threshold"
    summary = policy.summary()
    assert summary["observed_delta_x_stats"]["count"] == 2
    assert summary["decision_delta_x_stats"]["count"] == 1
    assert summary["observed_branch_error_stats"]["count"] == 4
    assert summary["decision_branch_error_stats"]["count"] == 2
    assert summary["observed_branch_error_stats"]["std"] >= 0.0


def test_debug_jsonl_records_shared_decision_fields(tmp_path: Path) -> None:
    path = tmp_path / "dicache.jsonl"
    policy = DiCachePolicy(total_blocks=4, total_steps=2, ret_ratio=0.0, debug_jsonl=path)
    h0 = torch.ones(1, 2, 2)
    probes = _features(2.0)
    decision = policy.decide(step_idx=0, input_feature=h0, probe_features=probes)
    _finish_full(policy, decision, h0, probes)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "dicache_step"
    assert payload["decision"] == "full"
    assert payload["reason"] == "first_step"
    assert payload["history_size_cond"] == 1
    assert payload["history_size_uncond"] == 1
