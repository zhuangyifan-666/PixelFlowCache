from __future__ import annotations

import inspect

import pytest
import torch

from pfc.cache.dicache_policy import (
    DiCachePolicy,
    dcta_residual_tensor,
    relative_l1_tensor,
)


def _features(value: float) -> dict[str, torch.Tensor]:
    return {
        "cond": torch.full((1, 2, 2), value),
        "uncond": torch.full((1, 2, 2), value),
    }


def _record_full(
    policy: DiCachePolicy,
    step: int,
    h0: torch.Tensor,
    probes: dict[str, torch.Tensor],
) -> None:
    for branch, probe in probes.items():
        policy.record_refresh(
            branch,
            input_feature=h0,
            probe_feature=probe,
            full_feature=probe + 2.0,
            step_idx=step,
        )


def test_relative_l1_tensor_is_device_scalar_with_no_host_conversion() -> None:
    value = relative_l1_tensor(torch.ones(4), torch.ones(4), 1e-10)
    assert value.ndim == 0
    assert value.dtype == torch.float32
    source = inspect.getsource(relative_l1_tensor)
    for forbidden in (".item(", ".cpu(", ".tolist("):
        assert forbidden not in source


def test_dcta_tensor_core_has_no_host_conversion() -> None:
    source = inspect.getsource(dcta_residual_tensor)
    assert ".item(" not in source
    assert ".cpu(" not in source


def test_each_metric_ready_step_adds_exactly_one_decision_sync() -> None:
    policy = DiCachePolicy(
        total_blocks=4,
        total_steps=4,
        ret_ratio=0.0,
        force_last_step_full=False,
        reuse_threshold=100.0,
    )
    h0 = torch.ones(1, 2, 2)
    first_probes = _features(2.0)
    first = policy.decide(step_idx=0, input_feature=h0, probe_features=first_probes)
    assert policy.summary()["decision_device_to_host_syncs"] == 0
    _record_full(policy, 0, h0, first_probes)
    policy.finish_step(first, input_feature=h0, probe_features=first_probes)

    for step, value in ((1, 2.1), (2, 2.2)):
        probes = _features(value)
        decision = policy.decide(
            step_idx=step,
            input_feature=h0,
            probe_features=probes,
        )
        assert decision.reason == "adaptive_threshold"
        assert policy.summary()["decision_device_to_host_syncs"] == step
        policy.finish_step(decision, input_feature=h0, probe_features=probes)
    assert policy.summary()["decision_syncs_per_step"] == pytest.approx(2 / 3)


def test_gamma_stats_are_buffered_until_one_batch_finalize() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5)
    zeros = torch.zeros(1, 2, 2)
    for step, (probe, full) in enumerate(((1.0, 10.0), (3.0, 14.0))):
        policy.record_refresh(
            "cond",
            input_feature=zeros,
            probe_feature=torch.full_like(zeros, probe),
            full_feature=torch.full_like(zeros, full),
            step_idx=step,
        )
    result = policy.approximate_residual(
        "cond",
        current_probe_residual=torch.full_like(zeros, 3.5),
        degenerate=False,
    )
    assert result.gamma_tensor is not None
    assert result.gamma_tensor.ndim == 0
    assert len(policy.pending_gamma_tensors) == 1
    assert policy.summary()["gamma_stats"]["count"] == 0
    assert not policy.summary()["debug_sync_overhead_enabled"]
    policy.finalize_batch_statistics()
    gamma = policy.summary()["gamma_stats"]
    assert gamma["count"] == 1
    assert gamma["mean"] == pytest.approx(1.25)
    assert len(policy.pending_gamma_tensors) == 0
