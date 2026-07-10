from __future__ import annotations

import torch

from pfc.cache.cache_state import CacheEntry
from pfc.cache.dicache_policy import DiCachePolicy
from pfc.cache.safe_map_policy import SafeMapCachePolicy
from pfc.cache.speca_policy import SpeCaCachePolicy
from pfc.cache.taylorseer_policy import TaylorSeerCachePolicy


def _safe_policy() -> SafeMapCachePolicy:
    return SafeMapCachePolicy(
        safe_map={
            "model_name": "JiT",
            "solver_stages": ["euler"],
            "branches": ["global"],
            "boundary_groups": {"backbone": ["blocks.0"]},
            "module_to_boundary": {"blocks.0": "backbone"},
            "max_age": 1,
            "safe": {"euler": {"global": {"backbone": {"1": {"1": True}}}}},
        }
    )


def _entry(step_idx: int = 0) -> CacheEntry:
    return CacheEntry(torch.ones(1, 1), step_idx=step_idx, t=float(step_idx))


def test_safe_map_warmup_stats_reset_without_config_change() -> None:
    policy = _safe_policy()
    config = policy.to_dict()
    assert policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(0),
    )
    policy.mark_reuse_committed(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(0),
    )

    policy.reset_runtime_state()
    policy.reset_stats()

    assert policy.to_dict() == config
    stats = policy.summary()["stats"]
    assert stats["total_managed_calls"] == 0
    assert stats["safe_reuse_committed"] == 0
    assert stats["refresh_committed"] == 0
    assert stats["by_reason"] == {}
    assert policy.should_reuse_entry(
        step_idx=1,
        t=0.1,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(0),
    )
    assert policy.summary()["stats"]["safe_reuse_decisions"] == 1


def test_taylorseer_warmup_history_and_stats_reset() -> None:
    policy = TaylorSeerCachePolicy(cache_modules={"blocks.0"}, interval=10, max_order=1)
    config = policy.to_dict()
    for step in (0, 1):
        policy.on_refresh_committed(
            step_idx=step,
            t=float(step),
            module_name="blocks.0",
            cfg_branch="cond",
            solver_stage="euler",
            entry=None,
            tensor=torch.full((1, 1), float(step)),
        )
    policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(1),
    )

    policy.reset_runtime_state()
    policy.reset_stats()

    assert policy.to_dict() == config
    assert not policy._history
    assert not policy._pending_forecasts
    assert all(value == 0 for value in policy._stats.values())
    policy.on_refresh_committed(
        step_idx=0,
        t=0.0,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=None,
        tensor=torch.zeros(1, 1),
    )
    assert policy.summary()["stats"]["history_appends"] == 1


def test_speca_warmup_verifier_errors_runs_and_timings_reset() -> None:
    policy = SpeCaCachePolicy(cache_modules={"blocks.0", "blocks.1"}, total_steps=10)
    config = policy.to_dict()
    policy.should_reuse_entry(
        step_idx=0,
        t=0.0,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(),
    )
    policy._verification_error_count = 2
    policy._verification_error_sum = 0.3
    policy._verification_error_sum_sq = 0.05
    policy._verification_error_samples.extend([0.1, 0.2])
    policy._all_completed_run_lengths.append(3)
    policy._speca_stats["verifier_fresh_calls"] = 2
    policy._verification_host_dispatch_time_sec = 1.0
    policy._forecast_host_dispatch_time_sec = 2.0
    policy._full_compute_host_dispatch_time_sec = 3.0

    policy.reset_runtime_state()
    policy.reset_stats()

    assert policy.to_dict() == config
    assert policy.current_step_decision() is None
    summary = policy.summary()
    assert summary["total_steps_seen"] == 0
    assert summary["verifier_fresh_calls"] == 0
    assert summary["verification_errors"]["count"] == 0
    assert summary["completed_speculative_runs"] == 0
    assert summary["verification_overhead_stats"]["verification_host_dispatch_time_sec"] == 0.0
    policy.should_reuse_entry(
        step_idx=0,
        t=0.0,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=_entry(),
    )
    assert policy.summary()["total_steps_seen"] == 1


def test_dicache_warmup_dcta_error_gamma_sync_and_timing_reset() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=5, ret_ratio=0.0)
    config = policy.summary()["config"]
    policy._counts["total_steps_seen"] = 4
    policy._counts["dcta_steps"] = 2
    policy._counts["decision_device_to_host_syncs"] = 3
    policy._observed_stats["delta_y"].add(0.25)
    policy._decision_stats["branch_error"].add(0.5)
    policy._gamma_stats.add(1.2)
    policy._timings["probe_host_dispatch_time_sec"] = 1.0
    policy.state.accumulated_error = 0.4
    policy.state.previous_input_feature = torch.ones(1)

    policy.reset_runtime_state()
    policy.reset_stats()

    summary = policy.summary()
    assert summary["config"] == config
    assert summary["total_steps_seen"] == 0
    assert summary["dcta_steps"] == 0
    assert summary["decision_device_to_host_syncs"] == 0
    assert summary["observed_delta_y_stats"]["count"] == 0
    assert summary["decision_branch_error_stats"]["count"] == 0
    assert summary["gamma_stats"]["count"] == 0
    assert summary["probe_host_dispatch_time_sec"] == 0.0
    assert summary["accumulated_error_current"] == 0.0
    assert summary["current_history_storage_bytes"] == 0
    policy.decide(
        step_idx=0,
        input_feature=torch.ones(1, 2, 2),
        probe_features={"cond": torch.ones(1, 2, 2), "uncond": torch.ones(1, 2, 2)},
    )
    assert policy.summary()["total_steps_seen"] == 1
