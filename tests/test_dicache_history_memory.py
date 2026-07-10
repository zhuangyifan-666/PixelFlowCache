from __future__ import annotations

import torch

from pfc.cache.dicache_policy import DiCachePolicy
from pfc.cache.dicache_state import DiCacheBranchHistory, compact_history_tensor


def test_unused_probe_residual_state_is_removed() -> None:
    assert not hasattr(
        DiCacheBranchHistory(),
        "previous_" + "probe_" + "residual",
    )


def test_sequence_tail_view_is_compacted_to_owned_storage() -> None:
    base = torch.arange(48, dtype=torch.float32).reshape(1, 12, 4)
    tail = base[:, -3:]
    compact = compact_history_tensor(tail)
    assert compact._base is None
    assert compact.storage_offset() == 0
    assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()
    assert torch.equal(compact, tail)


def test_history_memory_deduplicates_shared_storage_and_tracks_peak() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=3)
    base = torch.zeros(1, 8, 4)
    policy.state.previous_input_feature = base
    policy.state.branch_histories["cond"].previous_probe_feature = base[:, :4]
    policy.state.branch_histories["uncond"].previous_probe_feature = base[:, 4:]
    summary = policy.summary()
    assert summary["history_tensor_count"] == 3
    assert summary["history_unique_storage_count"] == 1
    assert summary["current_history_storage_bytes"] == base.untyped_storage().nbytes()
    peak = summary["peak_history_storage_bytes"]

    policy.clear_batch()
    cleared = policy.summary()
    assert cleared["current_history_storage_bytes"] == 0
    assert cleared["history_tensor_count"] == 0
    assert cleared["history_unique_storage_count"] == 0
    assert cleared["peak_history_storage_bytes"] == peak


def test_finish_step_compacts_probe_views() -> None:
    policy = DiCachePolicy(total_blocks=4, total_steps=2)
    h0 = torch.zeros(1, 3, 4)
    context_backing = torch.ones(1, 9, 4)
    probes = {
        "cond": context_backing[:, -3:],
        "uncond": context_backing[:, -3:],
    }
    decision = policy.decide(step_idx=0, input_feature=h0, probe_features=probes)
    policy.finish_step(decision, input_feature=h0, probe_features=probes)
    for history in policy.state.branch_histories.values():
        assert history.previous_probe_feature is not None
        assert history.previous_probe_feature._base is None
        assert history.previous_probe_feature.untyped_storage().nbytes() == (
            history.previous_probe_feature.numel()
            * history.previous_probe_feature.element_size()
        )
