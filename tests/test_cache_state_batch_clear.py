from __future__ import annotations

import torch

from pfc.cache.cache_state import RuntimeCacheState


def test_clear_entries_keeps_accumulated_stats() -> None:
    state = RuntimeCacheState()
    state.set_context(0, 0.0, "cond")
    key = state.make_key("blocks.0", "b:1")
    state.put(key, torch.ones(1, 2))
    state.mark_hit("blocks.0")

    state.clear_entries()

    assert state.get(key) is None
    assert state.summary()["hits"] == 1
    assert state.summary()["total_calls"] == 1


def test_clear_removes_entries_and_keeps_context_valid() -> None:
    state = RuntimeCacheState()
    state.set_context(3, 0.3, "uncond")
    key = state.make_key("blocks.1", "b:1")
    state.put(key, torch.zeros(1, 2))

    state.clear()

    assert state.get(key) is None
    summary = state.summary()
    assert summary["current_step_idx"] == 3
    assert summary["cfg_branch"] == "uncond"
