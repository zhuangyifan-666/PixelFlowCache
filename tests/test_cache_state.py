from __future__ import annotations

import json

import torch

from pfc.cache.cache_state import RuntimeCacheState


def test_put_get_and_clear_cache_entries() -> None:
    state = RuntimeCacheState(model_name="tiny")
    state.set_context(2, 0.2, "cond")
    key = state.make_key("blocks.0", batch_signature="b:1")
    tensor = torch.ones(1, 2)
    entry = state.put(key, tensor)

    assert state.get(key) is entry
    assert entry.step_idx == 2
    assert entry.t == 0.2
    assert not entry.tensor.requires_grad

    state.clear()
    assert state.get(key) is None


def test_stats_and_summary_are_json_serializable() -> None:
    state = RuntimeCacheState(model_name="tiny")
    state.mark_hit("blocks.0")
    state.mark_miss("blocks.0")
    state.mark_refresh("blocks.0")
    state.mark_disabled("blocks.1")

    summary = state.summary()
    assert summary["total_calls"] == 3
    assert summary["hits"] == 1
    assert summary["misses"] == 1
    assert summary["refreshes"] == 1
    assert summary["disabled"] == 1
    assert summary["by_module"]["blocks.0"]["refreshes"] == 1
    json.dumps(summary)


def test_reset_stats() -> None:
    state = RuntimeCacheState()
    state.mark_hit("blocks.0")
    state.reset_stats()
    assert state.summary()["total_calls"] == 0
