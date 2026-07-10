from __future__ import annotations

import torch
from torch import nn

from pfc.cache.cache_state import InputSignature, RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


class CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + self.calls


def _wrapped(clone: bool = False) -> tuple[CountingIdentity, RuntimeCacheState, CachedModule]:
    module = CountingIdentity()
    state = RuntimeCacheState(clone_on_store=clone)
    wrapped = CachedModule(module, "blocks.0", state, FixedIntervalCachePolicy(interval=2))
    return module, state, wrapped


def test_shape_and_dtype_changes_do_not_reuse() -> None:
    module, state, wrapped = _wrapped()
    state.set_context(0, 0.0, "cond")
    wrapped(torch.zeros(1, 2, dtype=torch.float32))
    state.set_context(1, 0.1, "cond")
    wrapped(torch.zeros(1, 3, dtype=torch.float32))
    state.set_context(3, 0.3, "cond")
    wrapped(torch.zeros(1, 3, dtype=torch.float64))
    assert module.calls == 3
    assert state.summary()["hits"] == 0


def test_session_change_clears_entries_and_prevents_reuse() -> None:
    module, state, wrapped = _wrapped()
    state.begin_batch(session_id="batch-a")
    state.set_context(0, 0.0, "cond")
    wrapped(torch.zeros(1, 2))
    state.begin_batch(session_id="batch-b")
    state.set_context(1, 0.1, "cond")
    wrapped(torch.zeros(1, 2))
    assert module.calls == 2


def test_input_signature_records_device_and_batch() -> None:
    signature = InputSignature.from_tensor(torch.zeros(4, 2), session_id=7)
    assert signature.device_type == "cpu"
    assert signature.device_index is None
    assert signature.batch_size == 4
    assert signature.session_id == 7


def test_clone_on_store_breaks_storage_alias() -> None:
    state = RuntimeCacheState(clone_on_store=True)
    tensor = torch.ones(1)
    entry = state.put(state.make_key("x"), tensor, input_signature=InputSignature.from_tensor(tensor))
    tensor.zero_()
    assert entry.tensor.item() == 1.0
