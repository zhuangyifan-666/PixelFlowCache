from __future__ import annotations

import torch
from torch import nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.speca_policy import SpeCaCachePolicy


class CountingAdd(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + float(self.calls)


def test_cached_module_speca_verifies_lazily_and_keeps_forecast_committed() -> None:
    module = CountingAdd()
    state = RuntimeCacheState(model_name="JiT")
    policy = SpeCaCachePolicy(
        cache_modules={"blocks.0"},
        verifier_module="auto",
        first_full_steps=2,
        min_history=2,
        min_forecast_steps=2,
        max_forecast_steps=5,
        base_threshold=0.1,
        decay_rate=1.0,
        total_steps=10,
    )
    wrapped = CachedModule(module, "blocks.0", state, policy)
    x = torch.zeros(1, 1)

    outputs: dict[tuple[int, str], torch.Tensor] = {}
    for step in range(5):
        for branch in ("cond", "uncond"):
            state.set_context(step, step / 10.0, branch)
            outputs[(step, branch)] = wrapped(x)

    assert module.calls == 6
    assert torch.allclose(outputs[(2, "cond")], torch.tensor([[5.0]]))
    assert torch.allclose(outputs[(2, "uncond")], torch.tensor([[6.0]]))
    assert torch.allclose(outputs[(3, "cond")], torch.tensor([[7.0]]))
    assert torch.allclose(outputs[(3, "uncond")], torch.tensor([[8.0]]))
    assert torch.allclose(outputs[(4, "cond")], torch.tensor([[9.0]]))
    assert torch.allclose(outputs[(4, "uncond")], torch.tensor([[10.0]]))
    assert state.summary()["hits"] == 6
    assert state.summary()["misses"] == 4

    cond_entry = state.get(state.make_key("blocks.0", batch_signature="b:1"))
    assert cond_entry is not None
    assert cond_entry.step_idx == 1
    cond_history = policy._history[("blocks.0", "uncond", "euler", "b:1")]
    assert [step for step, _tensor in cond_history] == [0, 1]
    summary = policy.summary()
    assert summary["verification_steps"] == 1
    assert summary["verifier_fresh_calls"] == 2
    assert summary["forecast_committed"] == 6
    assert summary["verification_errors"]["count"] == 2
    assert summary["full_compute_calls"] == 4
    assert summary["logical_managed_calls"] == 10
    assert summary["actual_original_module_forwards"] == 6
    assert summary["effective_skipped_block_calls"] == 4
    assert summary["raw_forecast_rate"] == 0.6
    assert summary["effective_compute_saving_rate"] == 0.4
    assert summary["verifier_overhead_rate"] == 0.2
    assert summary["actual_compute_fraction"] == 0.6
    overhead = summary["verification_overhead_stats"]
    assert overhead["timing_semantics"] == "host_dispatch_only"
    assert overhead["cuda_event_profiling_enabled"] is False
    assert overhead["verification_cuda_time_sec"] is None
    assert "verification_wall_time_sec" not in overhead


def test_speca_non_verifier_and_disabled_verification_do_not_call_fresh_compute() -> None:
    policy = SpeCaCachePolicy(
        cache_modules={"blocks.0", "blocks.1"},
        first_full_steps=2,
        min_history=2,
        total_steps=10,
    )
    for step in range(2):
        for module in ("blocks.0", "blocks.1"):
            for branch in ("cond", "uncond"):
                policy.on_refresh_committed(
                    step_idx=step,
                    t=float(step),
                    module_name=module,
                    cfg_branch=branch,
                    solver_stage="euler",
                    entry=None,
                    tensor=torch.tensor([[float(step)]]),
                )
    entry = type("Entry", (), {"tensor": torch.zeros(1, 1)})()
    assert policy.should_reuse_entry(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
    )
    calls = 0

    def fresh_compute() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones(1, 1)

    policy.process_reuse_tensor(
        step_idx=2,
        t=0.2,
        module_name="blocks.0",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
        current_input=torch.zeros(1, 1),
        reuse_tensor=torch.zeros(1, 1),
        fresh_compute=fresh_compute,
    )
    policy.process_reuse_tensor(
        step_idx=2,
        t=0.2,
        module_name="blocks.1",
        cfg_branch="cond",
        solver_stage="euler",
        entry=entry,
        current_input=torch.zeros(1, 1),
        reuse_tensor=torch.zeros(1, 1),
        fresh_compute=fresh_compute,
    )
    assert calls == 0
