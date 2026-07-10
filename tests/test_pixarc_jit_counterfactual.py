from types import SimpleNamespace

import pytest
import torch

from dicache_test_utils import FakeDenoiser, FakeJiT
from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor
from pfc.risk.jit_counterfactual import capture_fresh_step, evaluate_counterfactual_transition
from pfc.risk.jit_plans import JiTBoundaryPlan, resolve_jit_boundary_plans
from scripts.run_jit_pixarc_stage1_instrument import instrument_sample


@pytest.mark.parametrize(
    "plan",
    [
        JiTBoundaryPlan("before_context", 0, 2, "ends before context insertion"),
        JiTBoundaryPlan("after_context", 1, 4, "ends after context insertion"),
        JiTBoundaryPlan("whole", 0, 6, "whole stack"),
    ],
)
def test_age_zero_replay_matches_fresh_across_context_boundary(plan):
    net = FakeJiT(depth=6, in_context_start=2, in_context_len=2)
    executor = JiTDiCacheExecutor(net)
    state = torch.randn(1, 2, 2, 2)
    original_state = state.clone()
    flat_t = torch.tensor([0.4])
    velocity_t = flat_t.reshape(1, 1, 1, 1)
    cond = torch.tensor([1])
    uncond = torch.tensor([net.num_classes])
    fresh = capture_fresh_step(
        executor,
        state,
        flat_t,
        velocity_t,
        cond,
        uncond,
        plans=[plan],
        step_idx=2,
        t=0.4,
        t_next=0.6,
        dt=0.2,
        cfg_scale_effective=3.0,
        t_eps=0.05,
    )
    replacements = {
        "cond": fresh.cond.boundary_outputs[plan.name],
        "uncond": fresh.uncond.boundary_outputs[plan.name],
    }
    replacement_snapshots = {key: value.clone() for key, value in replacements.items()}
    replay = evaluate_counterfactual_transition(
        executor,
        state,
        flat_t,
        velocity_t,
        cond,
        uncond,
        plan=plan,
        action="replay_age_0",
        replacements=replacements,
        cfg_scale_effective=3.0,
        dt=0.2,
        t_eps=0.05,
    )
    assert torch.allclose(replay.cond.raw_output, fresh.cond.raw_output)
    assert torch.allclose(replay.uncond.raw_output, fresh.uncond.raw_output)
    assert torch.allclose(replay.next_state, fresh.next_state)
    assert torch.equal(state, original_state)
    assert all(torch.equal(replacements[key], replacement_snapshots[key]) for key in replacements)


def _args(actions):
    return SimpleNamespace(
        steps=3,
        cfg=3.0,
        cfg_interval_min=0.1,
        cfg_interval_max=1.0,
        correctness_only=False,
        frequency_low_ratio=0.15,
        frequency_high_ratio=0.45,
        measure_action_latency=False,
        resolved_equivalence_steps={0, 1, 2},
        equivalence_atol=1e-6,
        equivalence_rtol=1e-5,
        strict_correctness=True,
        risk_atol=1e-3,
        risk_rtol=1e-2,
        run_id="fake",
        shard_index=0,
        num_shards=1,
        seed=21,
        actions=actions,
    )


def test_candidates_do_not_change_fresh_trajectory_or_history():
    plans = resolve_jit_boundary_plans(6, ["early", "late"])
    state = torch.randn(1, 2, 2, 2)
    full_net = FakeJiT(depth=6)
    reference_net = FakeJiT(depth=6)
    reference_net.load_state_dict(full_net.state_dict())
    full_actions = ["fresh", "replay_age_0", "reuse_age_1", "reuse_age_2", "taylor_order_1"]
    full_final, records, correctness, summary = instrument_sample(
        FakeDenoiser(full_net),
        JiTDiCacheExecutor(full_net),
        state.clone(),
        global_index=0,
        class_label=1,
        plans=plans,
        actions=full_actions,
        args=_args(full_actions),
    )
    reference_actions = ["fresh", "replay_age_0"]
    reference_final, _, _, _ = instrument_sample(
        FakeDenoiser(reference_net),
        JiTDiCacheExecutor(reference_net),
        state.clone(),
        global_index=0,
        class_label=1,
        plans=plans,
        actions=reference_actions,
        args=_args(reference_actions),
    )
    assert torch.allclose(full_final, reference_final)
    assert summary["history_items_after_clear"] == 0
    assert all(row["allclose"] for row in correctness)
    age1 = [row for row in records if row["action"] == "reuse_age_1"]
    age2 = [row for row in records if row["action"] == "reuse_age_2"]
    taylor = [row for row in records if row["action"] == "taylor_order_1"]
    assert all(not row["action_ready"] for row in age1 if row["step_idx"] == 0)
    assert all(row["action_ready"] for row in age1 if row["step_idx"] >= 1)
    assert all(not row["action_ready"] for row in age2 + taylor if row["step_idx"] < 2)
    assert all(row["action_ready"] for row in age2 + taylor if row["step_idx"] >= 2)


def test_cpu_diagnostic_latency_covers_each_ready_action():
    plans = resolve_jit_boundary_plans(6, ["early"])
    net = FakeJiT(depth=6)
    actions = ["fresh", "replay_age_0", "reuse_age_1", "reuse_age_2", "taylor_order_1"]
    args = _args(actions)
    args.measure_action_latency = True
    _, records, _, _ = instrument_sample(
        FakeDenoiser(net),
        JiTDiCacheExecutor(net),
        torch.randn(1, 2, 2, 2),
        global_index=0,
        class_label=1,
        plans=plans,
        actions=actions,
        args=args,
    )
    assert all(
        row["diagnostic_action_latency_ms"] is not None
        for row in records
        if row["action_ready"]
    )
    assert all(
        row["diagnostic_action_latency_ms"] is None
        for row in records
        if not row["action_ready"]
    )
