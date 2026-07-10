import torch

from dicache_test_utils import FakeJiT
from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor
from pfc.risk.jit_counterfactual import capture_fresh_step
from pfc.risk.jit_plans import resolve_jit_boundary_plans


def test_fresh_capture_matches_normal_forward_for_both_branches():
    net = FakeJiT(depth=6, in_context_start=2, in_context_len=2)
    executor = JiTDiCacheExecutor(net)
    state = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 10
    flat_t = torch.tensor([0.25])
    velocity_t = flat_t.reshape(1, 1, 1, 1)
    cond_labels = torch.tensor([1])
    uncond_labels = torch.tensor([net.num_classes])
    plans = resolve_jit_boundary_plans(6)
    expected_cond = net(state, flat_t, cond_labels)
    expected_uncond = net(state, flat_t, uncond_labels)
    capture = capture_fresh_step(
        executor,
        state,
        flat_t,
        velocity_t,
        cond_labels,
        uncond_labels,
        plans=plans,
        step_idx=0,
        t=0.25,
        t_next=0.5,
        dt=0.25,
        cfg_scale_effective=3.0,
        t_eps=0.05,
    )
    assert torch.allclose(capture.cond.raw_output, expected_cond)
    assert torch.allclose(capture.uncond.raw_output, expected_uncond)
    assert capture.cond.shallow_probe.shape[1] == 4
    assert set(capture.cond.boundary_inputs) == {plan.name for plan in plans}
    assert set(capture.cond.boundary_outputs) == {plan.name for plan in plans}
    assert not capture.cond.raw_output.requires_grad
