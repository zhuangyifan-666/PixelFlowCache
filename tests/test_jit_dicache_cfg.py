from __future__ import annotations

import torch

from pfc.cache.dicache_policy import DiCachePolicy
from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor, sample_jit_dicache
from pfc.eval.jit_runtime import cfg_enabled, combine_cfg_velocity, xpred_to_velocity
from dicache_test_utils import FakeDenoiser, FakeJiT, runtime_config


def test_cfg_helpers_match_existing_xpred_math_and_interval() -> None:
    z = torch.ones(1, 2, 2, 2)
    t = torch.full((1, 1, 1, 1), 0.5)
    cond = 3.0 * z
    uncond = 2.0 * z
    v_cond = xpred_to_velocity(cond, z, t, 0.05)
    v_uncond = xpred_to_velocity(uncond, z, t, 0.05)
    assert torch.allclose(combine_cfg_velocity(v_cond, v_uncond, 3.0), v_uncond + 3.0 * (v_cond - v_uncond))
    assert cfg_enabled(0.5, 0.1, 1.0)
    assert not cfg_enabled(0.1, 0.1, 1.0)


def test_dicache_sampler_uses_shared_decisions_fresh_final_layer_and_branch_histories() -> None:
    net = FakeJiT()
    model = FakeDenoiser(net)
    executor = JiTDiCacheExecutor(net)
    policy = DiCachePolicy(
        total_blocks=len(net.blocks),
        total_steps=4,
        probe_depth=1,
        reuse_threshold=100.0,
        ret_ratio=0.0,
        force_last_step_full=True,
    )
    noise = torch.ones(2, 2, 2, 2)
    labels = torch.tensor([0, 1])
    output, records = sample_jit_dicache(
        model,
        labels,
        noise,
        runtime_config(steps=4),
        executor=executor,
        policy=policy,
        collect_step_records=True,
    )
    assert output.shape == noise.shape
    assert [record["dicache_decision"] for record in records] == ["full", "reuse", "reuse", "full"]
    assert net.blocks[0].calls == 8
    assert net.x_embedder.calls == 8
    assert net.t_embedder.calls == 8
    assert all(block.calls == 4 for block in net.blocks[1:])
    assert net.final_layer.calls == 8
    assert len(policy.state.branch_histories["cond"].full_residual_history) == 2
    assert len(policy.state.branch_histories["uncond"].full_residual_history) == 2
    assert policy.state.branch_histories["cond"].full_residual_history[-1] is not policy.state.branch_histories["uncond"].full_residual_history[-1]
    summary = policy.summary()
    assert summary["probe_block_calls"] == 8
    assert summary["deep_block_calls"] == 20
    assert summary["actual_block_calls"] == 28
    assert summary["reference_block_calls"] == 48
    assert summary["share_cfg_prefix"] is False
    assert summary["reference_cfg_prefix_calls"] == 8
    assert summary["actual_cfg_prefix_calls"] == 8
    assert summary["cfg_prefix_calls_saved"] == 0


def test_cond_and_uncond_receive_different_conditions() -> None:
    net = FakeJiT()
    executor = JiTDiCacheExecutor(net)
    _h0, t_emb, _tokens = executor.prepare_common_input(torch.ones(1, 2, 2, 2), torch.tensor([0.5]))
    _yc, cond = executor.prepare_branch_condition(t_emb, torch.tensor([1]))
    _yu, uncond = executor.prepare_branch_condition(t_emb, torch.tensor([net.num_classes]))
    assert not torch.allclose(cond, uncond)


def test_shared_prefix_ablation_reduces_only_prefix_calls() -> None:
    strict_net = FakeJiT()
    shared_net = FakeJiT()
    shared_net.load_state_dict(strict_net.state_dict())
    noise = torch.ones(2, 2, 2, 2)
    labels = torch.tensor([0, 1])
    config = runtime_config(steps=3)
    strict_output, _ = sample_jit_dicache(
        FakeDenoiser(strict_net),
        labels,
        noise,
        config,
        executor=JiTDiCacheExecutor(strict_net),
        policy=DiCachePolicy(
            total_blocks=len(strict_net.blocks),
            total_steps=3,
            force_full=True,
            share_cfg_prefix=False,
        ),
    )
    shared_policy = DiCachePolicy(
        total_blocks=len(shared_net.blocks),
        total_steps=3,
        force_full=True,
        share_cfg_prefix=True,
    )
    shared_output, _ = sample_jit_dicache(
        FakeDenoiser(shared_net),
        labels,
        noise,
        config,
        executor=JiTDiCacheExecutor(shared_net),
        policy=shared_policy,
    )
    assert torch.equal(strict_output, shared_output)
    assert strict_net.x_embedder.calls == 6
    assert strict_net.t_embedder.calls == 6
    assert shared_net.x_embedder.calls == 3
    assert shared_net.t_embedder.calls == 3
    summary = shared_policy.summary()
    assert summary["reference_cfg_prefix_calls"] == 6
    assert summary["actual_cfg_prefix_calls"] == 3
    assert summary["cfg_prefix_calls_saved"] == 3
