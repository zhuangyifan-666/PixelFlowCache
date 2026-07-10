from __future__ import annotations

import torch
import pytest

from dicache_test_utils import FakeDenoiser, FakeJiT, runtime_config
from pfc.cache.dicache_policy import DiCachePolicy
from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor, sample_jit_dicache
import pfc.eval.jit_dicache_runtime as jit_dicache_runtime
from pfc.eval.jit_runtime import sample_jit


def _sample_force_full(
    net: FakeJiT,
    labels: torch.Tensor,
    noise: torch.Tensor,
    *,
    share_cfg_prefix: bool,
) -> tuple[torch.Tensor, list[dict[str, object]], DiCachePolicy]:
    policy = DiCachePolicy(
        total_blocks=len(net.blocks),
        total_steps=4,
        probe_depth=1,
        force_full=True,
        share_cfg_prefix=share_cfg_prefix,
    )
    output, records = sample_jit_dicache(
        FakeDenoiser(net),
        labels,
        noise,
        runtime_config(steps=4),
        executor=JiTDiCacheExecutor(net),
        policy=policy,
        collect_step_records=True,
    )
    return output, records, policy


def test_force_full_no_shared_prefix_matches_no_cache_sampler() -> None:
    no_cache_net = FakeJiT()
    dicache_net = FakeJiT()
    dicache_net.load_state_dict(no_cache_net.state_dict())
    labels = torch.tensor([0, 2])
    noise = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2) / 20.0
    config = runtime_config(steps=4)

    expected, expected_records = sample_jit(
        FakeDenoiser(no_cache_net),
        labels,
        noise,
        config,
        mode="no_cache_50",
        collect_step_records=True,
    )
    actual, actual_records, policy = _sample_force_full(
        dicache_net,
        labels,
        noise,
        share_cfg_prefix=False,
    )

    assert torch.equal(actual, expected)
    assert no_cache_net.x_embedder.calls == dicache_net.x_embedder.calls == 8
    assert no_cache_net.t_embedder.calls == dicache_net.t_embedder.calls == 8
    assert [block.calls for block in no_cache_net.blocks] == [8] * len(no_cache_net.blocks)
    assert [block.calls for block in dicache_net.blocks] == [8] * len(dicache_net.blocks)
    assert no_cache_net.final_layer.calls == dicache_net.final_layer.calls == 8
    for expected_block, actual_block in zip(no_cache_net.blocks, dicache_net.blocks):
        assert len(expected_block.conditions) == len(actual_block.conditions)
        for expected_condition, actual_condition in zip(
            expected_block.conditions,
            actual_block.conditions,
        ):
            assert torch.equal(expected_condition, actual_condition)
    assert [record["t"] for record in expected_records] == [
        record["t"] for record in actual_records
    ]
    assert [record["cfg_enabled"] for record in expected_records] == [
        record["cfg_enabled"] for record in actual_records
    ]
    assert all(record["dicache_decision"] == "full" for record in actual_records)
    assert policy.summary()["actual_cfg_prefix_calls"] == 8


def test_shared_prefix_ablation_preserves_force_full_output() -> None:
    strict_net = FakeJiT()
    shared_net = FakeJiT()
    shared_net.load_state_dict(strict_net.state_dict())
    labels = torch.tensor([1, 2])
    noise = torch.ones(2, 2, 2, 2)
    strict, _, _ = _sample_force_full(
        strict_net,
        labels,
        noise,
        share_cfg_prefix=False,
    )
    shared, _, policy = _sample_force_full(
        shared_net,
        labels,
        noise,
        share_cfg_prefix=True,
    )
    assert torch.equal(shared, strict)
    assert strict_net.x_embedder.calls == 8
    assert shared_net.x_embedder.calls == 4
    assert policy.summary()["cfg_prefix_calls_saved"] == 4


def test_dicache_step_record_norm_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        jit_dicache_runtime,
        "l2_norm",
        lambda _value: pytest.fail("l2_norm called while records are disabled"),
    )
    net = FakeJiT()
    policy = DiCachePolicy(total_blocks=len(net.blocks), total_steps=2, force_full=True)
    output, records = sample_jit_dicache(
        FakeDenoiser(net),
        torch.tensor([0, 1]),
        torch.ones(2, 2, 2, 2),
        runtime_config(steps=2),
        executor=JiTDiCacheExecutor(net),
        policy=policy,
    )
    assert output.shape == (2, 2, 2, 2)
    assert records == []
