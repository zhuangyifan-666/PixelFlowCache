from __future__ import annotations

import pytest
import torch

from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor
from dicache_test_utils import FakeJiT


@pytest.mark.parametrize("labels", [torch.tensor([1, 2]), torch.tensor([3, 3])])
@pytest.mark.parametrize("probe_depth", [1, 3])
def test_split_full_matches_normal_forward_with_context(labels: torch.Tensor, probe_depth: int) -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2) / 10.0
    t = torch.tensor([0.2, 0.7])
    normal_net = FakeJiT()
    split_net = FakeJiT()
    split_net.load_state_dict(normal_net.state_dict())
    expected = normal_net(x, t, labels)
    actual = JiTDiCacheExecutor(split_net).forward_split_full(x, t, labels, probe_depth=probe_depth)
    assert torch.allclose(actual, expected)
    assert [block.calls for block in split_net.blocks] == [1] * len(split_net.blocks)
    assert split_net.final_layer.calls == 1


def test_resume_does_not_repeat_probe_and_context_is_inserted_once() -> None:
    net = FakeJiT()
    executor = JiTDiCacheExecutor(net)
    x = torch.ones(1, 2, 2, 2)
    executor.forward_split_full(x, torch.tensor([0.5]), torch.tensor([1]), probe_depth=1)
    assert [block.calls for block in net.blocks] == [1] * len(net.blocks)
    assert net.blocks[0].sequence_lengths == [4]
    assert net.blocks[1].sequence_lengths == [4]
    assert all(block.sequence_lengths == [6] for block in net.blocks[2:])


def test_probe_crossing_context_extracts_only_image_tail() -> None:
    net = FakeJiT()
    executor = JiTDiCacheExecutor(net)
    h0, t_emb, tokens = executor.prepare_common_input(torch.ones(1, 2, 2, 2), torch.tensor([0.5]))
    y_emb, condition = executor.prepare_branch_condition(t_emb, torch.tensor([1]))
    hidden = executor.run_blocks_range(h0, condition, y_emb, start=0, end=3, num_image_tokens=tokens)
    assert hidden.shape[1] == 6
    assert executor.extract_image_tokens(hidden, tokens).shape == h0.shape


def test_extract_image_tokens_rejects_invalid_sequence_shape() -> None:
    executor = JiTDiCacheExecutor(FakeJiT())
    with pytest.raises(ValueError, match="cannot extract"):
        executor.extract_image_tokens(torch.zeros(1, 5, 2), 4)


def test_bf16_fixed_position_buffers_do_not_promote_hidden_dtype() -> None:
    normal_net = FakeJiT()
    split_net = FakeJiT()
    split_net.load_state_dict(normal_net.state_dict())
    x = torch.ones(2, 2, 2, 2, dtype=torch.bfloat16)
    t = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
    labels = torch.tensor([1, 2])
    expected = normal_net(x, t, labels)
    actual = JiTDiCacheExecutor(split_net).forward_split_full(
        x,
        t,
        labels,
        probe_depth=1,
    )
    assert expected.dtype == torch.bfloat16
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
