from __future__ import annotations

import json

import torch

from pfc.profiling.tensor_stats import (
    cosine_similarity_flat,
    relative_l2_delta,
    summarize_delta,
    summarize_tensor,
)


def test_summarize_tensor_json_serializable() -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    record = summarize_tensor(tensor, name="x")
    json.dumps(record)
    assert record["name"] == "x"
    assert record["shape"] == [2, 4]
    assert record["numel"] == 8
    assert record["has_nan"] is False


def test_delta_and_cosine() -> None:
    previous = torch.tensor([1.0, 0.0])
    current = torch.tensor([2.0, 0.0])
    assert relative_l2_delta(current, previous) == 1.0
    assert cosine_similarity_flat(current, previous) == 1.0
    delta = summarize_delta(current, previous)
    assert delta["delta_l2"] == 1.0


def test_fp16_summary() -> None:
    tensor = torch.ones(4, dtype=torch.float16)
    record = summarize_tensor(tensor)
    assert record["dtype"] == "torch.float16"
    assert record["mean"] == 1.0

