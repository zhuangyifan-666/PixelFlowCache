from __future__ import annotations

import json

import torch.nn as nn

from pfc.adapters import JiTBoundaryAdapter


class FakeJiT(nn.Module):
    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_blocks)])


def test_jit_adapter_lists_whole_backbone() -> None:
    adapter = JiTBoundaryAdapter()
    model = FakeJiT(num_blocks=4)
    candidates = {candidate.name: candidate for candidate in adapter.list_boundary_candidates(model)}
    assert "whole_backbone" in candidates
    assert candidates["whole_backbone"].module_names == ("blocks.0", "blocks.1", "blocks.2", "blocks.3")
    assert candidates["whole_backbone"].is_quality_critical
    assert candidates["whole_backbone"].is_speed_critical


def test_jit_default_boundary_set_and_describe_are_json_ready() -> None:
    adapter = JiTBoundaryAdapter()
    boundary_set = adapter.default_boundary_set(FakeJiT(num_blocks=3))
    assert boundary_set.name == "jit_whole_backbone"
    assert boundary_set.module_names() == ("blocks.0", "blocks.1", "blocks.2")
    json.dumps(adapter.describe())
    json.dumps(boundary_set.to_dict())
