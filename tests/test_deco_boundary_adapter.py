from __future__ import annotations

import json

import torch
import torch.nn as nn

from pfc.adapters import DeCoBoundaryAdapter


class NonTinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeDeCo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([NonTinyModule(), NonTinyModule()])
        self.dec_net = nn.Module()
        self.dec_net.res_blocks = nn.ModuleList([NonTinyModule()])
        self.dec_net.final_layer = NonTinyModule()


def test_deco_adapter_lists_expected_boundaries() -> None:
    adapter = DeCoBoundaryAdapter()
    candidates = {candidate.name: candidate for candidate in adapter.list_boundary_candidates(FakeDeCo())}
    assert {"backbone_blocks", "decoder_blocks", "final_output", "backbone_plus_final", "all_candidates"}.issubset(
        candidates
    )
    assert candidates["final_output"].is_quality_critical
    assert candidates["backbone_blocks"].is_speed_critical
    assert candidates["final_output"].module_names == ("dec_net.final_layer",)


def test_deco_default_boundary_sets() -> None:
    adapter = DeCoBoundaryAdapter()
    model = FakeDeCo()
    all_candidates = adapter.default_boundary_set(model, "bfc_all_candidates_t02_10")
    backbone_plus_final = adapter.default_boundary_set(model, "bfc_backbone_plus_final_t02_10")
    assert "blocks.0" in all_candidates.module_names()
    assert "dec_net.res_blocks.0" in all_candidates.module_names()
    assert "dec_net.final_layer" in all_candidates.module_names()
    assert "dec_net.res_blocks.0" not in backbone_plus_final.module_names()
    assert "dec_net.final_layer" in backbone_plus_final.module_names()
    json.dumps(adapter.describe())
    json.dumps(all_candidates.to_dict())
