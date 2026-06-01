from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.deco_wrap import deco_cache_unit_category, parse_deco_cache_spec, wrap_deco_modules
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FakeDeCoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])
        self.dec_net = nn.Module()
        self.dec_net.res_blocks = nn.ModuleList([FakeBlock()])
        self.dec_net.final_layer = FakeBlock()
        self.dec_net.adaLN_modulation = nn.Linear(4, 4)


def test_deco_cache_spec_named_groups() -> None:
    candidates = ["blocks.0", "blocks.1", "dec_net.res_blocks.0", "dec_net.final_layer"]
    assert parse_deco_cache_spec("none", candidates) == []
    assert parse_deco_cache_spec("backbone_blocks", candidates) == ["blocks.0", "blocks.1"]
    assert parse_deco_cache_spec("decoder_blocks", candidates) == ["dec_net.res_blocks.0"]
    assert parse_deco_cache_spec("final", candidates) == ["dec_net.final_layer"]
    assert parse_deco_cache_spec("all_candidates", candidates) == candidates


def test_deco_cache_spec_explicit_and_topk(tmp_path: Path) -> None:
    candidates = ["blocks.0", "blocks.1", "dec_net.res_blocks.0"]
    assert parse_deco_cache_spec("blocks.1,dec_net.res_blocks.0", candidates) == ["blocks.1", "dec_net.res_blocks.0"]
    csv_path = tmp_path / "candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["module_name", "mean_rel_l2_delta"])
        writer.writeheader()
        writer.writerow({"module_name": "blocks.1", "mean_rel_l2_delta": "0.2"})
        writer.writerow({"module_name": "blocks.0", "mean_rel_l2_delta": "0.1"})
    assert parse_deco_cache_spec(f"topk:{csv_path}:1", candidates) == ["blocks.0"]


def test_deco_norm_and_parent_modules_are_not_cache_units() -> None:
    assert deco_cache_unit_category("blocks.0") == "backbone_block"
    assert deco_cache_unit_category("blocks.0.norm1") == "norm_or_modulation"
    assert deco_cache_unit_category("dec_net") == "other"
    assert deco_cache_unit_category("dec_net.adaLN_modulation") == "norm_or_modulation"


def test_wrap_deco_modules_wraps_only_safe_units() -> None:
    net = FakeDeCoNet()
    cache_state = RuntimeCacheState(model_name="DeCo")
    policy = FixedIntervalCachePolicy(interval=2, cache_modules={"blocks.0", "dec_net.res_blocks.0", "dec_net.adaLN_modulation"})
    wrapped = wrap_deco_modules(net, cache_state, policy, ["blocks.0", "dec_net.res_blocks.0", "dec_net.adaLN_modulation"])
    assert wrapped == ["blocks.0", "dec_net.res_blocks.0"]
    assert isinstance(net.blocks[0], CachedModule)
    assert isinstance(net.dec_net.res_blocks[0], CachedModule)
    assert not isinstance(net.dec_net.adaLN_modulation, CachedModule)

    cache_state.set_context(0, 0.2, "cfg_cat")
    out = net.blocks[0](torch.ones(2, 4))
    assert out.shape == (2, 4)
