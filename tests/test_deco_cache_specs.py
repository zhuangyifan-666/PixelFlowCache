from __future__ import annotations

import pytest

from pfc.cache.deco_wrap import parse_deco_cache_spec


MODULES = [
    "blocks.0",
    "blocks.1",
    "blocks.2",
    "blocks.3",
    "dec_net.res_blocks.0",
    "dec_net.res_blocks.1",
    "dec_net.final_layer",
]


def test_final_only_selects_exactly_final() -> None:
    assert parse_deco_cache_spec("final_only", MODULES) == ["dec_net.final_layer"]


def test_decoder_no_final_excludes_final() -> None:
    assert parse_deco_cache_spec("decoder_only_no_final", MODULES) == [
        "dec_net.res_blocks.0",
        "dec_net.res_blocks.1",
    ]


def test_plus_final_specs_include_final() -> None:
    assert parse_deco_cache_spec("backbone_plus_final", MODULES) == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
        "dec_net.final_layer",
    ]
    assert parse_deco_cache_spec("decoder_plus_final", MODULES) == [
        "dec_net.res_blocks.0",
        "dec_net.res_blocks.1",
        "dec_net.final_layer",
    ]


def test_all_candidates_and_no_final_specs() -> None:
    assert parse_deco_cache_spec("all_candidates", MODULES) == MODULES
    assert parse_deco_cache_spec("backbone_plus_decoder_no_final", MODULES) == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
        "dec_net.res_blocks.0",
        "dec_net.res_blocks.1",
    ]


def test_late_backbone_specs_parse_n() -> None:
    assert parse_deco_cache_spec("late_backbone_only:2", MODULES) == ["blocks.2", "blocks.3"]
    assert parse_deco_cache_spec("late_backbone_plus_final:2", MODULES) == [
        "blocks.2",
        "blocks.3",
        "dec_net.final_layer",
    ]


def test_invalid_specs_raise() -> None:
    with pytest.raises(ValueError):
        parse_deco_cache_spec("late_backbone_only:0", MODULES)
    with pytest.raises(ValueError):
        parse_deco_cache_spec("late_backbone_plus_final:not_int", MODULES)
    with pytest.raises(ValueError):
        parse_deco_cache_spec("missing_module", MODULES)
