from __future__ import annotations

import pytest

from pfc.cache.wrap import parse_layer_list


def test_parse_named_layer_specs() -> None:
    assert parse_layer_list("all", 12) == list(range(12))
    assert parse_layer_list("none", 12) == []
    assert parse_layer_list("middle", 12) == [3, 4, 5, 6, 7, 8]
    assert parse_layer_list("early", 12) == [0, 1, 2]
    assert parse_layer_list("late", 12) == [9, 10, 11]


def test_parse_comma_list() -> None:
    assert parse_layer_list("0,1,11,1", 12) == [0, 1, 11]


def test_parse_invalid_spec_raises() -> None:
    with pytest.raises(ValueError):
        parse_layer_list("bad", 12)
    with pytest.raises(ValueError):
        parse_layer_list("12", 12)
