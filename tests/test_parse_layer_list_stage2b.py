from __future__ import annotations

import pytest

from pfc.cache.wrap import parse_layer_list


def test_prefix_suffix_range_every_specs() -> None:
    assert parse_layer_list("prefix:6", 12) == [0, 1, 2, 3, 4, 5]
    assert parse_layer_list("suffix:6", 12) == [6, 7, 8, 9, 10, 11]
    assert parse_layer_list("range:0:6", 12) == [0, 1, 2, 3, 4, 5]
    assert parse_layer_list("range:6:12", 12) == [6, 7, 8, 9, 10, 11]
    assert parse_layer_list("every:2", 12) == [0, 2, 4, 6, 8, 10]


def test_complement_spec() -> None:
    assert parse_layer_list("complement:middle", 12) == [0, 1, 2, 9, 10, 11]


def test_stage2b_invalid_specs_raise() -> None:
    for spec in ["prefix:x", "suffix:-1", "range:6:0", "range:0:13", "every:0", "complement:bad"]:
        with pytest.raises(ValueError):
            parse_layer_list(spec, 12)
