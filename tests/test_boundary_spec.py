from __future__ import annotations

import json

from pfc.core.boundary_spec import BoundaryGranularity, BoundaryRole, BoundarySet, BoundarySpec


def test_boundary_spec_roundtrip() -> None:
    spec = BoundarySpec(
        name="whole_backbone",
        module_names=("blocks.0", "blocks.1"),
        role=BoundaryRole.BACKBONE,
        granularity=BoundaryGranularity.WHOLE_BACKBONE,
        is_quality_critical=True,
        extra={"source": "test"},
    )
    data = spec.to_dict()
    assert data["role"] == "backbone"
    assert BoundarySpec.from_dict(data) == spec
    json.dumps(data)


def test_boundary_set_helpers() -> None:
    first = BoundarySpec(
        name="first",
        module_names=("a", "b"),
        role=BoundaryRole.CUSTOM,
        granularity=BoundaryGranularity.MODULE_GROUP,
    )
    second = BoundarySpec(
        name="second",
        module_names=("b", "c"),
        role=BoundaryRole.CUSTOM,
        granularity=BoundaryGranularity.MODULE_GROUP,
    )
    boundary_set = BoundarySet(name="demo", boundaries=(first, second))
    assert boundary_set.module_names() == ("a", "b", "c")
    assert boundary_set.get("second") == second
    assert boundary_set.get("missing") is None
    assert boundary_set.contains_module("c")
    json.dumps(boundary_set.to_dict())
