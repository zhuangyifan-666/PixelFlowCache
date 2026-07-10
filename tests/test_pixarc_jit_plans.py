import pytest

from pfc.risk.jit_plans import resolve_jit_boundary_plans


def test_default_twelve_block_plans_match_research_definition():
    plans = resolve_jit_boundary_plans(12)
    assert [(plan.name, plan.start_block, plan.end_block) for plan in plans] == [
        ("early", 0, 4),
        ("middle", 4, 8),
        ("late", 8, 12),
        ("early_middle", 0, 8),
        ("whole", 0, 12),
    ]


def test_plans_resolve_for_non_twelve_depth_and_optional_middle_late():
    plans = resolve_jit_boundary_plans(10, ["early", "middle_late"])
    assert [(plan.start_block, plan.end_block) for plan in plans] == [(0, 3), (3, 10)]
    assert plans[0].skipped_block_count == 3


@pytest.mark.parametrize("names", [["unknown"], ["early", "early"]])
def test_invalid_plan_selection_is_rejected(names):
    with pytest.raises(ValueError):
        resolve_jit_boundary_plans(12, names)
