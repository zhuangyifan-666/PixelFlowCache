import pytest
import torch

from pfc.risk.jit_history import JiTFreshBoundaryHistory


def _append(history, sample, branch, step, value):
    tensor = torch.tensor([float(value)])
    history.append(
        sample_global_index=sample,
        branch=branch,
        boundary_plan="early",
        step_idx=step,
        t=step / 10,
        boundary_input=tensor + 10,
        boundary_output=tensor,
    )


def test_age_selection_taylor_and_no_current_step_access():
    history = JiTFreshBoundaryHistory(max_history=3)
    _append(history, 0, "cond", 0, 1)
    _append(history, 0, "cond", 1, 3)
    assert history.select_age(
        sample_global_index=0, branch="cond", boundary_plan="early", current_step_idx=2, age=1
    ).step_idx == 1
    assert history.select_age(
        sample_global_index=0, branch="cond", boundary_plan="early", current_step_idx=2, age=2
    ).step_idx == 0
    assert history.select_age(
        sample_global_index=0, branch="cond", boundary_plan="early", current_step_idx=1, age=2
    ) is None
    assert torch.equal(
        history.taylor_order_1(
            sample_global_index=0, branch="cond", boundary_plan="early", current_step_idx=2
        ),
        torch.tensor([5.0]),
    )


def test_branch_sample_isolation_clone_and_clear():
    history = JiTFreshBoundaryHistory()
    source = torch.tensor([2.0])
    _append(history, 0, "cond", 0, source.item())
    source.fill_(99)
    assert history.select_age(
        sample_global_index=0, branch="uncond", boundary_plan="early", current_step_idx=1, age=1
    ) is None
    assert history.select_age(
        sample_global_index=1, branch="cond", boundary_plan="early", current_step_idx=1, age=1
    ) is None
    assert history.select_age(
        sample_global_index=0, branch="cond", boundary_plan="early", current_step_idx=1, age=1
    ).boundary_output.item() == 2.0
    history.clear_sample(0)
    assert history.item_count() == 0


def test_non_increasing_history_and_age_zero_are_rejected():
    history = JiTFreshBoundaryHistory()
    _append(history, 0, "cond", 1, 1)
    with pytest.raises(ValueError, match="increase strictly"):
        _append(history, 0, "cond", 1, 2)
    with pytest.raises(ValueError, match="age 0"):
        history.select_age(
            sample_global_index=0, branch="cond", boundary_plan="early", current_step_idx=2, age=0
        )
