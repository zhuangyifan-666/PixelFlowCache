import math

import torch

from pfc.risk.metrics import (
    equivalence_metrics,
    solver_scaled_rms_tensor,
    transition_relative_l2_tensor,
)


def test_solver_scaled_rms_matches_manual_calculation():
    current = torch.tensor([[[[1.0, -2.0]]]])
    fresh = torch.tensor([[[[1.5, -1.0]]]])
    candidate = fresh + torch.tensor([[[[0.1, -0.2]]]])
    value = solver_scaled_rms_tensor(candidate, fresh, current, atol=0.5, rtol=0.25)
    scale = 0.5 + 0.25 * torch.maximum(current.abs(), fresh.abs())
    expected = torch.sqrt(torch.mean(torch.square((candidate - fresh) / scale)))
    assert torch.allclose(value, expected)


def test_transition_relative_l2_and_equivalence_are_finite():
    current = torch.zeros(1, 1, 1, 2)
    fresh = torch.tensor([[[[1.0, 2.0]]]])
    candidate = torch.tensor([[[[2.0, 2.0]]]])
    assert math.isfinite(float(transition_relative_l2_tensor(candidate, fresh, current)))
    mismatch = equivalence_metrics(torch.zeros(1), torch.zeros(2))
    assert not mismatch["shape_match"]
    assert math.isfinite(mismatch["max_abs"])
