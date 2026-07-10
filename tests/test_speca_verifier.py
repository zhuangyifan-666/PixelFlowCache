from __future__ import annotations

import math

import pytest
import torch

from pfc.cache.speca_verifier import SPECA_ERROR_METRICS, calculate_speca_error


@pytest.mark.parametrize("metric", SPECA_ERROR_METRICS)
def test_speca_error_is_zero_for_identical_tensors(metric: str) -> None:
    value = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    assert calculate_speca_error(value, value.clone(), metric) == pytest.approx(0.0)


def test_speca_relative_errors_match_manual_calculation() -> None:
    predicted = torch.tensor([[2.0, 2.0]])
    fresh = torch.tensor([[1.0, 4.0]])
    relative = torch.tensor([1.0, 0.5])
    assert calculate_speca_error(predicted, fresh, "relative_l1") == pytest.approx(
        relative.mean().item()
    )
    assert calculate_speca_error(predicted, fresh, "relative_l2") == pytest.approx(
        relative.square().mean().sqrt().item()
    )


def test_speca_error_uses_float32_for_bfloat16_and_can_return_per_sample() -> None:
    predicted = torch.tensor([[1.0, 3.0], [2.0, 6.0]], dtype=torch.bfloat16)
    fresh = torch.tensor([[1.0, 1.0], [1.0, 2.0]], dtype=torch.bfloat16)
    scalar, per_sample = calculate_speca_error(
        predicted,
        fresh,
        "l1",
        return_per_sample=True,
    )
    expected = (predicted.float() - fresh.float()).abs()
    assert scalar == pytest.approx(expected.mean().item())
    assert per_sample.dtype == torch.float32
    assert torch.allclose(per_sample, expected.mean(dim=1))


def test_speca_cosine_error_and_zero_fresh_are_finite() -> None:
    orthogonal_a = torch.tensor([[1.0, 0.0]])
    orthogonal_b = torch.tensor([[0.0, 1.0]])
    assert calculate_speca_error(orthogonal_a, orthogonal_b, "cosine_error") == pytest.approx(1.0)

    zeros = torch.zeros(2, 3)
    for metric in SPECA_ERROR_METRICS:
        error = calculate_speca_error(torch.ones_like(zeros), zeros, metric)
        assert math.isfinite(error)


def test_speca_error_does_not_mutate_inputs_and_validates_arguments() -> None:
    predicted = torch.tensor([[1.0, 2.0]])
    fresh = torch.tensor([[2.0, 4.0]])
    predicted_before = predicted.clone()
    fresh_before = fresh.clone()
    calculate_speca_error(predicted, fresh, "relative_l1")
    assert torch.equal(predicted, predicted_before)
    assert torch.equal(fresh, fresh_before)
    with pytest.raises(ValueError, match="Unsupported"):
        calculate_speca_error(predicted, fresh, "unknown")
    with pytest.raises(ValueError, match="identical shapes"):
        calculate_speca_error(predicted, fresh[:, :1], "l1")
