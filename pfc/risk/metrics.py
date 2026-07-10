from __future__ import annotations

import math
from typing import Any

import torch


def solver_scaled_rms_tensor(
    candidate_next_state: torch.Tensor,
    fresh_next_state: torch.Tensor,
    current_state: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> torch.Tensor:
    if atol <= 0.0 or rtol < 0.0:
        raise ValueError("risk tolerances require atol > 0 and rtol >= 0")
    candidate = candidate_next_state.float()
    fresh = fresh_next_state.float()
    current = current_state.float()
    delta = candidate - fresh
    scale = float(atol) + float(rtol) * torch.maximum(current.abs(), fresh.abs())
    value = torch.sqrt(torch.mean(torch.square(delta / scale.clamp_min(float(atol)))))
    return torch.nan_to_num(value, nan=0.0, posinf=torch.finfo(torch.float32).max, neginf=0.0)


def relative_l2_tensor(numerator: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    numerator_f32 = numerator.float()
    reference_f32 = reference.float()
    value = torch.linalg.vector_norm(numerator_f32.reshape(-1)) / (
        torch.linalg.vector_norm(reference_f32.reshape(-1)) + float(eps)
    )
    return torch.nan_to_num(value, nan=0.0, posinf=torch.finfo(torch.float32).max, neginf=0.0)


def transition_relative_l2_tensor(
    candidate_next_state: torch.Tensor,
    fresh_next_state: torch.Tensor,
    current_state: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    return relative_l2_tensor(
        (candidate_next_state - current_state) - (fresh_next_state - current_state),
        fresh_next_state - current_state,
        eps,
    )


def tensor_scalar(value: torch.Tensor | float | int) -> float:
    scalar = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"metric is not finite: {scalar}")
    return scalar


def equivalence_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    shape_match = tuple(candidate.shape) == tuple(reference.shape)
    dtype_match = candidate.dtype == reference.dtype
    if not shape_match:
        finite_failure = torch.finfo(torch.float32).max
        return {
            "max_abs": float(finite_failure),
            "mean_abs": float(finite_failure),
            "relative_l2": float(finite_failure),
            "allclose": False,
            "shape_match": False,
            "dtype_match": dtype_match,
        }
    difference = candidate.float() - reference.float()
    return {
        "max_abs": tensor_scalar(difference.abs().max()) if difference.numel() else 0.0,
        "mean_abs": tensor_scalar(difference.abs().mean()) if difference.numel() else 0.0,
        "relative_l2": tensor_scalar(relative_l2_tensor(difference, reference)),
        "allclose": bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol)),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
    }


def l2_scalar(value: torch.Tensor) -> float:
    return tensor_scalar(torch.linalg.vector_norm(value.float().reshape(-1)))
