from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn.functional as F


SpeCaErrorMetric = Literal["l1", "l2", "relative_l1", "relative_l2", "cosine_error"]
SPECA_ERROR_METRICS: tuple[SpeCaErrorMetric, ...] = (
    "l1",
    "l2",
    "relative_l1",
    "relative_l2",
    "cosine_error",
)


@overload
def calculate_speca_error(
    predicted: torch.Tensor,
    fresh: torch.Tensor,
    metric: str,
    eps: float = 1e-10,
    *,
    return_per_sample: Literal[False] = False,
) -> float: ...


@overload
def calculate_speca_error(
    predicted: torch.Tensor,
    fresh: torch.Tensor,
    metric: str,
    eps: float = 1e-10,
    *,
    return_per_sample: Literal[True],
) -> tuple[float, torch.Tensor]: ...


def calculate_speca_error(
    predicted: torch.Tensor,
    fresh: torch.Tensor,
    metric: str,
    eps: float = 1e-10,
    *,
    return_per_sample: bool = False,
) -> float | tuple[float, torch.Tensor]:
    """Calculate a SpeCa verification error with float32 accumulation.

    The returned scalar is detached and JSON-ready.  When requested, the
    per-sample tensor is also detached and kept on the input device.
    """

    if predicted.shape != fresh.shape:
        raise ValueError(
            "predicted and fresh must have identical shapes, "
            f"got {tuple(predicted.shape)} and {tuple(fresh.shape)}"
        )
    if metric not in SPECA_ERROR_METRICS:
        raise ValueError(f"Unsupported SpeCa error metric: {metric}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    predicted_f32 = predicted.float()
    fresh_f32 = fresh.float()
    difference = predicted_f32 - fresh_f32
    sample_dims = tuple(range(1, predicted_f32.ndim))

    if metric == "l1":
        value = difference.abs().mean()
        per_sample = difference.abs().mean(dim=sample_dims) if sample_dims else difference.abs()
    elif metric == "l2":
        value = difference.square().mean().sqrt()
        per_sample = (
            difference.square().mean(dim=sample_dims).sqrt()
            if sample_dims
            else difference.abs()
        )
    elif metric in {"relative_l1", "relative_l2"}:
        relative = difference.abs() / (fresh_f32.abs() + float(eps))
        if metric == "relative_l1":
            value = relative.mean()
            per_sample = relative.mean(dim=sample_dims) if sample_dims else relative
        else:
            value = relative.square().mean().sqrt()
            per_sample = (
                relative.square().mean(dim=sample_dims).sqrt()
                if sample_dims
                else relative
            )
    else:
        if predicted_f32.ndim <= 1:
            predicted_flat = predicted_f32.reshape(1, -1)
            fresh_flat = fresh_f32.reshape(1, -1)
        else:
            predicted_flat = predicted_f32.flatten(start_dim=1)
            fresh_flat = fresh_f32.flatten(start_dim=1)
        cosine = F.cosine_similarity(
            predicted_flat,
            fresh_flat,
            dim=1,
            eps=float(eps),
        )
        both_zero = (predicted_flat.norm(dim=1) <= float(eps)) & (
            fresh_flat.norm(dim=1) <= float(eps)
        )
        cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
        per_sample = (1.0 - cosine).clamp(min=0.0, max=2.0)
        numerical_zero = 8.0 * torch.finfo(torch.float32).eps
        per_sample = torch.where(
            per_sample <= numerical_zero,
            torch.zeros_like(per_sample),
            per_sample,
        )
        value = per_sample.mean()

    scalar = float(value.detach().cpu().item())
    if return_per_sample:
        return scalar, per_sample.detach()
    return scalar
