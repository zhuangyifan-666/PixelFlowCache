from __future__ import annotations

import math
from typing import Any

import torch

from pfc.profiling.frequency import frequency_delta_bands


def _safe_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def tensor_error_stats(a: torch.Tensor, b: torch.Tensor, name: str | None = None) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"tensor shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    a_float = a.detach().float()
    b_float = b.detach().float().to(device=a_float.device)
    diff = a_float - b_float
    mse = _safe_float(torch.mean(diff.square())) if diff.numel() else 0.0
    mae = _safe_float(torch.mean(torch.abs(diff))) if diff.numel() else 0.0
    rmse = math.sqrt(mse)
    l2_a = _safe_float(torch.linalg.vector_norm(a_float))
    l2_b = _safe_float(torch.linalg.vector_norm(b_float))
    l2_diff = _safe_float(torch.linalg.vector_norm(diff))
    rel_l2 = l2_diff / max(l2_b, 1e-8)
    denom = torch.linalg.vector_norm(a_float.flatten()) * torch.linalg.vector_norm(b_float.flatten())
    cosine = _safe_float(torch.dot(a_float.flatten(), b_float.flatten()) / denom.clamp_min(1e-8))
    return {
        "name": name,
        "shape": [int(dim) for dim in a.shape],
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "cosine": cosine,
        "l2_a": l2_a,
        "l2_b": l2_b,
        "l2_diff": l2_diff,
    }


def image_error_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    stats = tensor_error_stats(a, b, name="image")
    rmse = stats["rmse"]
    stats["psnr"] = float("inf") if rmse == 0 else 20.0 * math.log10(2.0 / rmse)
    return stats


def frequency_error_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return frequency_delta_bands(a, b)
