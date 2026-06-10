from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _as_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float()


def l2_norm(tensor: Any) -> float:
    found = _first_tensor(tensor)
    if found is None:
        return float("nan")
    return float(torch.linalg.vector_norm(_as_float_tensor(found)).cpu().item())


def summarize_tensor(tensor: Any, name: str | None = None) -> dict[str, Any]:
    found = _first_tensor(tensor)
    if found is None:
        return {"name": name, "skipped": True, "reason": "no_tensor_found"}
    detached = found.detach()
    x = detached.float()
    if x.numel() == 0:
        mean = std = min_value = max_value = abs_mean_value = l2_value = rms_value = 0.0
        has_nan = has_inf = False
    else:
        has_nan = bool(torch.isnan(x).any().cpu().item())
        has_inf = bool(torch.isinf(x).any().cpu().item())
        finite_mask = torch.isfinite(x)
        finite_x = x[finite_mask] if finite_mask.any() else torch.zeros(1, dtype=x.dtype, device=x.device)
        mean = float(finite_x.mean().cpu().item())
        std = float(finite_x.std(unbiased=False).cpu().item())
        min_value = float(finite_x.min().cpu().item())
        max_value = float(finite_x.max().cpu().item())
        abs_mean_value = float(torch.abs(finite_x).mean().cpu().item())
        l2_value = float(torch.linalg.vector_norm(finite_x).cpu().item())
        rms_value = float(torch.sqrt(torch.mean(finite_x * finite_x)).cpu().item())
    return {
        "name": name,
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "abs_mean": abs_mean_value,
        "l2": l2_value,
        "rms": rms_value,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def relative_l2_delta(current: Any, previous: Any, eps: float = 1e-8) -> float:
    cur = _first_tensor(current)
    prev = _first_tensor(previous)
    if cur is None or prev is None:
        return float("nan")
    cur_f = _as_float_tensor(cur)
    prev_f = _as_float_tensor(prev).to(device=cur_f.device)
    if cur_f.shape != prev_f.shape:
        return float("nan")
    delta = torch.linalg.vector_norm(cur_f - prev_f)
    denom = torch.linalg.vector_norm(prev_f).clamp_min(eps)
    return float((delta / denom).cpu().item())
