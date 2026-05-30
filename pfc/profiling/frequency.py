from __future__ import annotations

from typing import Any

import torch

from pfc.profiling.tensor_stats import relative_l2_delta


def _empty_record(x: Any, reason: str) -> dict[str, Any]:
    shape = list(x.shape) if torch.is_tensor(x) else []
    return {"skipped": True, "reason": reason, "shape": [int(dim) for dim in shape]}


def fft_frequency_bands(x: torch.Tensor, low_ratio: float = 0.15, high_ratio: float = 0.45) -> dict[str, Any]:
    if not torch.is_tensor(x):
        return _empty_record(x, "input_is_not_tensor")
    if x.ndim != 4:
        return _empty_record(x, "expected_bchw")
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return _empty_record(x, "spatial_size_too_small")

    data = x.detach().float()
    _, _, height, width = data.shape
    spectrum = torch.fft.fftshift(torch.fft.fft2(data, dim=(-2, -1)), dim=(-2, -1))
    energy = spectrum.real.square() + spectrum.imag.square()

    yy = torch.linspace(-0.5, 0.5, height, device=data.device)
    xx = torch.linspace(-0.5, 0.5, width, device=data.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square())
    max_radius = float(torch.sqrt(torch.tensor(0.5, device=data.device)).item())
    radius = radius / max_radius

    low_mask = radius <= low_ratio
    high_mask = radius > high_ratio
    mid_mask = (~low_mask) & (~high_mask)

    low_energy = float(energy[..., low_mask].sum().cpu().item())
    mid_energy = float(energy[..., mid_mask].sum().cpu().item())
    high_energy = float(energy[..., high_mask].sum().cpu().item())
    total_energy = low_energy + mid_energy + high_energy
    denom = max(total_energy, 1e-12)
    return {
        "shape": [int(dim) for dim in data.shape],
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "total_energy": total_energy,
        "low_ratio": low_energy / denom,
        "mid_ratio": mid_energy / denom,
        "high_ratio": high_energy / denom,
        "high_to_low": high_energy / max(low_energy, 1e-12),
        "low_cutoff_ratio": low_ratio,
        "high_cutoff_ratio": high_ratio,
    }


def frequency_delta_bands(
    current: torch.Tensor,
    previous: torch.Tensor,
    low_ratio: float = 0.15,
    high_ratio: float = 0.45,
) -> dict[str, Any]:
    if not torch.is_tensor(current) or not torch.is_tensor(previous):
        return _empty_record(current, "input_is_not_tensor")
    if current.shape != previous.shape:
        return _empty_record(current, "shape_mismatch")
    delta = current.detach().float() - previous.detach().float().to(device=current.device)
    record = fft_frequency_bands(delta, low_ratio=low_ratio, high_ratio=high_ratio)
    record["rel_l2_delta"] = relative_l2_delta(current, previous)
    return record

