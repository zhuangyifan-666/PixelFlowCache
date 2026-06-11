from __future__ import annotations

import torch
import torch.nn.functional as F


def proxy_from_image_state(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW image/state tensor, got shape {tuple(x.shape)}")
    return x.detach()


def maybe_downsample_proxy(x: torch.Tensor, max_size: int = 64) -> torch.Tensor:
    if max_size <= 0:
        return x.detach()
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW proxy tensor, got shape {tuple(x.shape)}")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if max(height, width) <= max_size:
        return x.detach()
    scale = max_size / float(max(height, width))
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return F.interpolate(x.detach().float(), size=(out_h, out_w), mode="bilinear", align_corners=False).to(
        dtype=x.dtype
    )


def normalize_proxy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW proxy tensor, got shape {tuple(x.shape)}")
    work = x.detach().float()
    dims = tuple(range(1, work.ndim))
    mean = work.mean(dim=dims, keepdim=True)
    std = work.std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)
    return ((work - mean) / std).to(dtype=x.dtype)
