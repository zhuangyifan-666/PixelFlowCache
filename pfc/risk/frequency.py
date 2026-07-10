from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pfc.risk.metrics import relative_l2_tensor, tensor_scalar


@dataclass
class RadialFrequencyRisk:
    low_ratio: float = 0.15
    high_ratio: float = 0.45
    eps: float = 1e-12
    _mask_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.low_ratio < self.high_ratio < 1.0:
            raise ValueError("frequency ratios must satisfy 0 < low < high < 1")

    @property
    def cache_size(self) -> int:
        return len(self._mask_cache)

    def risks(self, delta_transition: torch.Tensor, fresh_update: torch.Tensor) -> dict[str, float]:
        delta_low, delta_mid, delta_high = self.split(delta_transition)
        fresh_low, fresh_mid, fresh_high = self.split(fresh_update)
        return {
            "risk_low": tensor_scalar(relative_l2_tensor(delta_low, fresh_low, self.eps)),
            "risk_mid": tensor_scalar(relative_l2_tensor(delta_mid, fresh_mid, self.eps)),
            "risk_high": tensor_scalar(relative_l2_tensor(delta_high, fresh_high, self.eps)),
        }

    def split(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if value.ndim < 2:
            raise ValueError("frequency risk expects spatial dimensions")
        source = value.float()
        height, width = int(source.shape[-2]), int(source.shape[-1])
        low, mid, high = self._masks(height, width, source.device)
        spectrum = torch.fft.fft2(source, dim=(-2, -1), norm="ortho")
        return tuple(
            torch.fft.ifft2(spectrum * mask, dim=(-2, -1), norm="ortho").real
            for mask in (low, mid, high)
        )

    def _masks(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (height, width, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(height, device=device).reshape(height, 1)
        fx = torch.fft.fftfreq(width, device=device).reshape(1, width)
        radius = torch.sqrt(fy.square() + fx.square())
        maximum = radius.max().clamp_min(torch.finfo(torch.float32).eps)
        normalized = radius / maximum
        low = (normalized <= self.low_ratio).to(torch.float32)
        high = (normalized >= self.high_ratio).to(torch.float32)
        mid = ((normalized > self.low_ratio) & (normalized < self.high_ratio)).to(torch.float32)
        cached = (low, mid, high)
        self._mask_cache[key] = cached
        return cached
