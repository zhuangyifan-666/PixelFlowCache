from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch


ModelType = Literal["xpred", "vpred"]


@dataclass
class AdapterOutput:
    raw: torch.Tensor
    velocity: torch.Tensor
    x0_pred: Optional[torch.Tensor]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ModelAdapter:
    name: str
    model_type: ModelType

    def __init__(self, name: str, model_type: ModelType) -> None:
        self.name = name
        self.model_type = model_type

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def raw_to_velocity(
        self,
        raw: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | float,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        if self.model_type == "vpred":
            return raw
        if self.model_type != "xpred":
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        t_broadcast = self._broadcast_time(t, x)
        denom = torch.clamp(1.0 - t_broadcast, min=eps)
        return (raw - x) / denom

    def forward_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        cond: Any,
        **kwargs: Any,
    ) -> AdapterOutput:
        eps = kwargs.pop("eps", 1e-4)
        raw = self.forward_raw(x, t, cond, **kwargs)
        velocity = self.raw_to_velocity(raw, x, t, eps=eps)
        x0_pred = raw if self.model_type == "xpred" else None
        return AdapterOutput(raw=raw, velocity=velocity, x0_pred=x0_pred, diagnostics={})

    def get_cache_units(self) -> list[Any]:
        return []

    @staticmethod
    def _broadcast_time(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(t):
            t_tensor = t.to(device=x.device, dtype=x.dtype)
        else:
            t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)

        if t_tensor.ndim == 0:
            return t_tensor

        if t_tensor.ndim == 1:
            if x.ndim == 0:
                return t_tensor
            if t_tensor.shape[0] != x.shape[0]:
                raise ValueError(f"Vector t has batch {t_tensor.shape[0]} but x has batch {x.shape[0]}")
            return t_tensor.reshape(t_tensor.shape[0], *([1] * (x.ndim - 1)))

        if t_tensor.shape == x.shape:
            return t_tensor

        if x.ndim > 0 and t_tensor.shape[0] == x.shape[0] and t_tensor.ndim < x.ndim:
            return t_tensor.reshape(*t_tensor.shape, *([1] * (x.ndim - t_tensor.ndim)))

        return t_tensor
