from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from pfc.core.boundary_spec import BoundarySet, BoundarySpec, PredictionType


class PixelDiffusionModelAdapter(ABC):
    model_name: str = "unknown"
    prediction_type: PredictionType = PredictionType.UNKNOWN
    time_direction: str = "noise_to_image"

    def output_to_velocity(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if self.prediction_type == PredictionType.XPRED:
            t_broadcast = self._broadcast_t(t, x)
            return (output - x) / (1.0 - t_broadcast).clamp_min(eps)
        if self.prediction_type == PredictionType.VPRED:
            return output
        raise NotImplementedError(f"Velocity conversion is not defined for {self.prediction_type.value}")

    @abstractmethod
    def list_boundary_candidates(self, model: Any) -> list[BoundarySpec]:
        raise NotImplementedError

    @abstractmethod
    def default_boundary_set(self, model: Any, preset_name: str | None = None) -> BoundarySet:
        raise NotImplementedError

    @abstractmethod
    def wrap_boundary_set(
        self,
        model: Any,
        boundary_set: BoundarySet,
        cache_state: Any,
        policy: Any,
    ) -> list[str]:
        raise NotImplementedError

    def cache_proxy(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any | None = None) -> torch.Tensor:
        return x

    @abstractmethod
    def branch_mode(self) -> str:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "adapter_class": self.__class__.__name__,
            "model_name": self.model_name,
            "prediction_type": self.prediction_type.value,
            "time_direction": self.time_direction,
            "branch_mode": self.branch_mode(),
        }

    @staticmethod
    def _broadcast_t(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(t):
            t_tensor = t.to(device=x.device, dtype=x.dtype)
        else:
            t_tensor = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t_tensor.ndim == 0:
            return t_tensor
        if t_tensor.ndim == 1:
            if t_tensor.shape[0] != x.shape[0]:
                raise ValueError(f"Vector t has batch {t_tensor.shape[0]}, expected {x.shape[0]}")
            return t_tensor.reshape((t_tensor.shape[0],) + (1,) * (x.ndim - 1))
        while t_tensor.ndim < x.ndim:
            t_tensor = t_tensor.unsqueeze(-1)
        return t_tensor
