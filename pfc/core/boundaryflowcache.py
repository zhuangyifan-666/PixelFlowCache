from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pfc.cache.cache_state import RuntimeCacheState
from pfc.core.boundary_spec import BoundarySet, PredictionType
from pfc.core.model_adapter import PixelDiffusionModelAdapter


@dataclass(frozen=True)
class BoundaryFlowCacheConfig:
    model_name: str
    prediction_type: PredictionType
    boundary_set: BoundarySet
    scheduler_name: str
    scheduler_config: dict[str, Any]
    parameterization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prediction_type": self.prediction_type.value,
            "boundary_set": self.boundary_set.to_dict(),
            "scheduler_name": self.scheduler_name,
            "scheduler_config": dict(self.scheduler_config),
            "parameterization": dict(self.parameterization),
        }


@dataclass
class BoundaryFlowCacheRuntime:
    adapter: PixelDiffusionModelAdapter
    boundary_set: BoundarySet
    cache_state: RuntimeCacheState
    scheduler: Any
    wrapped_modules: list[str] = field(default_factory=list)

    def install(self, model: Any) -> list[str]:
        self.wrapped_modules = self.adapter.wrap_boundary_set(
            model,
            self.boundary_set,
            self.cache_state,
            self.scheduler,
        )
        return list(self.wrapped_modules)

    def clear_batch(self) -> None:
        self.cache_state.clear_entries()
        if hasattr(self.scheduler, "clear_batch"):
            self.scheduler.clear_batch()

    def cache_summary(self) -> dict[str, Any]:
        scheduler_summary = self.scheduler.summary() if hasattr(self.scheduler, "summary") else None
        return {
            "adapter": self.adapter.describe(),
            "boundary_set": self.boundary_set.to_dict(),
            "wrapped_modules": list(self.wrapped_modules),
            "cache_state": self.cache_state.summary(),
            "scheduler": scheduler_summary,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.cache_summary()
