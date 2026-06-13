from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


class CacheScheduler(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_batch(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_context(
        self,
        step_idx: int,
        t: float,
        branch: str,
        solver_stage: str = "euler",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_refresh(self, module_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def should_reuse(self, module_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


class FixedWindowScheduler(CacheScheduler):
    def __init__(
        self,
        interval: int,
        active_t_min: float | None = None,
        active_t_max: float | None = None,
        active_step_min: int | None = None,
        active_step_max: int | None = None,
        active_window_warmup_refreshes: int = 0,
        cache_cond: bool = True,
        cache_uncond: bool = True,
        solver_stages: set[str] | None = None,
    ) -> None:
        self.interval = int(interval)
        self.active_t_min = active_t_min
        self.active_t_max = active_t_max
        self.active_step_min = active_step_min
        self.active_step_max = active_step_max
        self.active_window_warmup_refreshes = int(active_window_warmup_refreshes)
        self.cache_cond = bool(cache_cond)
        self.cache_uncond = bool(cache_uncond)
        self.solver_stages = set(solver_stages) if solver_stages is not None else {"euler"}
        self._policy = self.to_policy_adapter(cache_modules=None)
        self._step_idx = 0
        self._t = 0.0
        self._branch = "global"
        self._solver_stage = "euler"

    def reset(self) -> None:
        self._policy = self.to_policy_adapter(cache_modules=self._policy.cache_modules)

    def clear_batch(self) -> None:
        self.reset()

    def set_context(
        self,
        step_idx: int,
        t: float,
        branch: str,
        solver_stage: str = "euler",
    ) -> None:
        self._step_idx = int(step_idx)
        self._t = float(t)
        self._branch = str(branch)
        self._solver_stage = str(solver_stage)

    def should_refresh(self, module_name: str) -> bool:
        return self._policy.should_refresh(
            self._step_idx,
            self._t,
            module_name,
            self._branch,
            self._solver_stage,
        )

    def should_reuse(self, module_name: str) -> bool:
        return self._policy.should_reuse(
            self._step_idx,
            self._t,
            module_name,
            self._branch,
            self._solver_stage,
        )

    def to_policy_adapter(self, cache_modules: set[str] | None = None) -> FixedIntervalCachePolicy:
        return FixedIntervalCachePolicy(
            enabled=True,
            interval=self.interval,
            cache_modules=cache_modules,
            active_t_min=self.active_t_min,
            active_t_max=self.active_t_max,
            active_step_min=self.active_step_min,
            active_step_max=self.active_step_max,
            active_window_warmup_refreshes=self.active_window_warmup_refreshes,
            cache_cond=self.cache_cond,
            cache_uncond=self.cache_uncond,
            solver_stages=set(self.solver_stages),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "scheduler": "fixed_window",
            "current_context": {
                "step_idx": self._step_idx,
                "t": self._t,
                "branch": self._branch,
                "solver_stage": self._solver_stage,
            },
            **self.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler": "fixed_window",
            "interval": self.interval,
            "active_t_min": self.active_t_min,
            "active_t_max": self.active_t_max,
            "active_step_min": self.active_step_min,
            "active_step_max": self.active_step_max,
            "active_window_warmup_refreshes": self.active_window_warmup_refreshes,
            "cache_cond": self.cache_cond,
            "cache_uncond": self.cache_uncond,
            "solver_stages": sorted(self.solver_stages),
        }
