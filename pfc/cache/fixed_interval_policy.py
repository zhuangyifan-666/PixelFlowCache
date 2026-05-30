from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class FixedIntervalCachePolicy:
    def __init__(
        self,
        enabled: bool = True,
        interval: int = 2,
        cache_modules: set[str] | None = None,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        max_steps: int | None = None,
        refresh_first_step: bool = True,
        cache_cond: bool = True,
        cache_uncond: bool = True,
        solver_stages: set[str] | None = None,
        active_t_min: float | None = None,
        active_t_max: float | None = None,
        active_step_min: int | None = None,
        active_step_max: int | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        self.enabled = enabled
        self.interval = interval
        self.cache_modules = set(cache_modules) if cache_modules is not None else None
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.max_steps = max_steps
        self.refresh_first_step = refresh_first_step
        self.cache_cond = cache_cond
        self.cache_uncond = cache_uncond
        self.solver_stages = set(solver_stages) if solver_stages is not None else {"euler"}
        self.active_t_min = active_t_min
        self.active_t_max = active_t_max
        self.active_step_min = active_step_min
        self.active_step_max = active_step_max

    def should_cache_module(self, module_name: str) -> bool:
        return self.cache_modules is None or module_name in self.cache_modules

    def is_branch_enabled(self, cfg_branch: str) -> bool:
        if cfg_branch == "cond":
            return self.cache_cond
        if cfg_branch == "uncond":
            return self.cache_uncond
        return True

    def _active_for_context(
        self,
        step_idx: int,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        t: float | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        if not self.should_cache_module(module_name):
            return False
        if not self.is_branch_enabled(cfg_branch):
            return False
        if solver_stage not in self.solver_stages:
            return False
        if step_idx < self.warmup_steps:
            return False
        if self.max_steps is not None and step_idx >= self.max_steps:
            return False
        if self.cooldown_steps and self.max_steps is not None and step_idx >= self.max_steps - self.cooldown_steps:
            return False
        if t is not None:
            if self.active_t_min is not None and t < self.active_t_min:
                return False
            if self.active_t_max is not None and t >= self.active_t_max:
                return False
        if self.active_step_min is not None and step_idx < self.active_step_min:
            return False
        if self.active_step_max is not None and step_idx >= self.active_step_max:
            return False
        return True

    def is_active(
        self,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
    ) -> bool:
        return self._active_for_context(step_idx, module_name, cfg_branch, solver_stage, t=t)

    def should_refresh(
        self,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
    ) -> bool:
        if not self._active_for_context(step_idx, module_name, cfg_branch, solver_stage, t=t):
            return True
        if self.interval == 1:
            return True
        if self.refresh_first_step and step_idx == 0:
            return True
        return step_idx % self.interval == 0

    def should_reuse(
        self,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
    ) -> bool:
        return (
            self._active_for_context(step_idx, module_name, cfg_branch, solver_stage, t=t)
            and not self.should_refresh(step_idx, t, module_name, cfg_branch, solver_stage)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "cache_modules": sorted(self.cache_modules) if self.cache_modules is not None else None,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "max_steps": self.max_steps,
            "refresh_first_step": self.refresh_first_step,
            "cache_cond": self.cache_cond,
            "cache_uncond": self.cache_uncond,
            "solver_stages": sorted(self.solver_stages),
            "active_t_min": self.active_t_min,
            "active_t_max": self.active_t_max,
            "active_step_min": self.active_step_min,
            "active_step_max": self.active_step_max,
        }

    @classmethod
    def from_branches(
        cls,
        branches: Iterable[str],
        **kwargs: Any,
    ) -> "FixedIntervalCachePolicy":
        selected = {branch.strip() for branch in branches if branch.strip()}
        return cls(
            cache_cond="cond" in selected,
            cache_uncond="uncond" in selected,
            **kwargs,
        )
