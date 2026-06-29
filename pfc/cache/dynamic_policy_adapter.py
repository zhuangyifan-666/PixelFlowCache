from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pfc.cache.spectral_dynamic_policy import (
    DynamicCacheDecision,
    RawAccumulatedDistancePolicy,
    SeaCacheSpectralDistancePolicy,
)

DynamicPolicy = RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy


class DynamicPolicyAdapter:
    def __init__(
        self,
        dynamic_policy: DynamicPolicy,
        cache_modules: set[str] | None,
        cache_cond: bool = True,
        cache_uncond: bool = True,
        solver_stages: set[str] | None = None,
    ) -> None:
        self.dynamic_policy = dynamic_policy
        self.cache_modules = set(cache_modules) if cache_modules is not None else None
        self.cache_cond = bool(cache_cond)
        self.cache_uncond = bool(cache_uncond)
        self.solver_stages = set(solver_stages) if solver_stages is not None else {"euler"}
        self.enabled = dynamic_policy.enabled

    def should_cache_module(self, module_name: str) -> bool:
        return self.cache_modules is None or module_name in self.cache_modules

    def is_branch_enabled(self, cfg_branch: str) -> bool:
        if cfg_branch == "cond":
            return self.cache_cond
        if cfg_branch == "uncond":
            return self.cache_uncond
        return True

    def current_decision(self, cfg_branch: str = "global") -> DynamicCacheDecision | None:
        return self.dynamic_policy.current_decision(cfg_branch)

    def is_active(
        self,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
    ) -> bool:
        del step_idx, t
        if not self.enabled:
            return False
        if solver_stage not in self.solver_stages:
            return False
        if not self.should_cache_module(module_name):
            return False
        if not self.is_branch_enabled(cfg_branch):
            return False
        return True

    def should_refresh(
        self,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
    ) -> bool:
        if not self.enabled:
            return True
        if solver_stage not in self.solver_stages:
            return True
        if not self.should_cache_module(module_name):
            return True
        if not self.is_branch_enabled(cfg_branch):
            return True
        decision = self.dynamic_policy.current_decision(cfg_branch)
        if decision is None or decision.step_idx != int(step_idx):
            return True
        return bool(decision.should_refresh)

    def should_reuse(
        self,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
    ) -> bool:
        return not self.should_refresh(step_idx, t, module_name, cfg_branch, solver_stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": "DynamicPolicyAdapter",
            "enabled": self.enabled,
            "cache_modules": sorted(self.cache_modules) if self.cache_modules is not None else None,
            "cache_cond": self.cache_cond,
            "cache_uncond": self.cache_uncond,
            "solver_stages": sorted(self.solver_stages),
            "dynamic_policy": self.dynamic_policy.to_dict(),
        }

    @classmethod
    def from_branches(
        cls,
        dynamic_policy: DynamicPolicy,
        branches: Iterable[str],
        **kwargs: Any,
    ) -> "DynamicPolicyAdapter":
        selected = {branch.strip() for branch in branches if branch.strip()}
        return cls(
            dynamic_policy=dynamic_policy,
            cache_cond="cond" in selected,
            cache_uncond="uncond" in selected,
            **kwargs,
        )
