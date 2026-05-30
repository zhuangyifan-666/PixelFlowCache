from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


class CachedModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        module_name: str,
        cache_state: RuntimeCacheState,
        policy: FixedIntervalCachePolicy,
    ) -> None:
        super().__init__()
        self.module = module
        self.module_name = module_name
        self.cache_state = cache_state
        self.policy = policy

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self._cache_active_for_current_context():
            output = self.module(*args, **kwargs)
            self.cache_state.mark_disabled(self.module_name)
            return output

        first_tensor = self._first_tensor(args, kwargs)
        batch_signature = self._batch_signature(first_tensor)
        key = self.cache_state.make_key(self.module_name, batch_signature=batch_signature)
        entry = self.cache_state.get(key)
        step_idx = self.cache_state.current_step_idx
        t = self.cache_state.current_t
        cfg_branch = self.cache_state.cfg_branch
        solver_stage = self.cache_state.solver_stage

        if self.policy.should_reuse(step_idx, t, self.module_name, cfg_branch, solver_stage) and entry is not None:
            if self._entry_matches_current_input(entry.tensor, first_tensor):
                entry.hit_count += 1
                self.cache_state.mark_hit(self.module_name)
                return entry.tensor

        output = self.module(*args, **kwargs)
        if not torch.is_tensor(output):
            self.cache_state.mark_disabled(self.module_name)
            return output

        self.cache_state.put(key, output)
        self.cache_state.mark_miss(self.module_name)
        self.cache_state.mark_refresh(self.module_name)
        return output

    def _cache_active_for_current_context(self) -> bool:
        if not self.cache_state.enabled:
            return False
        if not self.policy.enabled:
            return False
        if not self.policy.should_cache_module(self.module_name):
            return False
        if not self.policy.is_branch_enabled(self.cache_state.cfg_branch):
            return False
        if self.cache_state.solver_stage not in self.policy.solver_stages:
            return False
        return True

    @staticmethod
    def _first_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
        for value in args:
            if torch.is_tensor(value):
                return value
        for value in kwargs.values():
            if torch.is_tensor(value):
                return value
        return None

    @staticmethod
    def _batch_signature(tensor: torch.Tensor | None) -> str | None:
        if tensor is None or tensor.ndim == 0:
            return None
        return f"b:{int(tensor.shape[0])}"

    @staticmethod
    def _entry_matches_current_input(cached: torch.Tensor, current: torch.Tensor | None) -> bool:
        if current is None:
            return True
        if cached.device != current.device:
            return False
        if cached.ndim > 0 and current.ndim > 0 and cached.shape[0] != current.shape[0]:
            return False
        return True
