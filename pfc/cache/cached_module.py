from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from pfc.cache.cache_state import CacheEntry, InputSignature, RuntimeCacheState


@dataclass
class ReuseProcessingResult:
    output_tensor: torch.Tensor
    verification_performed: bool = False
    verification_error: float | None = None
    used_fresh_output: bool = False


class CachedModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        module_name: str,
        cache_state: RuntimeCacheState,
        policy: Any,
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
        input_signature = self._input_signature(first_tensor)
        batch_signature = self._batch_signature(first_tensor)
        key = self.cache_state.make_key(self.module_name, batch_signature=batch_signature)
        entry = self.cache_state.get(key)
        step_idx = self.cache_state.current_step_idx
        t = self.cache_state.current_t
        cfg_branch = self.cache_state.cfg_branch
        solver_stage = self.cache_state.solver_stage

        can_reuse = self._should_reuse_entry_aware(step_idx, t, cfg_branch, solver_stage, entry)
        entry_matches = False
        if can_reuse and entry is not None and self._entry_matches_input_signature(
            entry,
            input_signature,
        ):
            reuse_tensor = self._make_reuse_tensor(step_idx, t, cfg_branch, solver_stage, entry, first_tensor)
            if reuse_tensor is not None:
                entry_matches = self._reuse_tensor_matches(reuse_tensor, entry.tensor, first_tensor)
                if entry_matches:
                    processing = self._process_reuse_tensor(
                        step_idx=step_idx,
                        t=t,
                        cfg_branch=cfg_branch,
                        solver_stage=solver_stage,
                        entry=entry,
                        first_tensor=first_tensor,
                        reuse_tensor=reuse_tensor,
                        args=args,
                        kwargs=kwargs,
                    )
                    reuse_tensor = processing.output_tensor
                    if not self._reuse_tensor_matches(reuse_tensor, entry.tensor, first_tensor):
                        entry_matches = False
                    else:
                        entry.hit_count += 1
                        self.cache_state.mark_hit(self.module_name)
                        if hasattr(self.policy, "mark_reuse_committed"):
                            self.policy.mark_reuse_committed(
                                step_idx=step_idx,
                                t=t,
                                module_name=self.module_name,
                                cfg_branch=cfg_branch,
                                solver_stage=solver_stage,
                                entry=entry,
                            )
                        if hasattr(self.policy, "on_reuse_committed"):
                            self.policy.on_reuse_committed(
                                step_idx=step_idx,
                                t=t,
                                module_name=self.module_name,
                                cfg_branch=cfg_branch,
                                solver_stage=solver_stage,
                                entry=entry,
                                tensor=reuse_tensor,
                            )
                        return reuse_tensor

        measure_full = hasattr(self.policy, "mark_full_compute_host_dispatch_time")
        full_started = time.perf_counter() if measure_full else 0.0
        output = self.module(*args, **kwargs)
        if measure_full:
            self.policy.mark_full_compute_host_dispatch_time(time.perf_counter() - full_started)
        if not torch.is_tensor(output):
            self.cache_state.mark_disabled(self.module_name)
            return output

        refreshed_entry = self.cache_state.put(
            key,
            output,
            input_signature=input_signature,
        )
        self.cache_state.mark_miss(self.module_name)
        self.cache_state.mark_refresh(self.module_name)
        if hasattr(self.policy, "mark_refresh_committed"):
            self.policy.mark_refresh_committed(
                step_idx=step_idx,
                t=t,
                module_name=self.module_name,
                cfg_branch=cfg_branch,
                solver_stage=solver_stage,
                entry=entry,
                refreshed_entry=refreshed_entry,
                entry_matches=entry_matches if can_reuse else None,
            )
        if hasattr(self.policy, "on_refresh_committed"):
            self.policy.on_refresh_committed(
                step_idx=step_idx,
                t=t,
                module_name=self.module_name,
                cfg_branch=cfg_branch,
                solver_stage=solver_stage,
                entry=entry,
                refreshed_entry=refreshed_entry,
                entry_matches=entry_matches if can_reuse else None,
                tensor=output,
            )
        return output

    def _cache_active_for_current_context(self) -> bool:
        if not self.cache_state.enabled:
            return False
        if not self.policy.enabled:
            return False
        return self.policy.is_active(
            self.cache_state.current_step_idx,
            self.cache_state.current_t,
            self.module_name,
            self.cache_state.cfg_branch,
            self.cache_state.solver_stage,
        )

    def _should_reuse_entry_aware(
        self,
        step_idx: int,
        t: float,
        cfg_branch: str,
        solver_stage: str,
        entry: Any | None,
    ) -> bool:
        if hasattr(self.policy, "should_reuse_entry"):
            return bool(
                self.policy.should_reuse_entry(
                    step_idx=step_idx,
                    t=t,
                    module_name=self.module_name,
                    cfg_branch=cfg_branch,
                    solver_stage=solver_stage,
                    entry=entry,
                )
            )
        if entry is None:
            return False
        return bool(self.policy.should_reuse(step_idx, t, self.module_name, cfg_branch, solver_stage))

    def _make_reuse_tensor(
        self,
        step_idx: int,
        t: float,
        cfg_branch: str,
        solver_stage: str,
        entry: Any,
        first_tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if hasattr(self.policy, "make_reuse_tensor"):
            candidate = self.policy.make_reuse_tensor(
                step_idx=step_idx,
                t=t,
                module_name=self.module_name,
                cfg_branch=cfg_branch,
                solver_stage=solver_stage,
                entry=entry,
                current_input=first_tensor,
            )
            return candidate if torch.is_tensor(candidate) else None
        return entry.tensor

    def _process_reuse_tensor(
        self,
        *,
        step_idx: int,
        t: float,
        cfg_branch: str,
        solver_stage: str,
        entry: Any,
        first_tensor: torch.Tensor | None,
        reuse_tensor: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReuseProcessingResult:
        hook = getattr(self.policy, "process_reuse_tensor", None)
        if hook is None:
            return ReuseProcessingResult(output_tensor=reuse_tensor)
        result = hook(
            step_idx=step_idx,
            t=t,
            module_name=self.module_name,
            cfg_branch=cfg_branch,
            solver_stage=solver_stage,
            entry=entry,
            current_input=first_tensor,
            reuse_tensor=reuse_tensor,
            fresh_compute=lambda: self.module(*args, **kwargs),
        )
        if isinstance(result, ReuseProcessingResult):
            return result
        if torch.is_tensor(result):
            return ReuseProcessingResult(output_tensor=result)
        output_tensor = getattr(result, "output_tensor", None)
        if not torch.is_tensor(output_tensor):
            raise TypeError("process_reuse_tensor must return a tensor or ReuseProcessingResult")
        return ReuseProcessingResult(
            output_tensor=output_tensor,
            verification_performed=bool(getattr(result, "verification_performed", False)),
            verification_error=getattr(result, "verification_error", None),
            used_fresh_output=bool(getattr(result, "used_fresh_output", False)),
        )

    @staticmethod
    def _reuse_tensor_matches(
        reuse_tensor: torch.Tensor,
        cached_tensor: torch.Tensor,
        current: torch.Tensor | None,
    ) -> bool:
        if reuse_tensor.shape != cached_tensor.shape:
            return False
        if current is not None and reuse_tensor.device != current.device:
            return False
        if current is not None and reuse_tensor.dtype != current.dtype:
            return False
        if reuse_tensor.device != cached_tensor.device:
            return False
        if reuse_tensor.dtype != cached_tensor.dtype:
            return False
        if current is not None and reuse_tensor.ndim > 0 and current.ndim > 0 and reuse_tensor.shape[0] != current.shape[0]:
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

    def _input_signature(self, tensor: torch.Tensor | None) -> InputSignature | None:
        if tensor is None:
            return None
        return InputSignature.from_tensor(
            tensor,
            session_id=self.cache_state.session_id,
        )

    @staticmethod
    def _entry_matches_input_signature(
        entry: CacheEntry,
        current: InputSignature | None,
    ) -> bool:
        if entry.input_signature is None:
            return current is None
        if current is None or entry.input_signature != current:
            return False
        if entry.output_shape is not None and tuple(entry.tensor.shape) != entry.output_shape:
            return False
        if entry.output_dtype is not None and str(entry.tensor.dtype) != entry.output_dtype:
            return False
        if entry.output_device is not None and str(entry.tensor.device) != entry.output_device:
            return False
        return True

    @staticmethod
    def _entry_matches_current_input(cached: torch.Tensor, current: torch.Tensor | None) -> bool:
        if current is None:
            return True
        if cached.device != current.device:
            return False
        if cached.ndim > 0 and current.ndim > 0 and cached.shape[0] != current.shape[0]:
            return False
        return True
