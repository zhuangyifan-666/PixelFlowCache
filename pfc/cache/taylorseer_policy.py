from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from pfc.cache.safe_map_policy import canonical_module_name


HistoryKey = tuple[str, str, str, str | None]
HistoryItem = tuple[int, torch.Tensor]


class TaylorSeerCachePolicy:
    policy_name = "TaylorSeerCachePolicy"

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_name: str = "JiT",
        cache_modules: Iterable[str] | None = None,
        interval: int = 4,
        max_order: int = 4,
        min_history: int = 2,
        max_history: int | None = None,
        solver_stages: Iterable[str] | None = None,
        branches: Iterable[str] | None = None,
        fallback_to_global_branch: bool = True,
        forecast_on_steps: Iterable[int] | None = None,
        refresh_first_n_steps: int = 1,
        refresh_last_n_steps: int = 0,
        disable_final_step: bool = False,
        clone_forecast: bool = False,
        debug_jsonl_path: Path | str | None = None,
        total_steps: int | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if min_history <= 0:
            raise ValueError("min_history must be positive")
        resolved_max_history = max_history if max_history is not None else max_order + 2
        if resolved_max_history <= 0:
            raise ValueError("max_history must be positive")
        if refresh_first_n_steps < 0:
            raise ValueError("refresh_first_n_steps must be non-negative")
        if refresh_last_n_steps < 0:
            raise ValueError("refresh_last_n_steps must be non-negative")
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps must be positive when provided")

        self.enabled = bool(enabled)
        self.model_name = str(model_name)
        self.cache_modules = {canonical_module_name(str(item)) for item in cache_modules} if cache_modules is not None else None
        self.interval = int(interval)
        self.max_order = int(max_order)
        self.min_history = int(min_history)
        self.max_history = int(resolved_max_history)
        self.solver_stages = {str(item) for item in (solver_stages or {"euler"})}
        self.branches = {str(item) for item in (branches or {"cond", "uncond", "global"})}
        self.fallback_to_global_branch = bool(fallback_to_global_branch)
        self.forecast_on_steps = {int(item) for item in forecast_on_steps} if forecast_on_steps is not None else None
        self.refresh_first_n_steps = int(refresh_first_n_steps)
        self.refresh_last_n_steps = int(refresh_last_n_steps)
        self.disable_final_step = bool(disable_final_step)
        self.clone_forecast = bool(clone_forecast)
        self.debug_jsonl_path = Path(debug_jsonl_path) if debug_jsonl_path is not None else None
        self.total_steps = int(total_steps) if total_steps is not None else None

        self._history: dict[HistoryKey, list[HistoryItem]] = defaultdict(list)
        self._pending_forecasts: dict[tuple[HistoryKey, int], dict[str, Any]] = {}
        self._stats = self._empty_stats()
        self._order_values: list[int] = []
        self._by_module: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_branch: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_step: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_order: defaultdict[int, int] = defaultdict(int)

    def should_cache_module(self, module_name: str) -> bool:
        canonical = canonical_module_name(str(module_name))
        return self.cache_modules is None or canonical in self.cache_modules

    def is_active(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        del step_idx, t
        if not self.enabled:
            return False
        if not self.should_cache_module(module_name):
            return False
        if str(solver_stage) not in self.solver_stages:
            return False
        _branch, ok = self._select_branch(str(cfg_branch))
        return ok

    def should_refresh(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        return not self.should_reuse(
            step_idx=step_idx,
            t=t,
            module_name=module_name,
            cfg_branch=cfg_branch,
            solver_stage=solver_stage,
        )

    def should_reuse(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        del t
        if not self.is_active(step_idx, 0.0, module_name, cfg_branch, solver_stage):
            return False
        canonical = canonical_module_name(str(module_name))
        branch, _ok = self._select_branch(str(cfg_branch))
        key = self._make_key(canonical, branch, str(solver_stage), None)
        return self._decision_for_key(int(step_idx), key)[0]

    def should_reuse_entry(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any | None,
    ) -> bool:
        canonical = canonical_module_name(str(module_name))
        branch, branch_ok = self._select_branch(str(cfg_branch))
        active = self.enabled and self.should_cache_module(canonical) and str(solver_stage) in self.solver_stages and branch_ok
        if not active:
            return False

        key = self._make_key(canonical, branch, str(solver_stage), self._batch_signature(getattr(entry, "tensor", None)))
        self._stats["total_managed_calls"] += 1
        if entry is None:
            decision = False
            reason = "insufficient_history"
            stat_reason = "insufficient_history_refresh"
            effective_order = 0
        else:
            decision, reason, stat_reason, effective_order = self._decision_for_key(int(step_idx), key)
        history_steps = self._history_steps(key)
        if decision:
            self._pending_forecasts[(key, int(step_idx))] = {
                "effective_order": effective_order,
                "history_steps": history_steps,
            }
        self._record_decision(
            step_idx=int(step_idx),
            module_name=canonical,
            branch=branch,
            solver_stage=str(solver_stage),
            decision="forecast" if decision else "refresh",
            reason=reason,
            stat_reason=stat_reason,
            history_steps=history_steps,
            effective_order=effective_order,
        )
        return bool(decision)

    def make_reuse_tensor(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any | None,
        current_input: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del t
        canonical = canonical_module_name(str(module_name))
        branch, branch_ok = self._select_branch(str(cfg_branch))
        if not branch_ok or entry is None:
            self._stats["forecast_failures"] += 1
            return None
        signature = self._batch_signature(current_input) or self._batch_signature(getattr(entry, "tensor", None))
        key = self._make_key(canonical, branch, str(solver_stage), signature)
        history = self._history.get(key, [])
        if len(history) < self.min_history:
            self._stats["forecast_failures"] += 1
            return None
        effective_order = min(self.max_order, len(history) - 1)
        points = history[-(effective_order + 1) :]
        forecast = self._lagrange_extrapolate(points, int(step_idx))
        if forecast is None:
            self._stats["forecast_failures"] += 1
            return None
        return forecast.clone() if self.clone_forecast else forecast

    def on_refresh_committed(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any | None,
        refreshed_entry: Any | None = None,
        tensor: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        del t, entry, refreshed_entry, kwargs
        if not torch.is_tensor(tensor):
            return
        canonical = canonical_module_name(str(module_name))
        branch, branch_ok = self._select_branch(str(cfg_branch))
        if not branch_ok:
            return
        key = self._make_key(canonical, branch, str(solver_stage), self._batch_signature(tensor))
        history = self._history[key]
        item = (int(step_idx), tensor.detach().clone())
        if history and history[-1][0] == int(step_idx):
            history[-1] = item
        else:
            history.append(item)
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]
        self._stats["history_appends"] += 1

    def on_reuse_committed(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any,
        tensor: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        del t, tensor, kwargs
        canonical = canonical_module_name(str(module_name))
        branch, branch_ok = self._select_branch(str(cfg_branch))
        if not branch_ok:
            return
        key = self._make_key(canonical, branch, str(solver_stage), self._batch_signature(getattr(entry, "tensor", None)))
        pending = self._pending_forecasts.pop((key, int(step_idx)), {})
        effective_order = int(pending.get("effective_order", 0))
        self._stats["forecast_committed"] += 1
        self._order_values.append(effective_order)
        self._by_order[effective_order] += 1

    def clear_batch(self) -> None:
        self._history.clear()
        self._pending_forecasts.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "enabled": self.enabled,
            "model_name": self.model_name,
            "cache_modules": sorted(self.cache_modules) if self.cache_modules is not None else None,
            "interval": self.interval,
            "max_order": self.max_order,
            "min_history": self.min_history,
            "max_history": self.max_history,
            "solver_stages": sorted(self.solver_stages),
            "branches": sorted(self.branches),
            "fallback_to_global_branch": self.fallback_to_global_branch,
            "forecast_on_steps": sorted(self.forecast_on_steps) if self.forecast_on_steps is not None else None,
            "refresh_first_n_steps": self.refresh_first_n_steps,
            "refresh_last_n_steps": self.refresh_last_n_steps,
            "disable_final_step": self.disable_final_step,
            "clone_forecast": self.clone_forecast,
            "debug_jsonl_path": str(self.debug_jsonl_path) if self.debug_jsonl_path is not None else None,
            "total_steps": self.total_steps,
            "official_reference": "TaylorSeer adapted baseline, not official reproduction",
        }

    def summary(self) -> dict[str, Any]:
        mean_order = sum(self._order_values) / len(self._order_values) if self._order_values else 0.0
        stats = dict(self._stats)
        return {
            "policy": self.policy_name,
            "config": self.to_dict(),
            "stats": {
                **stats,
                "mean_effective_order": mean_order,
                "by_module": self._nested_to_dict(self._by_module),
                "by_branch": self._nested_to_dict(self._by_branch),
                "by_step": {str(key): dict(value) for key, value in sorted(self._by_step.items())},
                "by_order": {str(key): value for key, value in sorted(self._by_order.items())},
                "interval": self.interval,
                "max_order": self.max_order,
            },
        }

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "total_managed_calls": 0,
            "scheduled_refresh": 0,
            "insufficient_history_refresh": 0,
            "interval_refresh": 0,
            "forecast_decisions": 0,
            "forecast_committed": 0,
            "forecast_failures": 0,
            "history_appends": 0,
        }

    def _select_branch(self, cfg_branch: str) -> tuple[str, bool]:
        if cfg_branch in self.branches:
            return cfg_branch, True
        if self.fallback_to_global_branch and "global" in self.branches:
            return "global", True
        return cfg_branch, False

    @staticmethod
    def _make_key(module_name: str, branch: str, solver_stage: str, batch_signature: str | None) -> HistoryKey:
        return (canonical_module_name(module_name), str(branch), str(solver_stage), batch_signature)

    @staticmethod
    def _batch_signature(tensor: torch.Tensor | None) -> str | None:
        if not torch.is_tensor(tensor) or tensor.ndim == 0:
            return None
        return f"b:{int(tensor.shape[0])}"

    def _history_steps(self, key: HistoryKey) -> list[int]:
        return [step for step, _tensor in self._history.get(key, [])]

    def _decision_for_key(self, step_idx: int, key: HistoryKey) -> tuple[bool, str, str, int]:
        history = self._history.get(key, [])
        effective_order = min(self.max_order, max(0, len(history) - 1))
        if step_idx < self.refresh_first_n_steps:
            return False, "scheduled_refresh", "scheduled_refresh", effective_order
        if self.total_steps is not None:
            if self.disable_final_step and step_idx >= self.total_steps - 1:
                return False, "scheduled_refresh", "scheduled_refresh", effective_order
            if self.refresh_last_n_steps and step_idx >= self.total_steps - self.refresh_last_n_steps:
                return False, "scheduled_refresh", "scheduled_refresh", effective_order
        if step_idx % self.interval == 0:
            return False, "interval_refresh", "interval_refresh", effective_order
        if self.forecast_on_steps is not None and step_idx not in self.forecast_on_steps:
            return False, "scheduled_refresh", "scheduled_refresh", effective_order
        if len(history) < self.min_history:
            return False, "insufficient_history", "insufficient_history_refresh", effective_order
        return True, "forecast", "forecast_decisions", effective_order

    @staticmethod
    def _lagrange_extrapolate(points: list[HistoryItem], target_step: int) -> torch.Tensor | None:
        if not points:
            return None
        output: torch.Tensor | None = None
        for j, (step_j, tensor_j) in enumerate(points):
            weight = 1.0
            for k, (step_k, _tensor_k) in enumerate(points):
                if j == k:
                    continue
                denom = float(step_j - step_k)
                if denom == 0.0:
                    return None
                weight *= float(target_step - step_k) / denom
            term = tensor_j * weight
            output = term if output is None else output + term
        return output

    def _record_decision(
        self,
        *,
        step_idx: int,
        module_name: str,
        branch: str,
        solver_stage: str,
        decision: str,
        reason: str,
        stat_reason: str,
        history_steps: list[int],
        effective_order: int,
    ) -> None:
        if stat_reason in self._stats:
            self._stats[stat_reason] += 1
        self._by_module[module_name][stat_reason] += 1
        self._by_branch[branch][stat_reason] += 1
        self._by_step[int(step_idx)][stat_reason] += 1
        self._write_debug(
            {
                "policy": self.policy_name,
                "step_idx": int(step_idx),
                "module_name": module_name,
                "branch": branch,
                "solver_stage": solver_stage,
                "decision": decision,
                "reason": reason,
                "history_steps": history_steps,
                "effective_order": int(effective_order),
                "interval": self.interval,
                "max_order": self.max_order,
            }
        )

    @staticmethod
    def _nested_to_dict(value: defaultdict[str, defaultdict[str, int]]) -> dict[str, dict[str, int]]:
        return {key: dict(inner) for key, inner in sorted(value.items())}

    def _write_debug(self, payload: dict[str, Any]) -> None:
        if self.debug_jsonl_path is None:
            return
        self.debug_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
