from __future__ import annotations

import re
import math
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from pfc.cache.cached_module import ReuseProcessingResult
from pfc.cache.safe_map_policy import canonical_module_name
from pfc.cache.speca_verifier import SPECA_ERROR_METRICS, calculate_speca_error
from pfc.cache.taylorseer_policy import TaylorSeerCachePolicy


@dataclass(frozen=True)
class SpeCaStepDecision:
    step_idx: int
    mode: Literal["full", "speculative"]
    reason: str
    consecutive_speculative_steps: int
    verification_enabled: bool
    threshold: float
    previous_verification_error: float | None


@dataclass
class SpeCaStreamState:
    last_step_idx: int | None = None
    current_decision: SpeCaStepDecision | None = None
    consecutive_speculative_steps: int = 0
    last_aggregated_error: float | None = None
    pending_branch_errors: dict[str, list[float]] = field(default_factory=dict)
    full_steps: int = 0
    speculative_steps: int = 0
    current_spec_run_length: int = 0
    completed_spec_run_lengths: list[int] = field(default_factory=list)


def resolve_verifier_module(selected_modules: Iterable[str], requested: str = "auto") -> str:
    selected = [canonical_module_name(str(item)) for item in selected_modules]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("SpeCa requires at least one selected module")

    if requested != "auto":
        resolved = canonical_module_name(str(requested))
        if resolved not in selected:
            raise ValueError(
                f"Requested SpeCa verifier module {requested!r} is not in selected modules: {selected}"
            )
        return resolved

    indexed: list[tuple[int, str]] = []
    for module_name in selected:
        match = re.search(r"(?:^|\.)blocks\.(\d+)$", module_name)
        if match:
            indexed.append((int(match.group(1)), module_name))
    if indexed:
        return max(indexed, key=lambda item: item[0])[1]

    fallback = selected[-1]
    warnings.warn(
        "Could not parse numeric Transformer block indices for automatic SpeCa verifier "
        f"selection; falling back to the last selected module: {fallback}",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


class SpeCaCachePolicy(TaylorSeerCachePolicy):
    """Adapted SpeCa-style schedule around the existing TaylorSeer predictor.

    Taylor history remains branch/module/stage isolated in the parent policy,
    while each solver stage gets one shared batch-level full/speculative decision.
    """

    policy_name = "SpeCaCachePolicy"
    baseline_name = "adapted SpeCa-style"

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_name: str = "JiT",
        cache_modules: Iterable[str],
        max_order: int = 4,
        first_full_steps: int = 3,
        base_threshold: float = 0.1,
        decay_rate: float = 0.01,
        min_threshold: float = 0.01,
        min_forecast_steps: int = 2,
        max_forecast_steps: int = 5,
        error_metric: str = "relative_l1",
        branch_aggregation: Literal["mean", "max"] = "mean",
        min_history: int = 2,
        verifier_module: str = "auto",
        solver_stages: Iterable[str] | None = None,
        branches: Iterable[str] | None = None,
        verification_branches: Iterable[str] | None = None,
        fallback_to_global_branch: bool = True,
        clone_forecast: bool = False,
        debug_jsonl_path: Path | str | None = None,
        total_steps: int = 50,
        eps: float = 1e-10,
        max_verification_error_samples: int = 4096,
    ) -> None:
        selected_modules = [canonical_module_name(str(item)) for item in cache_modules]
        selected_modules = list(dict.fromkeys(selected_modules))
        if not selected_modules:
            raise ValueError("cache_modules must contain at least one selected module")
        if first_full_steps < 0:
            raise ValueError("first_full_steps must be non-negative")
        if base_threshold <= 0.0:
            raise ValueError("base_threshold must be positive")
        if min_threshold <= 0.0:
            raise ValueError("min_threshold must be positive")
        if min_threshold > base_threshold:
            raise ValueError("min_threshold must not exceed base_threshold")
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError("decay_rate must satisfy 0 < decay_rate <= 1")
        if min_forecast_steps <= 0:
            raise ValueError("min_forecast_steps must be positive")
        if max_forecast_steps < min_forecast_steps:
            raise ValueError("max_forecast_steps must be >= min_forecast_steps")
        if error_metric not in SPECA_ERROR_METRICS:
            raise ValueError(f"Unsupported SpeCa error metric: {error_metric}")
        if branch_aggregation not in {"mean", "max"}:
            raise ValueError("branch_aggregation must be 'mean' or 'max'")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if max_verification_error_samples <= 0:
            raise ValueError("max_verification_error_samples must be positive")

        resolved_branches = tuple(str(item) for item in (branches or ("cond", "uncond", "global")))
        expected_branches = tuple(str(item) for item in (verification_branches or ("cond", "uncond")))
        if not expected_branches:
            raise ValueError("verification_branches must not be empty")

        super().__init__(
            enabled=enabled,
            model_name=model_name,
            cache_modules=selected_modules,
            interval=total_steps + 1,
            max_order=max_order,
            min_history=min_history,
            solver_stages=solver_stages or {"euler"},
            branches=resolved_branches,
            fallback_to_global_branch=fallback_to_global_branch,
            refresh_first_n_steps=0,
            refresh_last_n_steps=0,
            clone_forecast=clone_forecast,
            debug_jsonl_path=debug_jsonl_path,
            total_steps=total_steps,
        )
        self.first_full_steps = int(first_full_steps)
        self.base_threshold = float(base_threshold)
        self.decay_rate = float(decay_rate)
        self.min_threshold = float(min_threshold)
        self.min_forecast_steps = int(min_forecast_steps)
        self.max_forecast_steps = int(max_forecast_steps)
        self.error_metric = str(error_metric)
        self.branch_aggregation = str(branch_aggregation)
        self.verifier_module_requested = str(verifier_module)
        self.verifier_module_resolved = resolve_verifier_module(selected_modules, verifier_module)
        self.verifier_module = self.verifier_module_resolved
        self.verification_branches = expected_branches
        self.eps = float(eps)
        self.max_verification_error_samples = int(max_verification_error_samples)

        self._stream_states: dict[str, SpeCaStreamState] = {}
        self._batch_session = 0
        self._verification_call_keys: set[tuple[int, str, int, str]] = set()
        self._verification_error_count = 0
        self._verification_error_sum = 0.0
        self._verification_error_sum_sq = 0.0
        self._verification_error_min: float | None = None
        self._verification_error_max: float | None = None
        self._verification_error_samples: deque[float] = deque(
            maxlen=self.max_verification_error_samples
        )
        self._full_compute_calls = 0
        self._all_completed_run_lengths: list[int] = []
        self._speca_stats = self._empty_speca_stats()
        self._step_decisions: dict[str, dict[str, Any]] = {}
        self._speca_by_module: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._speca_by_branch: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._verification_host_dispatch_time_sec = 0.0
        self._forecast_host_dispatch_time_sec = 0.0
        self._full_compute_host_dispatch_time_sec = 0.0

    @staticmethod
    def _empty_speca_stats() -> dict[str, int]:
        return {
            "total_steps_seen": 0,
            "full_step_decisions": 0,
            "speculative_step_decisions": 0,
            "initial_full_steps": 0,
            "insufficient_history_full_steps": 0,
            "verification_reject_full_steps": 0,
            "max_length_full_steps": 0,
            "missing_verification_error_full_steps": 0,
            "verification_steps": 0,
            "verifier_fresh_calls": 0,
            "verification_accept_decisions": 0,
            "verification_reject_decisions": 0,
            "missing_branch_verification": 0,
        }

    def threshold_for_step(self, step_idx: int) -> float:
        if step_idx < 0:
            raise ValueError("step_idx must be non-negative")
        total_steps = max(float(self.total_steps or 1), 1.0)
        progress = min(max((float(step_idx) + 1.0) / total_steps, 0.0), 1.0)
        return max(self.base_threshold * (self.decay_rate**progress), self.min_threshold)

    def histories_ready(self, *, solver_stage: str, batch_signature: str | None) -> bool:
        for module_name in sorted(self.cache_modules or ()):
            for branch in self.verification_branches:
                key = self._make_key(module_name, branch, str(solver_stage), batch_signature)
                if len(self._history.get(key, [])) < self.min_history:
                    return False
        return True

    def current_step_decision(self, solver_stage: str = "euler") -> SpeCaStepDecision | None:
        state = self._stream_states.get(str(solver_stage))
        return state.current_decision if state is not None else None

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
        active = (
            self.enabled
            and self.should_cache_module(canonical)
            and str(solver_stage) in self.solver_stages
            and branch_ok
        )
        if not active:
            return False

        signature = self._batch_signature(getattr(entry, "tensor", None))
        decision = self._decision_for_step(
            step_idx=int(step_idx),
            t=float(t),
            solver_stage=str(solver_stage),
            batch_signature=signature,
        )
        self._stats["total_managed_calls"] += 1
        self._speca_by_module[canonical][decision.mode] += 1
        self._speca_by_branch[branch][decision.mode] += 1
        if decision.mode != "speculative":
            return False
        if entry is None:
            self._stats["forecast_failures"] += 1
            raise RuntimeError(
                "SpeCa selected a speculative step but a selected module has no cache entry; "
                "this would violate all-module step consistency"
            )

        key = self._make_key(canonical, branch, str(solver_stage), signature)
        effective_order = min(self.max_order, max(0, len(self._history.get(key, [])) - 1))
        self._pending_forecasts[(key, int(step_idx))] = {
            "effective_order": effective_order,
            "history_steps": self._history_steps(key),
        }
        self._stats["forecast_decisions"] += 1
        return True

    def make_reuse_tensor(self, **kwargs: Any) -> torch.Tensor | None:
        started = time.perf_counter()
        try:
            return super().make_reuse_tensor(**kwargs)
        finally:
            self._forecast_host_dispatch_time_sec += time.perf_counter() - started

    def on_refresh_committed(self, **kwargs: Any) -> None:
        if torch.is_tensor(kwargs.get("tensor")):
            self._full_compute_calls += 1
        super().on_refresh_committed(**kwargs)

    def process_reuse_tensor(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any,
        current_input: torch.Tensor | None,
        reuse_tensor: torch.Tensor,
        fresh_compute: Callable[[], Any],
    ) -> ReuseProcessingResult:
        del entry, current_input
        canonical = canonical_module_name(str(module_name))
        state = self._stream_states.get(str(solver_stage))
        decision = state.current_decision if state is not None else None
        if (
            canonical != self.verifier_module
            or decision is None
            or decision.step_idx != int(step_idx)
            or not decision.verification_enabled
        ):
            return ReuseProcessingResult(output_tensor=reuse_tensor)

        call_key = (self._batch_session, str(solver_stage), int(step_idx), str(cfg_branch))
        if call_key in self._verification_call_keys:
            return ReuseProcessingResult(output_tensor=reuse_tensor)
        self._verification_call_keys.add(call_key)

        started = time.perf_counter()
        fresh_tensor = fresh_compute()
        self._verification_host_dispatch_time_sec += time.perf_counter() - started
        if not torch.is_tensor(fresh_tensor):
            raise TypeError("SpeCa verifier module must return a tensor")
        error = calculate_speca_error(reuse_tensor, fresh_tensor, self.error_metric, self.eps)
        self.record_verification_error(
            step_idx=int(step_idx),
            cfg_branch=str(cfg_branch),
            solver_stage=str(solver_stage),
            module_name=canonical,
            error=error,
            t=float(t),
        )
        return ReuseProcessingResult(
            output_tensor=reuse_tensor,
            verification_performed=True,
            verification_error=error,
            used_fresh_output=False,
        )

    def record_verification_error(
        self,
        *,
        step_idx: int,
        cfg_branch: str,
        solver_stage: str,
        module_name: str | None = None,
        error: float,
        t: float = 0.0,
    ) -> None:
        state = self._stream_states.get(str(solver_stage))
        decision = state.current_decision if state is not None else None
        if decision is None or decision.step_idx != int(step_idx) or not decision.verification_enabled:
            raise ValueError("Verification error does not match an enabled current SpeCa step")
        value = float(error)
        if not math.isfinite(value):
            raise ValueError("Verification error must be finite")
        state.pending_branch_errors.setdefault(str(cfg_branch), []).append(value)
        self._verification_error_count += 1
        self._verification_error_sum += value
        self._verification_error_sum_sq += value * value
        self._verification_error_min = (
            value if self._verification_error_min is None else min(self._verification_error_min, value)
        )
        self._verification_error_max = (
            value if self._verification_error_max is None else max(self._verification_error_max, value)
        )
        self._verification_error_samples.append(value)
        self._speca_stats["verifier_fresh_calls"] += 1
        self._speca_by_branch[str(cfg_branch)]["verification"] += 1
        resolved_module = canonical_module_name(module_name or self.verifier_module)
        self._speca_by_module[resolved_module]["verification"] += 1
        self._write_debug(
            {
                "event": "verification",
                "batch_session": self._batch_session,
                "step_idx": int(step_idx),
                "t": float(t),
                "branch": str(cfg_branch),
                "solver_stage": str(solver_stage),
                "module_name": resolved_module,
                "metric": self.error_metric,
                "error": value,
                "threshold": decision.threshold,
                "current_prediction_used": True,
                "affects_step": int(step_idx) + 1,
            }
        )

    def mark_full_compute_host_dispatch_time(self, elapsed_sec: float) -> None:
        self._full_compute_host_dispatch_time_sec += max(float(elapsed_sec), 0.0)

    def clear_batch(self) -> None:
        for state in self._stream_states.values():
            self._finish_speculative_run(state)
        super().clear_batch()
        self._stream_states.clear()
        self._verification_call_keys.clear()
        self._batch_session += 1

    def reset_runtime_state(self) -> None:
        self.clear_batch()
        self._batch_session = 0

    def reset_stats(self) -> None:
        super().reset_stats()
        self._verification_error_count = 0
        self._verification_error_sum = 0.0
        self._verification_error_sum_sq = 0.0
        self._verification_error_min = None
        self._verification_error_max = None
        self._verification_error_samples.clear()
        self._full_compute_calls = 0
        self._all_completed_run_lengths.clear()
        self._speca_stats = self._empty_speca_stats()
        self._step_decisions.clear()
        self._speca_by_module.clear()
        self._speca_by_branch.clear()
        self._verification_host_dispatch_time_sec = 0.0
        self._forecast_host_dispatch_time_sec = 0.0
        self._full_compute_host_dispatch_time_sec = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "baseline_name": self.baseline_name,
            "official_reproduction": False,
            "model_name": self.model_name,
            "enabled": self.enabled,
            "cache_modules": sorted(self.cache_modules or ()),
            "verifier_module_requested": self.verifier_module_requested,
            "verifier_module_resolved": self.verifier_module_resolved,
            "verifier_module": self.verifier_module_resolved,
            "max_order": self.max_order,
            "min_history": self.min_history,
            "first_full_steps": self.first_full_steps,
            "base_threshold": self.base_threshold,
            "decay_rate": self.decay_rate,
            "min_threshold": self.min_threshold,
            "min_forecast_steps": self.min_forecast_steps,
            "max_forecast_steps": self.max_forecast_steps,
            "error_metric": self.error_metric,
            "branch_aggregation": self.branch_aggregation,
            "verification_branches": list(self.verification_branches),
            "solver_stages": sorted(self.solver_stages),
            "total_steps": self.total_steps,
            "eps": self.eps,
            "max_verification_error_samples": self.max_verification_error_samples,
            "clone_forecast": self.clone_forecast,
            "debug_jsonl_path": str(self.debug_jsonl_path) if self.debug_jsonl_path else None,
            "decision_timing": "next_step",
            "sample_adaptivity": "batch-level",
            "draft_type": "adapted TaylorSeer block-output forecasting",
            "timing_semantics": "host_dispatch_only",
            "cuda_event_profiling_enabled": False,
        }

    def verification_overhead_stats(self) -> dict[str, Any]:
        selected_count = len(self.cache_modules or ())
        return {
            "verifier_fresh_calls": self._speca_stats["verifier_fresh_calls"],
            "number_of_selected_blocks": selected_count,
            "estimated_verifier_block_fraction": (1.0 / selected_count) if selected_count else None,
            "timing_semantics": "host_dispatch_only",
            "cuda_event_profiling_enabled": False,
            "verification_host_dispatch_time_sec": self._verification_host_dispatch_time_sec,
            "forecast_host_dispatch_time_sec": self._forecast_host_dispatch_time_sec,
            "full_compute_host_dispatch_time_sec": self._full_compute_host_dispatch_time_sec,
            "verification_cuda_time_sec": None,
            "forecast_cuda_time_sec": None,
            "full_compute_cuda_time_sec": None,
        }

    def summary(self) -> dict[str, Any]:
        errors = self._error_summary()
        completed = list(self._all_completed_run_lengths)
        completed.extend(
            state.current_spec_run_length
            for state in self._stream_states.values()
            if state.current_spec_run_length > 0
        )
        accepted = self._speca_stats["verification_accept_decisions"]
        rejected = self._speca_stats["verification_reject_decisions"]
        acceptance_denominator = accepted + rejected
        full_steps = self._speca_stats["full_step_decisions"]
        speculative_steps = self._speca_stats["speculative_step_decisions"]
        total_step_decisions = full_steps + speculative_steps
        forecast_committed = self._stats["forecast_committed"]
        full_compute_calls = self._full_compute_calls
        verifier_fresh_calls = self._speca_stats["verifier_fresh_calls"]
        logical_managed_calls = full_compute_calls + forecast_committed
        actual_original_module_forwards = full_compute_calls + verifier_fresh_calls
        effective_skipped_block_calls = max(forecast_committed - verifier_fresh_calls, 0)
        mean_order = sum(self._order_values) / len(self._order_values) if self._order_values else 0.0
        result: dict[str, Any] = {
            "policy_name": self.policy_name,
            "baseline_name": self.baseline_name,
            **self._speca_stats,
            "verification_errors": errors,
            "total_step_decisions": total_step_decisions,
            "verification_acceptance_rate": (
                accepted / acceptance_denominator if acceptance_denominator else 0.0
            ),
            "speculative_step_ratio": (
                speculative_steps / total_step_decisions if total_step_decisions else 0.0
            ),
            "completed_speculative_runs": len(completed),
            "mean_speculative_run_length": sum(completed) / len(completed) if completed else 0.0,
            "max_speculative_run_length": max(completed) if completed else 0,
            "raw_cache_hits": forecast_committed,
            "forecast_committed": forecast_committed,
            "forecast_failures": self._stats["forecast_failures"],
            "mean_effective_order": mean_order,
            "logical_managed_calls": logical_managed_calls,
            "full_compute_calls": full_compute_calls,
            "actual_original_module_forwards": actual_original_module_forwards,
            "effective_skipped_block_calls": effective_skipped_block_calls,
            "raw_forecast_rate": (
                forecast_committed / logical_managed_calls if logical_managed_calls else 0.0
            ),
            "effective_compute_saving_rate": (
                effective_skipped_block_calls / logical_managed_calls
                if logical_managed_calls
                else 0.0
            ),
            "verifier_overhead_rate": (
                verifier_fresh_calls / logical_managed_calls if logical_managed_calls else 0.0
            ),
            "actual_compute_fraction": (
                actual_original_module_forwards / logical_managed_calls
                if logical_managed_calls
                else 0.0
            ),
            "verifier_module_requested": self.verifier_module_requested,
            "verifier_module_resolved": self.verifier_module_resolved,
            "verifier_module": self.verifier_module_resolved,
            "max_order": self.max_order,
            "base_threshold": self.base_threshold,
            "decay_rate": self.decay_rate,
            "min_threshold": self.min_threshold,
            "min_forecast_steps": self.min_forecast_steps,
            "max_forecast_steps": self.max_forecast_steps,
            "first_full_steps": self.first_full_steps,
            "error_metric": self.error_metric,
            "branch_aggregation": self.branch_aggregation,
            "by_step": dict(sorted(self._step_decisions.items())),
            "by_branch": self._nested_counts(self._speca_by_branch),
            "by_module": self._nested_counts(self._speca_by_module),
            "verification_overhead_stats": self.verification_overhead_stats(),
            "timing_semantics": "host_dispatch_only",
        }
        result["config"] = self.to_dict()
        result["stats"] = {
            key: value
            for key, value in result.items()
            if key not in {"config", "stats", "by_step", "by_branch", "by_module"}
        }
        return result

    def _decision_for_step(
        self,
        *,
        step_idx: int,
        t: float,
        solver_stage: str,
        batch_signature: str | None,
    ) -> SpeCaStepDecision:
        state = self._stream_states.setdefault(solver_stage, SpeCaStreamState())
        if state.last_step_idx == step_idx and state.current_decision is not None:
            return state.current_decision
        if state.last_step_idx is not None and step_idx <= state.last_step_idx:
            raise ValueError("SpeCa step indices must increase within a batch; call clear_batch between batches")

        missing_error = self._finalize_pending_verification(state)
        previous = state.current_decision
        previous_mode = previous.mode if previous is not None else None
        previous_speculative_steps = (
            state.consecutive_speculative_steps if previous_mode == "speculative" else 0
        )
        previous_error = state.last_aggregated_error
        threshold = self.threshold_for_step(step_idx)
        ready = self.histories_ready(solver_stage=solver_stage, batch_signature=batch_signature)

        if step_idx < self.first_full_steps:
            mode, reason, consecutive = "full", "initial_full_steps", 0
        elif not ready:
            mode, reason, consecutive = "full", "insufficient_history", 0
        elif previous is not None and previous.mode == "full":
            mode, reason, consecutive = "speculative", "min_forecast_run", 1
        elif missing_error:
            mode, reason, consecutive = "full", "missing_verification_error", 0
        elif previous_speculative_steps >= self.max_forecast_steps:
            mode, reason, consecutive = "full", "max_forecast_steps", 0
        elif (
            previous_speculative_steps >= self.min_forecast_steps
            and previous_error is not None
            and previous_error > threshold
        ):
            mode, reason, consecutive = "full", "verification_reject", 0
        else:
            consecutive = previous_speculative_steps + 1 if previous_mode == "speculative" else 1
            mode = "speculative"
            reason = "verification_accept" if previous_error is not None else "min_forecast_run"

        verification_enabled = (
            mode == "speculative"
            and previous_speculative_steps >= self.min_forecast_steps
        )
        decision = SpeCaStepDecision(
            step_idx=step_idx,
            mode=mode,
            reason=reason,
            consecutive_speculative_steps=consecutive,
            verification_enabled=verification_enabled,
            threshold=threshold,
            previous_verification_error=previous_error,
        )

        if mode == "full":
            if previous is not None and previous.mode == "speculative":
                self._finish_speculative_run(state)
            state.consecutive_speculative_steps = 0
            state.current_spec_run_length = 0
            state.full_steps += 1
            self._speca_stats["full_step_decisions"] += 1
            reason_key = {
                "initial_full_steps": "initial_full_steps",
                "insufficient_history": "insufficient_history_full_steps",
                "verification_reject": "verification_reject_full_steps",
                "max_forecast_steps": "max_length_full_steps",
                "missing_verification_error": "missing_verification_error_full_steps",
            }.get(reason)
            if reason_key is not None:
                self._speca_stats[reason_key] += 1
            if reason == "verification_reject":
                self._speca_stats["verification_reject_decisions"] += 1
            state.last_aggregated_error = None
        else:
            state.consecutive_speculative_steps = consecutive
            state.current_spec_run_length += 1
            state.speculative_steps += 1
            self._speca_stats["speculative_step_decisions"] += 1
            if reason == "verification_accept":
                self._speca_stats["verification_accept_decisions"] += 1
            if verification_enabled:
                self._speca_stats["verification_steps"] += 1

        state.last_step_idx = step_idx
        state.current_decision = decision
        state.pending_branch_errors.clear()
        self._speca_stats["total_steps_seen"] += 1
        step_key = f"batch{self._batch_session}:{solver_stage}:{step_idx}"
        payload = {
            "event": "step_decision",
            "batch_session": self._batch_session,
            "step_idx": step_idx,
            "t": t,
            "solver_stage": solver_stage,
            "mode": mode,
            "reason": reason,
            "previous_speculative_steps": previous_speculative_steps,
            "consecutive_speculative_steps": consecutive,
            "verification_enabled": verification_enabled,
            "threshold": threshold,
            "previous_aggregated_error": previous_error,
            "verifier_module": self.verifier_module,
        }
        self._step_decisions[step_key] = dict(payload)
        self._write_debug(payload)
        return decision

    def _finalize_pending_verification(self, state: SpeCaStreamState) -> bool:
        previous = state.current_decision
        if previous is None or not previous.verification_enabled:
            state.last_aggregated_error = None
            state.pending_branch_errors.clear()
            return False

        branch_values = {
            branch: sum(values) / len(values)
            for branch, values in state.pending_branch_errors.items()
            if values
        }
        if not branch_values:
            state.last_aggregated_error = None
            return True
        missing = [branch for branch in self.verification_branches if branch not in branch_values]
        if missing:
            self._speca_stats["missing_branch_verification"] += 1
        values = list(branch_values.values())
        state.last_aggregated_error = (
            sum(values) / len(values) if self.branch_aggregation == "mean" else max(values)
        )
        state.pending_branch_errors.clear()
        return False

    def _finish_speculative_run(self, state: SpeCaStreamState) -> None:
        if state.current_spec_run_length <= 0:
            return
        length = state.current_spec_run_length
        state.completed_spec_run_lengths.append(length)
        self._all_completed_run_lengths.append(length)
        state.current_spec_run_length = 0

    def _error_summary(self) -> dict[str, float | int | bool | None]:
        if not self._verification_error_count:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "sample_count": 0,
                "quantiles_approximate": False,
                "max_samples": self.max_verification_error_samples,
            }
        values = sorted(self._verification_error_samples)
        count = self._verification_error_count
        mean = self._verification_error_sum / count
        variance = max(self._verification_error_sum_sq / count - mean * mean, 0.0)
        return {
            "count": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self._verification_error_min,
            "max": self._verification_error_max,
            "p50": self._percentile(values, 0.50),
            "p90": self._percentile(values, 0.90),
            "p95": self._percentile(values, 0.95),
            "sample_count": len(values),
            "quantiles_approximate": count > len(values),
            "max_samples": self.max_verification_error_samples,
        }

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    @staticmethod
    def _nested_counts(
        payload: defaultdict[str, defaultdict[str, int]],
    ) -> dict[str, dict[str, int]]:
        return {key: dict(sorted(value.items())) for key, value in sorted(payload.items())}
