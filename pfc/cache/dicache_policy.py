from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from pfc.cache.dicache_state import (
    DiCacheRuntimeState,
    RunningStats,
    compact_history_tensor,
)


ErrorChoice = Literal["delta_y", "delta_minus"]
BranchAggregation = Literal["mean", "max"]
ScheduleVariant = Literal["released_flux_compat"]

_FINITE_CONSERVATIVE_ERROR = 1.0e6
_BRANCHES = ("cond", "uncond")


def relative_l1_tensor(
    current: torch.Tensor,
    previous: torch.Tensor,
    eps: float,
    max_error: float = _FINITE_CONSERVATIVE_ERROR,
) -> torch.Tensor:
    """Return a finite relative-L1 scalar without leaving the input device."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if max_error <= 0.0:
        raise ValueError("max_error must be positive")
    if current.shape != previous.shape:
        raise ValueError(
            f"relative_l1 shape mismatch: {tuple(current.shape)} != {tuple(previous.shape)}"
        )
    current_f = current.float()
    previous_f = previous.float()
    numerator = (current_f - previous_f).abs().mean()
    denominator = previous_f.abs().mean()
    value = numerator / (denominator + eps)
    return torch.nan_to_num(
        value,
        nan=max_error,
        posinf=max_error,
        neginf=max_error,
    )


def relative_l1(
    current: torch.Tensor,
    previous: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """Offline/test wrapper around :func:`relative_l1_tensor`."""

    return float(relative_l1_tensor(current, previous, eps).detach().cpu().item())


def delta_minus_tensor(
    delta_y: torch.Tensor,
    delta_x: torch.Tensor,
    max_error: float = _FINITE_CONSERVATIVE_ERROR,
) -> torch.Tensor:
    value = (delta_y - delta_x).abs()
    return torch.nan_to_num(
        value,
        nan=max_error,
        posinf=max_error,
        neginf=max_error,
    )


def delta_minus(delta_y: float, delta_x: float) -> float:
    result = abs(float(delta_y) - float(delta_x))
    return result if math.isfinite(result) else _FINITE_CONSERVATIVE_ERROR


def aggregate_branch_errors(
    values: Mapping[str, float],
    aggregation: BranchAggregation,
) -> float:
    if not values:
        raise ValueError("at least one branch error is required")
    finite_values = [float(value) for value in values.values()]
    if aggregation == "mean":
        result = sum(finite_values) / len(finite_values)
    elif aggregation == "max":
        result = max(finite_values)
    else:
        raise ValueError(f"unsupported branch aggregation: {aggregation}")
    return result if math.isfinite(result) else _FINITE_CONSERVATIVE_ERROR


def is_retention_full_step(step_idx: int, total_steps: int, ret_ratio: float) -> bool:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step_idx < 0 or step_idx >= total_steps:
        raise ValueError("step_idx must satisfy 0 <= step_idx < total_steps")
    if not 0.0 <= ret_ratio < 1.0:
        raise ValueError("ret_ratio must satisfy 0 <= value < 1")
    return step_idx <= int(ret_ratio * total_steps)


def released_flux_schedule_reason(
    *,
    enabled: bool,
    force_full: bool,
    step_idx: int,
    total_steps: int,
    ret_ratio: float,
    force_last_step_full: bool,
) -> str | None:
    """Resolve non-adaptive released-FLUX full-step boundaries in one place."""

    if not enabled or force_full:
        return "force_full"
    if step_idx == 0:
        return "first_step"
    if is_retention_full_step(step_idx, total_steps, ret_ratio):
        return "retention_full"
    if force_last_step_full and step_idx == total_steps - 1:
        return "last_step"
    return None


def released_flux_should_reuse(
    accumulated_error: float,
    reuse_threshold: float,
) -> bool:
    """Released FLUX decision rule: reuse only for a strict threshold inequality."""

    return accumulated_error < reuse_threshold


def dcta_residual_tensor(
    *,
    r_old: torch.Tensor,
    r_new: torch.Tensor,
    p_old: torch.Tensor,
    p_new: torch.Tensor,
    p_current: torch.Tensor,
    eps: float,
    gamma_min: float,
    gamma_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DCTA on-device and return residual, clipped/raw gamma, and validity."""

    numerator_tensor = (p_current - p_old).float().abs().mean()
    denominator_tensor = (p_new - p_old).float().abs().mean()
    gamma_raw_tensor = numerator_tensor / (denominator_tensor + eps)
    valid_tensor = (
        torch.isfinite(denominator_tensor)
        & (denominator_tensor > eps)
    )
    gamma_tensor = torch.nan_to_num(
        gamma_raw_tensor,
        nan=1.0,
        posinf=gamma_max,
        neginf=gamma_min,
    ).clamp(gamma_min, gamma_max)
    residual_hat = r_old + gamma_tensor.to(
        device=r_old.device,
        dtype=r_old.dtype,
    ) * (r_new - r_old)
    residual = torch.where(valid_tensor, residual_hat, r_new)
    return residual, gamma_tensor, gamma_raw_tensor, valid_tensor


@dataclass(frozen=True)
class DiCacheDecision:
    step_idx: int
    decision: Literal["full", "reuse"]
    reason: str
    delta_y: dict[str, float | None]
    branch_errors: dict[str, float | None]
    delta_x: float | None
    aggregated_error: float | None
    accumulated_error_before: float
    accumulated_error_after: float
    dcta_degenerate: dict[str, bool | None]


@dataclass(frozen=True)
class DCTAResult:
    residual: torch.Tensor
    gamma_tensor: torch.Tensor | None
    dcta_used: bool
    fallback_used: bool
    degenerate: bool
    reason: str


class DiCachePolicy:
    """Batch-level shared Online Probe scheduler with branch-specific DCTA."""

    policy_name = "DiCachePolicy"
    baseline_name = "adapted DiCache-style"

    def __init__(
        self,
        *,
        total_blocks: int,
        enabled: bool = True,
        total_steps: int = 50,
        probe_depth: int = 1,
        reuse_threshold: float = 0.4,
        error_choice: ErrorChoice = "delta_y",
        branch_aggregation: BranchAggregation = "mean",
        ret_ratio: float = 0.2,
        force_last_step_full: bool = True,
        dcta_enabled: bool = True,
        gamma_min: float = 1.0,
        gamma_max: float = 1.5,
        eps: float = 1e-10,
        clone_history: bool = False,
        debug_jsonl: Path | None = None,
        max_error_samples: int = 4096,
        schedule_variant: ScheduleVariant = "released_flux_compat",
        share_cfg_prefix: bool = False,
        force_full: bool = False,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if total_blocks <= 1:
            raise ValueError("total_blocks must be greater than one")
        if not 1 <= probe_depth < total_blocks:
            raise ValueError("probe_depth must satisfy 1 <= probe_depth < total_blocks")
        if reuse_threshold <= 0.0:
            raise ValueError("reuse_threshold must be positive")
        if error_choice not in {"delta_y", "delta_minus"}:
            raise ValueError(f"unsupported error_choice: {error_choice}")
        if branch_aggregation not in {"mean", "max"}:
            raise ValueError(f"unsupported branch_aggregation: {branch_aggregation}")
        if not 0.0 <= ret_ratio < 1.0:
            raise ValueError("ret_ratio must satisfy 0 <= value < 1")
        if gamma_min < 0.0 or gamma_min > gamma_max:
            raise ValueError("gamma bounds must satisfy 0 <= gamma_min <= gamma_max")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if max_error_samples <= 0:
            raise ValueError("max_error_samples must be positive")
        if schedule_variant != "released_flux_compat":
            raise ValueError(f"unsupported DiCache schedule variant: {schedule_variant}")

        self.enabled = bool(enabled)
        self.total_steps = int(total_steps)
        self.total_blocks = int(total_blocks)
        self.probe_depth = int(probe_depth)
        self.reuse_threshold = float(reuse_threshold)
        self.error_choice = error_choice
        self.branch_aggregation = branch_aggregation
        self.ret_ratio = float(ret_ratio)
        self.force_last_step_full = bool(force_last_step_full)
        self.dcta_enabled = bool(dcta_enabled)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.eps = float(eps)
        self.clone_history = bool(clone_history)
        self.debug_jsonl = Path(debug_jsonl) if debug_jsonl is not None else None
        self.max_error_samples = int(max_error_samples)
        self.schedule_variant = schedule_variant
        self.share_cfg_prefix = bool(share_cfg_prefix)
        self.force_full = bool(force_full)
        self.state = DiCacheRuntimeState(enabled=self.enabled, total_steps=self.total_steps)

        self._counts = self._new_counts()
        self._observed_stats = self._new_error_stats()
        self._decision_stats = self._new_error_stats()
        self._gamma_stats = RunningStats(max_samples=self.max_error_samples)
        self.pending_gamma_tensors: list[torch.Tensor] = []
        self._pending_gamma_raw_tensors: list[torch.Tensor] = []
        self._timings = self._new_timings()
        self._current_history_storage_bytes = 0
        self._peak_history_storage_bytes = 0
        self._history_tensor_count = 0
        self._history_unique_storage_count = 0

    @staticmethod
    def _new_counts() -> dict[str, int]:
        return {
            "total_steps_seen": 0,
            "full_step_decisions": 0,
            "reuse_step_decisions": 0,
            "first_full_steps": 0,
            "retention_full_steps": 0,
            "adaptive_refresh_steps": 0,
            "last_step_full_steps": 0,
            "insufficient_history_full_steps": 0,
            "force_full_steps": 0,
            "decision_device_to_host_syncs": 0,
            "dcta_branch_calls": 0,
            "dcta_branch_fallback_calls": 0,
            "dcta_branch_degenerate_fallback_calls": 0,
            "dcta_branch_insufficient_history_fallback_calls": 0,
            "dcta_steps": 0,
            "dcta_fallback_steps": 0,
            "gamma_clip_low_count": 0,
            "gamma_clip_high_count": 0,
        }

    @staticmethod
    def _new_timings() -> dict[str, float]:
        return {
            "probe_host_dispatch_time_sec": 0.0,
            "deep_compute_host_dispatch_time_sec": 0.0,
            "dcta_host_dispatch_time_sec": 0.0,
            "final_layer_host_dispatch_time_sec": 0.0,
        }

    def _new_error_stats(self) -> dict[str, RunningStats]:
        return {
            "delta_y": RunningStats(max_samples=self.max_error_samples),
            "delta_x": RunningStats(max_samples=self.max_error_samples),
            "branch_error": RunningStats(max_samples=self.max_error_samples),
        }

    def clear_batch(self) -> None:
        self.finalize_batch_statistics()
        self.state.clear_batch()
        self._update_history_memory_stats()

    def reset_runtime_state(self) -> None:
        self.clear_batch()

    def reset_stats(self) -> None:
        self.finalize_batch_statistics()
        self._counts = self._new_counts()
        self._observed_stats = self._new_error_stats()
        self._decision_stats = self._new_error_stats()
        self._gamma_stats = RunningStats(max_samples=self.max_error_samples)
        self.pending_gamma_tensors.clear()
        self._pending_gamma_raw_tensors.clear()
        self._timings = self._new_timings()
        self._current_history_storage_bytes = 0
        self._peak_history_storage_bytes = 0
        self._history_tensor_count = 0
        self._history_unique_storage_count = 0
        self._update_history_memory_stats()

    def finalize_batch_statistics(self) -> None:
        """Flush pending device gamma scalars with one batch-level host transfer."""

        if not self.pending_gamma_tensors:
            return
        pairs = torch.stack(
            [
                torch.stack((gamma.detach().float(), raw.detach().float()))
                for gamma, raw in zip(
                    self.pending_gamma_tensors,
                    self._pending_gamma_raw_tensors,
                )
            ]
        )
        values = pairs.cpu().tolist()
        for gamma, raw in values:
            self._gamma_stats.add(float(gamma))
            if float(raw) < self.gamma_min:
                self._counts["gamma_clip_low_count"] += 1
            if float(raw) > self.gamma_max:
                self._counts["gamma_clip_high_count"] += 1
        self.pending_gamma_tensors.clear()
        self._pending_gamma_raw_tensors.clear()

    def add_host_dispatch_time(self, field: str, duration_sec: float) -> None:
        if field not in self._timings:
            raise ValueError(f"unsupported DiCache timing field: {field}")
        self._timings[field] += max(float(duration_sec), 0.0)

    def decide(
        self,
        *,
        step_idx: int,
        input_feature: torch.Tensor,
        probe_features: Mapping[str, torch.Tensor],
    ) -> DiCacheDecision:
        if step_idx < 0 or step_idx >= self.total_steps:
            raise ValueError("step_idx must satisfy 0 <= step_idx < total_steps")
        if set(probe_features) != set(self.state.branch_histories):
            raise ValueError("probe_features must contain exactly cond and uncond branches")

        self.state.current_step_idx = step_idx
        before = float(self.state.accumulated_error)
        mandatory_reason = released_flux_schedule_reason(
            enabled=self.enabled,
            force_full=self.force_full,
            step_idx=step_idx,
            total_steps=self.total_steps,
            ret_ratio=self.ret_ratio,
            force_last_step_full=self.force_last_step_full,
        )
        has_full_history = all(
            bool(history.full_residual_history)
            for history in self.state.branch_histories.values()
        )
        metric_ready = self.state.previous_input_feature is not None and all(
            self.state.branch_histories[branch].previous_probe_feature is not None
            for branch in _BRANCHES
        )

        delta_x_value: float | None = None
        delta_y_values: dict[str, float | None] = {branch: None for branch in _BRANCHES}
        branch_errors: dict[str, float | None] = {branch: None for branch in _BRANCHES}
        dcta_degenerate: dict[str, bool | None] = {branch: None for branch in _BRANCHES}
        aggregated_error: float | None = None

        if metric_ready:
            assert self.state.previous_input_feature is not None
            delta_x_tensor = relative_l1_tensor(
                input_feature,
                self.state.previous_input_feature,
                self.eps,
            )
            delta_y_tensors = {
                branch: relative_l1_tensor(
                    probe_features[branch],
                    self.state.branch_histories[branch].previous_probe_feature,
                    self.eps,
                )
                for branch in _BRANCHES
            }
            metric_tensors = [
                delta_x_tensor,
                delta_y_tensors["cond"],
                delta_y_tensors["uncond"],
            ]
            denominator_branches: list[str] = []
            if mandatory_reason is None and self.dcta_enabled:
                for branch in _BRANCHES:
                    history = self.state.branch_histories[branch]
                    if len(history.probe_residual_history) >= 2:
                        p_old, p_new = history.probe_residual_history
                        metric_tensors.append((p_new - p_old).float().abs().mean())
                        denominator_branches.append(branch)
            metric_values = torch.stack(metric_tensors).detach().float().cpu().tolist()
            self._counts["decision_device_to_host_syncs"] += 1
            delta_x_value = float(metric_values[0])
            delta_y_values = {
                "cond": float(metric_values[1]),
                "uncond": float(metric_values[2]),
            }
            for branch, denominator in zip(denominator_branches, metric_values[3:]):
                dcta_degenerate[branch] = (
                    not math.isfinite(float(denominator))
                    or float(denominator) <= self.eps
                )

            self._observed_stats["delta_x"].add(delta_x_value)
            for branch in _BRANCHES:
                delta_y_value = delta_y_values[branch]
                assert delta_y_value is not None
                self._observed_stats["delta_y"].add(delta_y_value)
                if self.error_choice == "delta_y":
                    error = delta_y_value
                else:
                    error = delta_minus(delta_y_value, delta_x_value)
                branch_errors[branch] = error
                self._observed_stats["branch_error"].add(error)
            aggregated_error = aggregate_branch_errors(
                {branch: float(branch_errors[branch]) for branch in _BRANCHES},
                self.branch_aggregation,
            )

        if mandatory_reason is not None:
            decision, reason = "full", mandatory_reason
            candidate = before
        elif not metric_ready or not has_full_history or aggregated_error is None:
            decision, reason = "full", "insufficient_history"
            candidate = before
        else:
            reason = "adaptive_threshold"
            assert delta_x_value is not None
            self._decision_stats["delta_x"].add(delta_x_value)
            for branch in _BRANCHES:
                delta_y_value = delta_y_values[branch]
                branch_error = branch_errors[branch]
                assert delta_y_value is not None and branch_error is not None
                self._decision_stats["delta_y"].add(delta_y_value)
                self._decision_stats["branch_error"].add(branch_error)
            candidate = before + aggregated_error
            decision = (
                "reuse"
                if released_flux_should_reuse(candidate, self.reuse_threshold)
                else "full"
            )

        after = candidate if decision == "reuse" else 0.0
        self.state.accumulated_error = after
        self._record_decision(decision, reason)
        return DiCacheDecision(
            step_idx=step_idx,
            decision=decision,
            reason=reason,
            delta_y=delta_y_values,
            branch_errors=branch_errors,
            delta_x=delta_x_value,
            aggregated_error=aggregated_error,
            accumulated_error_before=before,
            accumulated_error_after=after,
            dcta_degenerate=dcta_degenerate,
        )

    def _record_decision(self, decision: str, reason: str) -> None:
        self._counts["total_steps_seen"] += 1
        if decision == "reuse":
            self._counts["reuse_step_decisions"] += 1
            return
        self._counts["full_step_decisions"] += 1
        reason_fields = {
            "first_step": "first_full_steps",
            "retention_full": "retention_full_steps",
            "adaptive_threshold": "adaptive_refresh_steps",
            "last_step": "last_step_full_steps",
            "insufficient_history": "insufficient_history_full_steps",
            "force_full": "force_full_steps",
        }
        field = reason_fields.get(reason)
        if field is not None:
            self._counts[field] += 1

    def record_refresh(
        self,
        branch: str,
        *,
        input_feature: torch.Tensor,
        probe_feature: torch.Tensor,
        full_feature: torch.Tensor,
        step_idx: int,
    ) -> None:
        history = self.state.branch_histories[branch]
        probe_residual = probe_feature - input_feature
        full_residual = full_feature - input_feature
        history.full_residual_history.append(
            compact_history_tensor(full_residual, clone_history=self.clone_history)
        )
        history.probe_residual_history.append(
            compact_history_tensor(probe_residual, clone_history=self.clone_history)
        )
        history.refresh_step_history.append(int(step_idx))
        self._update_history_memory_stats()

    def approximate_residual(
        self,
        branch: str,
        *,
        current_probe_residual: torch.Tensor,
        degenerate: bool | None = None,
    ) -> DCTAResult:
        history = self.state.branch_histories[branch]
        if not history.full_residual_history:
            raise RuntimeError(f"branch {branch!r} has no full residual history")
        self._counts["dcta_branch_calls"] += 1
        latest = history.full_residual_history[-1]
        if not self.dcta_enabled:
            self._counts["dcta_branch_fallback_calls"] += 1
            return DCTAResult(
                residual=latest,
                gamma_tensor=None,
                dcta_used=False,
                fallback_used=True,
                degenerate=False,
                reason="dcta_disabled_fallback",
            )
        if len(history.full_residual_history) < 2:
            self._counts["dcta_branch_fallback_calls"] += 1
            self._counts["dcta_branch_insufficient_history_fallback_calls"] += 1
            return DCTAResult(
                residual=latest,
                gamma_tensor=None,
                dcta_used=False,
                fallback_used=True,
                degenerate=False,
                reason="insufficient_history_fallback",
            )

        r_old, r_new = history.full_residual_history
        p_old, p_new = history.probe_residual_history
        residual, gamma_tensor, gamma_raw_tensor, valid_tensor = dcta_residual_tensor(
            r_old=r_old,
            r_new=r_new,
            p_old=p_old,
            p_new=p_new,
            p_current=current_probe_residual,
            eps=self.eps,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
        )
        if degenerate is None:
            degenerate = not bool(valid_tensor)
        if degenerate:
            self._counts["dcta_branch_fallback_calls"] += 1
            self._counts["dcta_branch_degenerate_fallback_calls"] += 1
            return DCTAResult(
                residual=r_new,
                gamma_tensor=None,
                dcta_used=False,
                fallback_used=True,
                degenerate=True,
                reason="degenerate_probe_trajectory_fallback",
            )

        self.pending_gamma_tensors.append(gamma_tensor)
        self._pending_gamma_raw_tensors.append(gamma_raw_tensor)
        return DCTAResult(
            residual=residual,
            gamma_tensor=gamma_tensor,
            dcta_used=True,
            fallback_used=False,
            degenerate=False,
            reason="dcta",
        )

    def finish_step(
        self,
        decision: DiCacheDecision,
        *,
        input_feature: torch.Tensor,
        probe_features: Mapping[str, torch.Tensor],
        dcta_results: Mapping[str, DCTAResult] | None = None,
        t: float | None = None,
    ) -> None:
        self.state.previous_input_feature = compact_history_tensor(
            input_feature,
            clone_history=self.clone_history,
        )
        for branch, probe_feature in probe_features.items():
            self.state.branch_histories[branch].previous_probe_feature = compact_history_tensor(
                probe_feature,
                clone_history=self.clone_history,
            )
        results = dcta_results or {}
        if results:
            self._counts["dcta_steps"] += 1
            if any(result.fallback_used for result in results.values()):
                self._counts["dcta_fallback_steps"] += 1
        self._update_history_memory_stats()
        if self.debug_jsonl is not None:
            self._write_debug(decision, results, t=t)
        self.state.current_step_idx = None

    def _write_debug(
        self,
        decision: DiCacheDecision,
        dcta_results: Mapping[str, DCTAResult],
        *,
        t: float | None,
    ) -> None:
        assert self.debug_jsonl is not None
        self.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        gamma_values: dict[str, float | None] = {branch: None for branch in _BRANCHES}
        gamma_branches = [
            branch
            for branch in _BRANCHES
            if branch in dcta_results and dcta_results[branch].gamma_tensor is not None
        ]
        if gamma_branches:
            host_gammas = torch.stack(
                [dcta_results[branch].gamma_tensor.detach().float() for branch in gamma_branches]
            ).cpu().tolist()
            gamma_values.update(zip(gamma_branches, (float(value) for value in host_gammas)))
        payload = {
            "event": "dicache_step",
            "step_idx": decision.step_idx,
            "t": t,
            "decision": decision.decision,
            "reason": decision.reason,
            "probe_depth": self.probe_depth,
            "error_choice": self.error_choice,
            "cond_delta_y": decision.delta_y.get("cond"),
            "uncond_delta_y": decision.delta_y.get("uncond"),
            "delta_x": decision.delta_x,
            "aggregated_error": decision.aggregated_error,
            "accumulated_error_before": decision.accumulated_error_before,
            "accumulated_error_after": decision.accumulated_error_after,
            "threshold": self.reuse_threshold,
            "dcta_used": any(result.dcta_used for result in dcta_results.values()),
            "cond_gamma": gamma_values["cond"],
            "uncond_gamma": gamma_values["uncond"],
            "history_size_cond": len(
                self.state.branch_histories["cond"].full_residual_history
            ),
            "history_size_uncond": len(
                self.state.branch_histories["uncond"].full_residual_history
            ),
            "debug_sync_overhead_enabled": True,
        }
        with self.debug_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    def _history_tensors(self) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        if self.state.previous_input_feature is not None:
            values.append(self.state.previous_input_feature)
        for history in self.state.branch_histories.values():
            if history.previous_probe_feature is not None:
                values.append(history.previous_probe_feature)
            values.extend(history.full_residual_history)
            values.extend(history.probe_residual_history)
        return values

    def _update_history_memory_stats(self) -> None:
        tensors = self._history_tensors()
        storages: dict[tuple[str, int], int] = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()))
            storages.setdefault(key, int(storage.nbytes()))
        current = sum(storages.values())
        self._current_history_storage_bytes = current
        self._peak_history_storage_bytes = max(self._peak_history_storage_bytes, current)
        self._history_tensor_count = len(tensors)
        self._history_unique_storage_count = len(storages)

    def summary(self) -> dict[str, Any]:
        self._update_history_memory_stats()
        total_steps_seen = self._counts["total_steps_seen"]
        full_steps = self._counts["full_step_decisions"]
        reuse_steps = self._counts["reuse_step_decisions"]
        branches = len(self.state.branch_histories)
        deep_blocks = self.total_blocks - self.probe_depth
        reference_calls = total_steps_seen * self.total_blocks * branches
        probe_calls = total_steps_seen * self.probe_depth * branches
        deep_calls = full_steps * deep_blocks * branches
        actual_calls = probe_calls + deep_calls
        skipped_calls = max(reference_calls - actual_calls, 0)
        reference_prefix_calls = total_steps_seen * branches
        actual_prefix_calls = total_steps_seen * (1 if self.share_cfg_prefix else branches)
        observed = {
            key: stats.summary() for key, stats in self._observed_stats.items()
        }
        decision = {
            key: stats.summary() for key, stats in self._decision_stats.items()
        }
        gamma = self._gamma_stats.summary()
        retention_last = min(self.total_steps - 1, int(self.ret_ratio * self.total_steps))
        result = {
            "policy_name": self.policy_name,
            "baseline_name": self.baseline_name,
            "official_reproduction": False,
            "schedule_variant": self.schedule_variant,
            "share_cfg_prefix": self.share_cfg_prefix,
            "cfg_prefix_sharing_enabled": self.share_cfg_prefix,
            **self._counts,
            "probe_depth": self.probe_depth,
            "total_blocks": self.total_blocks,
            "deep_blocks": deep_blocks,
            "cfg_branches": branches,
            "reference_block_calls": reference_calls,
            "probe_block_calls": probe_calls,
            "deep_block_calls": deep_calls,
            "actual_block_calls": actual_calls,
            "effective_skipped_block_calls": skipped_calls,
            "effective_block_compute_saving_rate": (
                skipped_calls / reference_calls if reference_calls else 0.0
            ),
            "actual_block_compute_fraction": (
                actual_calls / reference_calls if reference_calls else 0.0
            ),
            "reuse_step_ratio": reuse_steps / total_steps_seen if total_steps_seen else 0.0,
            "reference_cfg_prefix_calls": reference_prefix_calls,
            "actual_cfg_prefix_calls": actual_prefix_calls,
            "cfg_prefix_calls_saved": reference_prefix_calls - actual_prefix_calls,
            "decision_syncs_per_step": (
                self._counts["decision_device_to_host_syncs"] / total_steps_seen
                if total_steps_seen
                else 0.0
            ),
            "observed_delta_y_stats": observed["delta_y"],
            "observed_delta_x_stats": observed["delta_x"],
            "observed_branch_error_stats": observed["branch_error"],
            "decision_delta_y_stats": decision["delta_y"],
            "decision_delta_x_stats": decision["delta_x"],
            "decision_branch_error_stats": decision["branch_error"],
            "gamma_stats": gamma,
            "current_history_storage_bytes": self._current_history_storage_bytes,
            "peak_history_storage_bytes": self._peak_history_storage_bytes,
            "history_tensor_count": self._history_tensor_count,
            "history_unique_storage_count": self._history_unique_storage_count,
            "accumulated_error_current": self.state.accumulated_error,
            "branch_aggregation": self.branch_aggregation,
            "error_choice": self.error_choice,
            "reuse_threshold": self.reuse_threshold,
            "dcta_enabled": self.dcta_enabled,
            "retention_full_last_step_idx": retention_last,
            "retention_full_step_count": retention_last + 1,
            "timing_semantics": "host_dispatch_only",
            "debug_sync_overhead_enabled": self.debug_jsonl is not None,
            **self._timings,
            "config": {
                "enabled": self.enabled,
                "total_steps": self.total_steps,
                "total_blocks": self.total_blocks,
                "probe_depth": self.probe_depth,
                "reuse_threshold": self.reuse_threshold,
                "error_choice": self.error_choice,
                "branch_aggregation": self.branch_aggregation,
                "ret_ratio": self.ret_ratio,
                "force_last_step_full": self.force_last_step_full,
                "dcta_enabled": self.dcta_enabled,
                "gamma_min": self.gamma_min,
                "gamma_max": self.gamma_max,
                "eps": self.eps,
                "clone_history": self.clone_history,
                "max_error_samples": self.max_error_samples,
                "schedule_variant": self.schedule_variant,
                "decision_rule": "strict_lt",
                "share_cfg_prefix": self.share_cfg_prefix,
                "force_full": self.force_full,
            },
        }
        return result
