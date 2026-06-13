from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DynamicCacheDecision:
    step_idx: int
    t: float
    branch: str
    delta: float
    accumulated_delta: float
    threshold: float
    should_refresh: bool
    reason: str


@dataclass(frozen=True)
class DynamicCacheStats:
    total_steps: int
    refresh_steps: int
    reuse_steps: int
    forced_refresh_steps: int
    mean_delta: float
    mean_accumulated_delta: float
    refresh_ratio: float


def relative_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"relative_l1 expects equal shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    a_f = a.detach().float()
    b_f = b.detach().float().to(device=a_f.device)
    return (a_f - b_f).abs().mean() / b_f.abs().mean().clamp_min(eps)


def make_radial_frequency_grid(
    height: int,
    width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    freq_y = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
    freq_x = torch.fft.fftfreq(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square())
    return radius.to(dtype=dtype)


def _as_float_t(t: float | torch.Tensor, eps: float, time_direction: str = "noise_to_image") -> float:
    value = float(t.detach().float().cpu().item()) if torch.is_tensor(t) else float(t)
    if time_direction == "image_to_noise":
        value = 1.0 - value
    elif time_direction != "noise_to_image":
        raise ValueError(f"Unsupported time_direction: {time_direction}")
    return max(eps, min(1.0, value))


def make_sea_filter(
    height: int,
    width: int,
    t: float | torch.Tensor,
    beta: float,
    eps: float,
    normalize_filter: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    time_direction: str = "noise_to_image",
) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive")
    t_value = _as_float_t(t, eps=eps, time_direction=time_direction)
    a_t = torch.tensor(t_value, device=device, dtype=torch.float32).clamp(eps, 1.0)
    b_t = torch.tensor(1.0 - t_value, device=device, dtype=torch.float32).clamp(eps, 1.0)
    radius = make_radial_frequency_grid(height, width, device=device, dtype=torch.float32).clamp_min(eps)
    spectrum = radius.pow(-float(beta))
    gain = (a_t * spectrum) / (a_t.square() * spectrum + b_t.square()).clamp_min(eps)
    if normalize_filter:
        mean_gain = gain.mean().clamp_min(eps)
        gain = gain / mean_gain
    return gain.to(dtype=dtype).view(1, 1, height, width)


def apply_sea_filter(
    x: torch.Tensor,
    t: float | torch.Tensor,
    beta: float,
    eps: float,
    normalize_filter: bool,
    time_direction: str = "noise_to_image",
) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError("apply_sea_filter expects a torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"apply_sea_filter currently supports BCHW tensors, got shape {tuple(x.shape)}")
    orig_dtype = x.dtype
    work = x.detach().float()
    height, width = int(work.shape[-2]), int(work.shape[-1])
    filt = make_sea_filter(
        height,
        width,
        t,
        beta=beta,
        eps=eps,
        normalize_filter=normalize_filter,
        device=work.device,
        dtype=work.dtype,
        time_direction=time_direction,
    )
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    filtered = torch.fft.ifft2(spectrum * filt, dim=(-2, -1)).real
    return filtered.to(dtype=orig_dtype)


class RawAccumulatedDistancePolicy:
    policy_name = "teacache_style_raw_accumulated_distance"

    def __init__(
        self,
        threshold: float,
        enabled: bool = True,
        eps: float = 1e-6,
        force_first_step_refresh: bool = True,
        force_first_n_steps: int = 0,
        min_t: float | None = None,
        max_t: float | None = None,
        per_branch: bool = False,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if force_first_n_steps < 0:
            raise ValueError("force_first_n_steps must be non-negative")
        self.threshold = float(threshold)
        self.enabled = bool(enabled)
        self.eps = float(eps)
        self.force_first_step_refresh = bool(force_first_step_refresh)
        self.force_first_n_steps = int(force_first_n_steps)
        self.min_t = min_t
        self.max_t = max_t
        self.per_branch = bool(per_branch)
        self._previous: dict[str, torch.Tensor] = {}
        self._accumulated: dict[str, float] = {}
        self._decisions: dict[str, DynamicCacheDecision] = {}
        self._delta_values: list[float] = []
        self._accumulated_values: list[float] = []
        self.total_steps = 0
        self.refresh_steps = 0
        self.reuse_steps = 0
        self.forced_refresh_steps = 0

    def _state_key(self, branch: str) -> str:
        return str(branch) if self.per_branch else "global"

    def _distance_feature(self, proxy_feature: torch.Tensor, t: float) -> torch.Tensor:
        return proxy_feature.detach()

    def _outside_window(self, t: float) -> bool:
        if self.min_t is not None and t < self.min_t:
            return True
        if self.max_t is not None and t >= self.max_t:
            return True
        return False

    def update(
        self,
        proxy_feature: torch.Tensor,
        step_idx: int,
        t: float | torch.Tensor,
        branch: str = "global",
    ) -> DynamicCacheDecision:
        if not torch.is_tensor(proxy_feature):
            raise TypeError("proxy_feature must be a torch.Tensor")
        t_value = float(t.detach().float().cpu().item()) if torch.is_tensor(t) else float(t)
        key = self._state_key(branch)
        feature = self._distance_feature(proxy_feature, t_value).detach()
        previous = self._previous.get(key)
        delta = 0.0
        accumulated = self._accumulated.get(key, 0.0)
        reason = "reuse"
        should_refresh = False
        forced = False

        if not self.enabled:
            should_refresh = True
            forced = True
            reason = "disabled"
            accumulated = 0.0
        elif self._outside_window(t_value):
            should_refresh = True
            forced = True
            reason = "outside_t_window"
            accumulated = 0.0
        elif previous is None:
            should_refresh = True
            forced = True
            reason = "first_observation"
            accumulated = 0.0
        elif self.force_first_step_refresh and step_idx == 0:
            should_refresh = True
            forced = True
            reason = "first_step"
            accumulated = 0.0
        elif step_idx < self.force_first_n_steps:
            should_refresh = True
            forced = True
            reason = "force_first_n_steps"
            accumulated = 0.0
        else:
            delta = float(relative_l1(feature, previous, eps=self.eps).detach().cpu().item())
            accumulated += delta
            if accumulated > self.threshold:
                should_refresh = True
                reason = "threshold_exceeded"
                accumulated = 0.0

        decision = DynamicCacheDecision(
            step_idx=int(step_idx),
            t=t_value,
            branch=str(branch),
            delta=delta,
            accumulated_delta=accumulated,
            threshold=self.threshold,
            should_refresh=should_refresh,
            reason=reason,
        )
        self._previous[key] = feature
        self._accumulated[key] = accumulated
        self._decisions[key] = decision
        self.total_steps += 1
        self._delta_values.append(delta)
        self._accumulated_values.append(accumulated)
        if should_refresh:
            self.refresh_steps += 1
            if forced:
                self.forced_refresh_steps += 1
        else:
            self.reuse_steps += 1
        return decision

    def should_reuse(self, branch: str = "global") -> bool:
        decision = self.current_decision(branch)
        return bool(decision is not None and not decision.should_refresh)

    def current_decision(self, branch: str = "global") -> DynamicCacheDecision | None:
        return self._decisions.get(self._state_key(branch))

    def reset(self) -> None:
        self.clear_batch()
        self._delta_values.clear()
        self._accumulated_values.clear()
        self.total_steps = 0
        self.refresh_steps = 0
        self.reuse_steps = 0
        self.forced_refresh_steps = 0

    def clear_batch(self) -> None:
        self._previous.clear()
        self._accumulated.clear()
        self._decisions.clear()

    def stats(self) -> DynamicCacheStats:
        total = self.total_steps
        mean_delta = sum(self._delta_values) / len(self._delta_values) if self._delta_values else 0.0
        mean_acc = (
            sum(self._accumulated_values) / len(self._accumulated_values)
            if self._accumulated_values
            else 0.0
        )
        return DynamicCacheStats(
            total_steps=total,
            refresh_steps=self.refresh_steps,
            reuse_steps=self.reuse_steps,
            forced_refresh_steps=self.forced_refresh_steps,
            mean_delta=mean_delta,
            mean_accumulated_delta=mean_acc,
            refresh_ratio=self.refresh_steps / total if total else 0.0,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "config": self.to_dict(),
            "stats": asdict(self.stats()),
            "current_decisions": {
                key: asdict(decision) for key, decision in sorted(self._decisions.items())
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "enabled": self.enabled,
            "eps": self.eps,
            "force_first_step_refresh": self.force_first_step_refresh,
            "force_first_n_steps": self.force_first_n_steps,
            "min_t": self.min_t,
            "max_t": self.max_t,
            "per_branch": self.per_branch,
        }


class SeaCacheSpectralDistancePolicy(RawAccumulatedDistancePolicy):
    policy_name = "seacache_style_spectral_accumulated_distance"

    def __init__(
        self,
        threshold: float,
        beta: float = 2.0,
        eps: float = 1e-6,
        enabled: bool = True,
        force_first_step_refresh: bool = True,
        force_first_n_steps: int = 0,
        min_t: float | None = None,
        max_t: float | None = None,
        per_branch: bool = False,
        time_direction: str = "noise_to_image",
        normalize_filter: bool = True,
        feature_layout: str = "bchw",
    ) -> None:
        if feature_layout != "bchw":
            raise ValueError("Only feature_layout='bchw' is supported for the adapted baseline")
        super().__init__(
            threshold=threshold,
            enabled=enabled,
            eps=eps,
            force_first_step_refresh=force_first_step_refresh,
            force_first_n_steps=force_first_n_steps,
            min_t=min_t,
            max_t=max_t,
            per_branch=per_branch,
        )
        self.beta = float(beta)
        self.time_direction = time_direction
        self.normalize_filter = bool(normalize_filter)
        self.feature_layout = feature_layout

    def _distance_feature(self, proxy_feature: torch.Tensor, t: float) -> torch.Tensor:
        return apply_sea_filter(
            proxy_feature,
            t=t,
            beta=self.beta,
            eps=self.eps,
            normalize_filter=self.normalize_filter,
            time_direction=self.time_direction,
        ).detach()

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "beta": self.beta,
                "time_direction": self.time_direction,
                "normalize_filter": self.normalize_filter,
                "feature_layout": self.feature_layout,
            }
        )
        return data
