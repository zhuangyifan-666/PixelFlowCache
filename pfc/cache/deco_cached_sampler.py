from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import torch

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.dynamic_proxy import maybe_downsample_proxy, proxy_from_image_state
from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy
from pfc.diagnostics.frequency import fft_frequency_bands, frequency_delta_bands
from pfc.diagnostics.tensor_stats import summarize_tensor

try:
    from src.diffusion.flow_matching.sampling import EulerSampler
except ImportError as exc:  # pragma: no cover - only used inside DeCo runtime
    raise ImportError("CachedDeCoEulerSampler must be imported with third_party/DeCo on PYTHONPATH") from exc


class CachedDeCoEulerSampler(EulerSampler):
    def __init__(
        self,
        cache_state: RuntimeCacheState | None = None,
        velocity_writer: Any | None = None,
        frequency_writer: Any | None = None,
        step_writer: Any | None = None,
        dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None,
        dynamic_decision_writer: Callable[[dict[str, Any]], None] | None = None,
        dynamic_proxy_downsample: int = 64,
        log_diagnostics: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache_state = cache_state
        self.velocity_writer = velocity_writer
        self.frequency_writer = frequency_writer
        self.step_writer = step_writer
        self.dynamic_policy = dynamic_policy
        self.dynamic_decision_writer = dynamic_decision_writer
        self.dynamic_proxy_downsample = int(dynamic_proxy_downsample)
        self.log_diagnostics = log_diagnostics

    def _impl_sampling(self, net: torch.nn.Module, noise: torch.Tensor, condition: Any, uncondition: Any):
        batch_size = noise.shape[0]
        cpu_steps = self.timesteps.detach().to(device="cpu", dtype=torch.float64)
        timestep_values = [float(value) for value in cpu_steps.tolist()]
        steps = cpu_steps.to(device=noise.device, dtype=noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise]
        v_trajs = []
        prev_v: torch.Tensor | None = None
        if self.cache_state is not None:
            self.cache_state.clear_entries()
        if self.dynamic_policy is not None:
            self.dynamic_policy.clear_batch()

        for i, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next_scalar - t_cur_scalar
            t_cur = t_cur_scalar.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            w = self.w_scheduler.w(t_cur) if self.w_scheduler else 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            t_value = timestep_values[i]
            t_next_value = timestep_values[i + 1]
            dt_value = t_next_value - t_value
            cfg_enabled = (
                t_value > self.guidance_interval_min
                and t_value <= self.guidance_interval_max
            )
            guidance = self.guidance if cfg_enabled else 1.0

            if self.dynamic_policy is not None:
                self._update_dynamic_policy(x, i, t_value)
            if self.cache_state is not None:
                self.cache_state.set_context(i, t_value, "cfg_cat", solver_stage="euler")
            out_raw = net(cfg_x, cfg_t, cfg_condition)
            out_uncond, out_cond = out_raw.chunk(2, dim=0)
            v = self.guidance_fn(out_raw, guidance)

            if self.log_diagnostics:
                self._write_velocity_records(i, t_value, t_next_value, dt_value, cfg_enabled, out_uncond, out_cond, v)
                self._write_frequency_record(i, t_value, t_next_value, dt_value, cfg_enabled, v, prev_v)
                self._write_step_record(i, t_value, t_next_value, dt_value, cfg_enabled)
            if self.log_diagnostics and self.frequency_writer is not None:
                prev_v = v.detach().to(dtype=torch.float16, device="cpu")

            s = ((1 / dalpha_over_alpha) * v - x) / (sigma**2 - (1 / dalpha_over_alpha) * dsigma_mul_sigma)
            if i < self.num_steps - 1:
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs

    def _update_dynamic_policy(self, x: torch.Tensor, step_idx: int, t_value: float) -> None:
        if self.dynamic_policy is None:
            return
        proxy = maybe_downsample_proxy(proxy_from_image_state(x), max_size=self.dynamic_proxy_downsample)
        decision = self.dynamic_policy.update(proxy, step_idx=step_idx, t=t_value, branch="cfg_cat")
        if self.dynamic_decision_writer is not None:
            payload = asdict(decision)
            payload.update(
                {
                    "record_type": "dynamic_cache_decision",
                    "model_name": "DeCo",
                    "policy": self.dynamic_policy.policy_name,
                    "proxy_shape": [int(dim) for dim in proxy.shape],
                }
            )
            self.dynamic_decision_writer(payload)

    def _write_velocity_records(
        self,
        step_idx: int,
        t: float,
        t_next: float,
        dt: float,
        cfg_enabled: bool,
        out_uncond: torch.Tensor,
        out_cond: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if self.velocity_writer is None:
            return
        for branch, tensor in (("uncond", out_uncond), ("cond", out_cond), ("cfg", v)):
            self.velocity_writer.write(
                {
                    "record_type": "velocity",
                    "model_name": "DeCo",
                    "step_idx": step_idx,
                    "t": t,
                    "t_next": t_next,
                    "dt": dt,
                    "branch": branch,
                    "cfg_scale": float(self.guidance),
                    "cfg_enabled": cfg_enabled,
                    "velocity": summarize_tensor(tensor, name=f"velocity:{branch}"),
                }
            )

    def _write_frequency_record(
        self,
        step_idx: int,
        t: float,
        t_next: float,
        dt: float,
        cfg_enabled: bool,
        v: torch.Tensor,
        prev_v: torch.Tensor | None,
    ) -> None:
        if self.frequency_writer is None:
            return
        record: dict[str, Any] = {
            "record_type": "frequency",
            "model_name": "DeCo",
            "step_idx": step_idx,
            "t": t,
            "t_next": t_next,
            "dt": dt,
            "branch": "cfg",
            "cfg_scale": float(self.guidance),
            "cfg_enabled": cfg_enabled,
            "frequency": fft_frequency_bands(v),
        }
        if prev_v is not None:
            record["frequency_delta"] = frequency_delta_bands(v, prev_v)
        self.frequency_writer.write(record)

    def _write_step_record(self, step_idx: int, t: float, t_next: float, dt: float, cfg_enabled: bool) -> None:
        if self.step_writer is None:
            return
        self.step_writer.write(
            {
                "record_type": "step",
                "model_name": "DeCo",
                "step_idx": step_idx,
                "t": t,
                "t_next": t_next,
                "dt": dt,
                "cfg_enabled": cfg_enabled,
                "cfg_scale": float(self.guidance),
                "cache": self.cache_state.summary() if self.cache_state is not None else None,
            }
        )
