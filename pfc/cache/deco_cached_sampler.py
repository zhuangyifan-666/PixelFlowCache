from __future__ import annotations

from typing import Any

import torch

from pfc.cache.cache_state import RuntimeCacheState
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
        log_diagnostics: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache_state = cache_state
        self.velocity_writer = velocity_writer
        self.frequency_writer = frequency_writer
        self.step_writer = step_writer
        self.log_diagnostics = log_diagnostics

    def _impl_sampling(self, net: torch.nn.Module, noise: torch.Tensor, condition: Any, uncondition: Any):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise]
        v_trajs = []
        prev_v: torch.Tensor | None = None
        if self.cache_state is not None:
            self.cache_state.clear_entries()

        for i, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next_scalar - t_cur_scalar
            t_cur = t_cur_scalar.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            w = self.w_scheduler.w(t_cur) if self.w_scheduler else 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            t_value = float(t_cur_scalar.detach().float().cpu().item())
            t_next_value = float(t_next_scalar.detach().float().cpu().item())
            dt_value = float(dt.detach().float().cpu().item())
            cfg_enabled = bool(t_cur[0] > self.guidance_interval_min and t_cur[0] <= self.guidance_interval_max)
            guidance = self.guidance if cfg_enabled else 1.0

            if self.cache_state is not None:
                self.cache_state.set_context(i, t_value, "cfg_cat", solver_stage="euler")
            out_raw = net(cfg_x, cfg_t, cfg_condition)
            out_uncond, out_cond = out_raw.chunk(2, dim=0)
            v = self.guidance_fn(out_raw, guidance)

            if self.log_diagnostics:
                self._write_velocity_records(i, t_value, t_next_value, dt_value, cfg_enabled, out_uncond, out_cond, v)
                self._write_frequency_record(i, t_value, t_next_value, dt_value, cfg_enabled, v, prev_v)
                self._write_step_record(i, t_value, t_next_value, dt_value, cfg_enabled)
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
