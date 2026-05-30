from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from pfc.profiling.feature_recorder import FeatureRecorder
from pfc.profiling.frequency import fft_frequency_bands
from pfc.profiling.jsonl import JsonlWriter
from pfc.profiling.module_selectors import categorize_deco_module, select_deco_candidate_modules
from pfc.profiling.velocity_recorder import VelocityRecorder

try:
    from src.diffusion.flow_matching.sampling import EulerSampler
except ImportError as exc:  # pragma: no cover - only used inside DeCo runtime
    raise ImportError("ProfiledEulerSampler must be imported with third_party/DeCo on PYTHONPATH") from exc


class ProfiledEulerSampler(EulerSampler):
    def _impl_sampling(self, net: torch.nn.Module, noise: torch.Tensor, condition: Any, uncondition: Any):
        log_dir = Path(os.environ.get("PFC_DECO_PROFILE_LOG_DIR", "logs/stage1/deco/default"))
        log_dir.mkdir(parents=True, exist_ok=True)
        feature_writer = JsonlWriter(log_dir / "feature_stats.jsonl")
        velocity_writer = JsonlWriter(log_dir / "velocity_stats.jsonl")
        frequency_writer = JsonlWriter(log_dir / "frequency_stats.jsonl")
        step_writer = JsonlWriter(log_dir / "step_stats.jsonl")
        velocity_recorder = VelocityRecorder(velocity_writer)

        candidates = select_deco_candidate_modules(net)
        module_categories = {name: categorize_deco_module(name, module) for name, module in candidates}
        (log_dir / "module_candidates.json").write_text(
            json.dumps(
                [
                    {
                        "name": name,
                        "module_kind": module.__class__.__name__,
                        "category": module_categories[name],
                    }
                    for name, module in candidates
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        candidate_names = {name for name, _module in candidates}
        recorder = FeatureRecorder(
            module_filter=lambda name, _module: name in candidate_names,
            writer=feature_writer,
            model_name="DeCo",
            previous_on_cpu=True,
            previous_dtype="float16",
            split_batch_dim0=True,
        )
        recorder.attach(net)

        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise]
        v_trajs = []
        prev_v: torch.Tensor | None = None

        try:
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

                recorder.set_context(
                    i,
                    t_value,
                    solver_stage="euler",
                    cfg_branch="cfg_cat",
                    extra={"cfg_cat_batch_size": batch_size, "module_categories": module_categories},
                )
                out_raw = net(cfg_x, cfg_t, cfg_condition)
                out_uncond, out_cond = out_raw.chunk(2, dim=0)
                out = self.guidance_fn(out_raw, guidance)
                v = out

                velocity_recorder.log_velocity(
                    model_name="DeCo",
                    step_idx=i,
                    t=t_value,
                    t_next=t_next_value,
                    dt=dt_value,
                    branch="uncond",
                    v=out_uncond,
                    cfg_scale=float(self.guidance),
                    cfg_enabled=cfg_enabled,
                )
                velocity_recorder.log_velocity(
                    model_name="DeCo",
                    step_idx=i,
                    t=t_value,
                    t_next=t_next_value,
                    dt=dt_value,
                    branch="cond",
                    v=out_cond,
                    cfg_scale=float(self.guidance),
                    cfg_enabled=cfg_enabled,
                )
                velocity_recorder.log_velocity(
                    model_name="DeCo",
                    step_idx=i,
                    t=t_value,
                    t_next=t_next_value,
                    dt=dt_value,
                    branch="cfg",
                    v=v,
                    cfg_scale=float(self.guidance),
                    cfg_enabled=cfg_enabled,
                )

                freq_record = {
                    "record_type": "frequency",
                    "model_name": "DeCo",
                    "step_idx": i,
                    "t": t_value,
                    "t_next": t_next_value,
                    "dt": dt_value,
                    "branch": "cfg",
                    "cfg_scale": float(self.guidance),
                    "cfg_enabled": cfg_enabled,
                    "frequency": fft_frequency_bands(v),
                }
                if prev_v is not None:
                    from pfc.profiling.frequency import frequency_delta_bands

                    freq_record["frequency_delta"] = frequency_delta_bands(v, prev_v)
                frequency_writer.write(freq_record)
                prev_v = v.detach().to(dtype=torch.float16, device="cpu")

                step_writer.write(
                    {
                        "record_type": "step",
                        "model_name": "DeCo",
                        "step_idx": i,
                        "t": t_value,
                        "t_next": t_next_value,
                        "dt": dt_value,
                        "cfg_enabled": cfg_enabled,
                        "cfg_scale": float(self.guidance),
                    }
                )

                s = ((1 / dalpha_over_alpha) * v - x) / (sigma**2 - (1 / dalpha_over_alpha) * dsigma_mul_sigma)
                if i < self.num_steps - 1:
                    x = self.step_fn(x, v, dt, s=s, w=w)
                else:
                    x = self.last_step_fn(x, v, dt, s=s, w=w)
                x_trajs.append(x)
                v_trajs.append(v)
            v_trajs.append(torch.zeros_like(x))
        finally:
            recorder.remove()
            feature_writer.close()
            velocity_recorder.close()
            frequency_writer.close()
            step_writer.close()

        return x_trajs, v_trajs
