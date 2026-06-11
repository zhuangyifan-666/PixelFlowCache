from __future__ import annotations

import sys
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.dynamic_proxy import maybe_downsample_proxy, proxy_from_image_state
from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy
from pfc.diagnostics.tensor_stats import l2_norm


@dataclass
class JiTRuntimeConfig:
    jit_dir: Path
    ckpt_dir: Path
    run_id: str
    run_dir: Path
    preview_dir: Path
    model: str = "JiT-B/16"
    img_size: int = 256
    num_samples: int = 8
    batch_size: int = 4
    steps: int = 20
    seed: int = 0
    cfg: float = 3.0
    interval_min: float = 0.1
    interval_max: float = 1.0
    noise_scale: float = 1.0
    cache_interval: int = 2
    cache_layers: str = "all"
    cache_branches: str = "cond,uncond"
    active_t_min: float | None = 0.1
    active_t_max: float | None = 0.8
    active_step_min: int | None = None
    active_step_max: int | None = None
    active_window_warmup_refreshes: int = 0
    dynamic_proxy_downsample: int = 64
    warmup_runs: int = 0
    save_previews: bool = False


def _build_model_args(config: JiTRuntimeConfig) -> Namespace:
    return Namespace(
        model=config.model,
        img_size=config.img_size,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=config.noise_scale,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="euler",
        num_sampling_steps=config.steps,
        cfg=config.cfg,
        interval_min=config.interval_min,
        interval_max=config.interval_max,
    )


def load_jit_model(config: JiTRuntimeConfig, device: torch.device) -> Any:
    jit_dir = config.jit_dir.resolve()
    if str(jit_dir) not in sys.path:
        sys.path.insert(0, str(jit_dir))
    from denoiser import Denoiser  # type: ignore

    model = Denoiser(_build_model_args(config))
    checkpoint = torch.load(config.ckpt_dir / "checkpoint-last.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if "model_ema1" in checkpoint:
        ema_state = model.state_dict()
        for name, _param in model.named_parameters():
            if name in checkpoint["model_ema1"]:
                ema_state[name] = checkpoint["model_ema1"][name]
        model.load_state_dict(ema_state)
    model.to(device)
    model.eval()
    return model


def cfg_enabled(t_value: float, low: float, high: float) -> bool:
    return (t_value < high) and ((low == 0.0) or (t_value > low))


def sample_jit(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: JiTRuntimeConfig,
    mode: str,
    cache_state: RuntimeCacheState | None = None,
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None,
    dynamic_decision_writer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    outputs: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    device = noise.device
    timesteps = torch.linspace(0.0, 1.0, config.steps + 1, device=device, dtype=noise.dtype)

    for batch_start in range(0, config.num_samples, config.batch_size):
        batch_end = min(batch_start + config.batch_size, config.num_samples)
        z = noise[batch_start:batch_end].clone()
        batch_labels = labels[batch_start:batch_end]
        if cache_state is not None:
            cache_state.clear_entries()
        if dynamic_policy is not None:
            dynamic_policy.clear_batch()

        for step_idx in range(config.steps):
            t_scalar = timesteps[step_idx]
            t_next_scalar = timesteps[step_idx + 1]
            dt = t_next_scalar - t_scalar
            t_value = float(t_scalar.detach().float().cpu().item())
            t_next_value = float(t_next_scalar.detach().float().cpu().item())
            dt_value = float(dt.detach().float().cpu().item())
            t = t_scalar.expand(z.shape[0], 1, 1, 1)
            cfg_active = cfg_enabled(t_value, config.interval_min, config.interval_max)
            cfg_scale_interval = config.cfg if cfg_active else 1.0

            if dynamic_policy is not None:
                _update_dynamic_policy(
                    dynamic_policy,
                    z,
                    step_idx,
                    t_value,
                    branch="cond" if dynamic_policy.per_branch else "global",
                    max_size=config.dynamic_proxy_downsample,
                    writer=dynamic_decision_writer,
                    batch_start=batch_start,
                    batch_end=batch_end,
                )
            if cache_state is not None:
                cache_state.set_context(step_idx, t_value, "cond", solver_stage="euler")
            x_cond = model.net(z, t.flatten(), batch_labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(model.t_eps)

            if dynamic_policy is not None and dynamic_policy.per_branch:
                _update_dynamic_policy(
                    dynamic_policy,
                    z,
                    step_idx,
                    t_value,
                    branch="uncond",
                    max_size=config.dynamic_proxy_downsample,
                    writer=dynamic_decision_writer,
                    batch_start=batch_start,
                    batch_end=batch_end,
                )
            if cache_state is not None:
                cache_state.set_context(step_idx, t_value, "uncond", solver_stage="euler")
            null_labels = torch.full_like(batch_labels, model.num_classes)
            x_uncond = model.net(z, t.flatten(), null_labels)
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(model.t_eps)

            v_cfg = v_uncond + cfg_scale_interval * (v_cond - v_uncond)
            records.append(
                {
                    "record_type": "jit_step",
                    "mode": mode,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "step_idx": step_idx,
                    "t": t_value,
                    "t_next": t_next_value,
                    "dt": dt_value,
                    "cfg_enabled": cfg_active,
                    "cfg_scale": config.cfg,
                    "velocity_l2": l2_norm(v_cfg),
                }
            )
            z = z + dt * v_cfg
        outputs.append(z.detach())
    return torch.cat(outputs, dim=0), records


def _update_dynamic_policy(
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy,
    z: torch.Tensor,
    step_idx: int,
    t_value: float,
    branch: str,
    max_size: int,
    writer: Callable[[dict[str, Any]], None] | None,
    batch_start: int,
    batch_end: int,
) -> None:
    proxy = maybe_downsample_proxy(proxy_from_image_state(z), max_size=max_size)
    decision = dynamic_policy.update(proxy, step_idx=step_idx, t=t_value, branch=branch)
    if writer is not None:
        payload = asdict(decision)
        payload.update(
            {
                "record_type": "dynamic_cache_decision",
                "model_name": "JiT",
                "batch_start": batch_start,
                "batch_end": batch_end,
                "policy": dynamic_policy.policy_name,
                "proxy_shape": [int(dim) for dim in proxy.shape],
            }
        )
        writer(payload)
