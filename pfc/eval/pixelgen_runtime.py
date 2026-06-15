from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch
import torch.nn as nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


PIXELGEN_SOLVER_STAGES = ("heun_predictor", "heun_corrector")


@dataclass
class PixelGenRuntimeConfig:
    pixelgen_dir: Path
    ckpt_path: Path
    run_id: str = ""
    run_dir: Path | None = None
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.1
    num_classes: int = 1000
    use_bottleneck: bool = False
    in_context_len: int = 32
    in_context_start: int = 8
    cfg: float = 2.25
    timeshift: float = 2.0
    guidance_interval_min: float = 0.1
    guidance_interval_max: float = 0.9
    t_eps: float = 0.05
    steps: int = 50
    batch_size: int = 4
    seed: int = 0
    cache_interval: int = 2
    active_t_min: float | None = None
    active_t_max: float | None = None
    noise_scale: float = 1.0
    exact_heun: bool = True
    enable_compile: bool = False


def load_pixelgen_model(config: PixelGenRuntimeConfig, device: torch.device) -> nn.Module:
    if not config.ckpt_path.is_file():
        raise FileNotFoundError(f"Missing PixelGen checkpoint: {config.ckpt_path}")
    module = _import_pixelgen_jit(config.pixelgen_dir, enable_compile=config.enable_compile)
    denoiser_cls = getattr(module, "JiT")
    denoiser = denoiser_cls(
        input_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        attn_drop=config.attn_drop,
        proj_drop=config.proj_drop,
        num_classes=config.num_classes,
        use_bottleneck=config.use_bottleneck,
        in_context_len=config.in_context_len,
        in_context_start=config.in_context_start,
    )
    checkpoint = torch.load(config.ckpt_path, map_location="cpu")
    state_dict, source = select_pixelgen_denoiser_state_dict(checkpoint, denoiser)
    denoiser.load_state_dict(state_dict, strict=True)
    denoiser._pfc_checkpoint_source = source  # type: ignore[attr-defined]
    denoiser.to(device)
    denoiser.eval()
    return denoiser


def select_pixelgen_denoiser_state_dict(
    checkpoint: Any,
    denoiser: nn.Module,
) -> tuple[dict[str, torch.Tensor], str]:
    raw = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint else checkpoint
    if not isinstance(raw, Mapping):
        raise ValueError(f"Unsupported PixelGen checkpoint type: {type(checkpoint).__name__}")

    expected_keys = set(denoiser.state_dict().keys())
    ordered_prefixes = (
        "ema_denoiser.",
        "denoiser.",
        "model.ema_denoiser.",
        "model.denoiser.",
    )
    for prefix in ordered_prefixes:
        stripped = {
            key[len(prefix) :]: value
            for key, value in raw.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if stripped and (set(stripped) & expected_keys):
            return dict(stripped), prefix.rstrip(".")

    raw_denoiser = {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and key in expected_keys
    }
    if raw_denoiser:
        return dict(raw_denoiser), "raw_denoiser"

    keys = [str(key) for key in list(raw.keys())[:20]]
    raise ValueError(
        "Could not find PixelGen denoiser weights in checkpoint. "
        "Expected prefixes ema_denoiser., denoiser., model.ema_denoiser., model.denoiser., "
        f"or a raw denoiser state_dict. First checkpoint keys: {keys}"
    )


def policy_for_pixelgen_modules(
    config: PixelGenRuntimeConfig,
    selected_modules: list[str],
) -> FixedIntervalCachePolicy:
    return FixedIntervalCachePolicy(
        enabled=bool(selected_modules),
        interval=config.cache_interval,
        cache_modules=set(selected_modules),
        active_t_min=config.active_t_min,
        active_t_max=config.active_t_max,
        solver_stages=set(PIXELGEN_SOLVER_STAGES),
    )


def sample_pixelgen_heun_jit(
    denoiser: nn.Module,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: PixelGenRuntimeConfig,
    *,
    cache_state: RuntimeCacheState | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if config.steps <= 0:
        raise ValueError("PixelGen steps must be positive")
    if labels.shape[0] != noise.shape[0]:
        raise ValueError(f"labels batch {labels.shape[0]} does not match noise batch {noise.shape[0]}")

    if cache_state is not None:
        cache_state.clear_entries()

    records: list[dict[str, Any]] = []
    x = noise
    batch_size = x.shape[0]
    condition, uncondition = make_pixelgen_label_conditions(labels, config.num_classes)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    steps = pixelgen_heun_timesteps(config.steps, config.timeshift, device=x.device, dtype=x.dtype)
    v_hat: torch.Tensor | None = None

    for step_idx, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
        dt = t_next_scalar - t_cur_scalar
        t_cur = t_cur_scalar.repeat(batch_size)
        t_next = t_next_scalar.repeat(batch_size)
        t_value = _scalar_float(t_cur_scalar)
        t_next_value = _scalar_float(t_next_scalar)
        cfg_scale = _guidance_for_t(t_value, config)
        predictor_ran = step_idx == 0 or config.exact_heun or v_hat is None

        if predictor_ran:
            if cache_state is not None:
                cache_state.set_context(step_idx, t_value, "cfg_cat", solver_stage="heun_predictor")
            v = _cfg_cat_velocity(denoiser, x, t_cur, cfg_condition, cfg_scale, config.t_eps)
        else:
            v = v_hat

        x_hat = x + v * dt
        corrector_ran = step_idx < config.steps - 1
        if corrector_ran:
            if cache_state is not None:
                cache_state.set_context(step_idx, t_next_value, "cfg_cat", solver_stage="heun_corrector")
            v_hat = _cfg_cat_velocity(denoiser, x_hat, t_next, cfg_condition, cfg_scale, config.t_eps)
            v = (v + v_hat) / 2.0
            x = x + v * dt
        else:
            x = x + v * dt

        records.append(
            {
                "record_type": "pixelgen_heun_step",
                "step_idx": step_idx,
                "t": t_value,
                "t_next": t_next_value,
                "dt": _scalar_float(dt),
                "cfg_enabled": cfg_scale != 1.0,
                "cfg_scale": config.cfg,
                "predictor_ran": predictor_ran,
                "corrector_ran": corrector_ran,
                "branch": "cfg_cat",
            }
        )
    return x, records


def pixelgen_heun_timesteps(
    steps: int,
    timeshift: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    last_step = 1.0 / steps
    timesteps = torch.linspace(0.0, 1.0 - last_step, steps, device=device, dtype=dtype)
    timesteps = torch.cat([timesteps, torch.ones(1, device=device, dtype=dtype)], dim=0)
    return shift_respace_fn(timesteps, timeshift)


def shift_respace_fn(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    return t / (t + (1.0 - t) * shift)


def make_pixelgen_label_conditions(labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    condition = labels.to(device=labels.device, dtype=torch.long)
    uncondition = torch.full_like(condition, int(num_classes), dtype=torch.long)
    return condition, uncondition


def _cfg_cat_velocity(
    denoiser: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg_condition: torch.Tensor,
    cfg_scale: float,
    t_eps: float,
) -> torch.Tensor:
    cfg_x = torch.cat([x, x], dim=0)
    cfg_t = t.repeat(2)
    out = denoiser(cfg_x, cfg_t, cfg_condition)
    out = (out - cfg_x) / (1.0 - cfg_t.view(-1, 1, 1, 1)).clamp_min(t_eps)
    return simple_guidance_fn(out, cfg_scale)


def simple_guidance_fn(out: torch.Tensor, cfg: float) -> torch.Tensor:
    uncondition, condition = out.chunk(2, dim=0)
    return uncondition + cfg * (condition - uncondition)


def _guidance_for_t(t_value: float, config: PixelGenRuntimeConfig) -> float:
    if t_value > config.guidance_interval_min and t_value <= config.guidance_interval_max:
        return config.cfg
    return 1.0


def _scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _import_pixelgen_jit(pixelgen_dir: Path, *, enable_compile: bool) -> ModuleType:
    root = pixelgen_dir.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with _maybe_disable_torch_compile(enable_compile):
        from src.models.transformer import JiT as jit_module  # type: ignore

    return jit_module


@contextmanager
def _maybe_disable_torch_compile(enable_compile: bool):
    if enable_compile:
        yield
        return
    original_compile = getattr(torch, "compile", None)

    def identity_compile(fn=None, *args: Any, **kwargs: Any):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    torch.compile = identity_compile  # type: ignore[assignment]
    try:
        yield
    finally:
        if original_compile is not None:
            torch.compile = original_compile  # type: ignore[assignment]
