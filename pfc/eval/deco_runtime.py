from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.deco_wrap import deco_cache_unit_category, is_safe_deco_cache_unit
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy


DECO_DEFAULT_MODEL_ARGS = {
    "in_channels": 3,
    "patch_size": 16,
    "num_groups": 16,
    "hidden_size": 1152,
    "hidden_size_x": 32,
    "num_blocks": 31,
    "num_cond_blocks": 28,
    "nerf_mlpratio": 2,
    "num_classes": 1000,
}


@dataclass
class DeCoRuntimeConfig:
    deco_dir: Path
    ckpt_path: Path
    config_path: Path
    run_id: str
    run_dir: Path
    num_samples: int = 8
    batch_size: int = 4
    steps: int = 20
    seed: int = 0
    cfg: float = 3.2
    cfg_interval_min: float = 0.1
    cfg_interval_max: float = 1.0
    cache_interval: int = 2
    active_t_min: float | None = 0.2
    active_t_max: float | None = 1.0
    cache_units: str = "backbone_blocks"
    resolution: int = 256
    dynamic_proxy_downsample: int = 64


def setup_deco_pythonpath(deco_dir: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    deco_text = str(deco_dir.resolve())
    if deco_text not in sys.path:
        sys.path.insert(0, deco_text)
    os.environ["PYTHONPATH"] = f"{root}:{deco_text}:{os.environ.get('PYTHONPATH', '')}"


def load_model_args_from_config(config_path: Path) -> dict[str, Any]:
    args = dict(DECO_DEFAULT_MODEL_ARGS)
    if not config_path.exists():
        return args
    try:
        import yaml
    except ImportError:
        return args
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args = (((data or {}).get("model") or {}).get("denoiser") or {}).get("init_args") or {}
    args.update({key: value for key, value in model_args.items() if key in args})
    return args


def load_deco_denoiser(config: DeCoRuntimeConfig, device: torch.device) -> torch.nn.Module:
    setup_deco_pythonpath(config.deco_dir)
    from src.models.transformer.dit_c2i_DeCo import PixNerDiT

    denoiser = PixNerDiT(**load_model_args_from_config(config.config_path))
    denoiser.eval().to(device)
    checkpoint = torch.load(config.ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    ema_state = {
        key.removeprefix("ema_denoiser."): value
        for key, value in state_dict.items()
        if key.startswith("ema_denoiser.")
    }
    if not ema_state:
        ema_state = {
            key.removeprefix("denoiser."): value
            for key, value in state_dict.items()
            if key.startswith("denoiser.")
        }
    if not ema_state:
        raise RuntimeError(f"No denoiser or ema_denoiser weights found in {config.ckpt_path}")
    missing, unexpected = denoiser.load_state_dict(ema_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected DeCo checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"Warning: missing DeCo denoiser keys: {missing[:10]}")
    return denoiser


def build_deco_sampler(
    config: DeCoRuntimeConfig,
    cache_state: RuntimeCacheState | None = None,
    velocity_writer: Any | None = None,
    frequency_writer: Any | None = None,
    step_writer: Any | None = None,
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None,
    dynamic_decision_writer: Any | None = None,
    log_diagnostics: bool = False,
) -> Any:
    setup_deco_pythonpath(config.deco_dir)
    from pfc.cache.deco_cached_sampler import CachedDeCoEulerSampler
    from src.diffusion.base.guidance import simple_guidance_fn
    from src.diffusion.flow_matching.sampling import ode_step_fn
    from src.diffusion.flow_matching.scheduling import LinearScheduler

    return CachedDeCoEulerSampler(
        cache_state=cache_state,
        velocity_writer=velocity_writer,
        frequency_writer=frequency_writer,
        step_writer=step_writer,
        dynamic_policy=dynamic_policy,
        dynamic_decision_writer=dynamic_decision_writer,
        dynamic_proxy_downsample=config.dynamic_proxy_downsample,
        log_diagnostics=log_diagnostics,
        num_steps=config.steps,
        guidance=config.cfg,
        guidance_interval_min=config.cfg_interval_min,
        guidance_interval_max=config.cfg_interval_max,
        scheduler=LinearScheduler(),
        w_scheduler=LinearScheduler(),
        guidance_fn=simple_guidance_fn,
        step_fn=ode_step_fn,
    )


def candidate_modules(net: torch.nn.Module) -> list[tuple[str, torch.nn.Module, str, bool]]:
    rows: list[tuple[str, torch.nn.Module, str, bool]] = []
    for name, module in net.named_modules():
        if not name:
            continue
        category = deco_cache_unit_category(name, module)
        cacheable = is_safe_deco_cache_unit(name, module)
        if cacheable or category in {"norm_or_modulation", "tiny_module"}:
            rows.append((name, module, category, cacheable))
    return rows


def candidate_names(net: torch.nn.Module) -> list[str]:
    return [name for name, _module, _category, cacheable in candidate_modules(net) if cacheable]


def policy_for_modules(config: DeCoRuntimeConfig, selected_modules: list[str]) -> FixedIntervalCachePolicy:
    return FixedIntervalCachePolicy(
        enabled=bool(selected_modules),
        interval=config.cache_interval,
        cache_modules=set(selected_modules),
        active_t_min=config.active_t_min,
        active_t_max=config.active_t_max,
        solver_stages={"euler"},
    )
