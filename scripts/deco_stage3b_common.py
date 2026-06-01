from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BOOTSTRAP_DECO_DIR = Path(os.environ.get("PFC_DECO_DIR", ROOT / "third_party/DeCo")).resolve()
if str(_BOOTSTRAP_DECO_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_DECO_DIR))

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.deco_cached_sampler import CachedDeCoEulerSampler
from pfc.cache.deco_wrap import deco_cache_unit_category, is_safe_deco_cache_unit, parse_deco_cache_spec, wrap_deco_modules
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
from pfc.diagnostics.velocity_error import frequency_error_stats, image_error_stats
from pfc.profiling.jsonl import JsonlWriter
from pfc.profiling.run_meta import collect_env_info, collect_git_status, write_run_meta
from pfc.utils.seeding import set_seed


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
class DeCoStage3BConfig:
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
    timing_repeats: int = 2
    warmup_runs: int = 1
    resolution: int = 256
    save_diagnostics: bool = True


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_optional_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer list must not be empty")
    return items


def parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("string list must not be empty")
    return items


def make_run_id(seed: int, steps: int, suffix: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_suffix = "".join(ch if ch.isalnum() else "-" for ch in suffix).strip("-")
    return f"{stamp}_seed{seed}_steps{steps}{'_' + clean_suffix if clean_suffix else ''}"


def detect_deco_ckpt() -> Path:
    env_path = os.environ.get("PFC_DECO_CKPT")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()
    expected_files = [
        ROOT / "ckpts/DeCo/imagenet256_epoch800.ckpt",
        ROOT / "ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt",
    ]
    for path in expected_files:
        if path.is_file():
            return path.resolve()
    matches = sorted((ROOT / "ckpts").glob("**/*.ckpt"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError("DeCo checkpoint not found under ckpts/DeCo")


def default_deco_dir() -> Path:
    return Path(os.environ.get("PFC_DECO_DIR", ROOT / "third_party/DeCo")).resolve()


def default_deco_config() -> Path:
    return Path(os.environ.get("PFC_DECO_CONFIG", default_deco_dir() / "configs_c2i/DeCo_XL.yaml")).resolve()


def setup_deco_pythonpath(deco_dir: Path) -> None:
    deco_text = str(deco_dir)
    if deco_text not in sys.path:
        sys.path.insert(0, deco_text)
    os.environ["PYTHONPATH"] = f"{ROOT}:{deco_dir}:{os.environ.get('PYTHONPATH', '')}"


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


def build_deco_denoiser_structure(config: DeCoStage3BConfig, device: torch.device) -> torch.nn.Module:
    setup_deco_pythonpath(config.deco_dir)
    from src.models.transformer.dit_c2i_DeCo import PixNerDiT

    denoiser = PixNerDiT(**load_model_args_from_config(config.config_path))
    denoiser.eval().to(device)
    return denoiser


def load_deco_denoiser(config: DeCoStage3BConfig, device: torch.device) -> torch.nn.Module:
    denoiser = build_deco_denoiser_structure(config, device)
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
    config: DeCoStage3BConfig,
    cache_state: RuntimeCacheState | None = None,
    velocity_writer: Any | None = None,
    frequency_writer: Any | None = None,
    step_writer: Any | None = None,
    log_diagnostics: bool = False,
) -> CachedDeCoEulerSampler:
    setup_deco_pythonpath(config.deco_dir)
    from src.diffusion.base.guidance import simple_guidance_fn
    from src.diffusion.flow_matching.sampling import ode_step_fn
    from src.diffusion.flow_matching.scheduling import LinearScheduler

    return CachedDeCoEulerSampler(
        cache_state=cache_state,
        velocity_writer=velocity_writer,
        frequency_writer=frequency_writer,
        step_writer=step_writer,
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


def make_inputs(config: DeCoStage3BConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(config.seed)
    labels = torch.arange(config.num_samples, device=device, dtype=torch.long) % DECO_DEFAULT_MODEL_ARGS["num_classes"]
    uncondition = torch.full_like(labels, DECO_DEFAULT_MODEL_ARGS["num_classes"])
    noise = torch.randn(
        config.num_samples,
        3,
        config.resolution,
        config.resolution,
        device=device,
        dtype=torch.float32,
    )
    return labels, uncondition, noise


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


def policy_for_modules(config: DeCoStage3BConfig, selected_modules: list[str]) -> FixedIntervalCachePolicy:
    return FixedIntervalCachePolicy(
        enabled=bool(selected_modules),
        interval=config.cache_interval,
        cache_modules=set(selected_modules),
        active_t_min=config.active_t_min,
        active_t_max=config.active_t_max,
        solver_stages={"euler"},
    )


def time_deco_sampling(
    denoiser: torch.nn.Module,
    sampler: CachedDeCoEulerSampler,
    labels: torch.Tensor,
    uncondition: torch.Tensor,
    noise: torch.Tensor,
    config: DeCoStage3BConfig,
    cache_state: RuntimeCacheState | None = None,
) -> dict[str, Any]:
    device = noise.device
    for _idx in range(config.warmup_runs):
        if cache_state is not None:
            cache_state.clear_entries()
            cache_state.reset_stats()
        with torch.no_grad():
            _sample_batches(denoiser, sampler, labels, uncondition, noise, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    latencies: list[float] = []
    output: torch.Tensor | None = None
    for _idx in range(config.timing_repeats):
        if cache_state is not None:
            cache_state.clear_entries()
            cache_state.reset_stats()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = _sample_batches(denoiser, sampler, labels, uncondition, noise, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
    if output is None:
        raise RuntimeError("timing_repeats must be positive")
    return {
        "latencies_sec": latencies,
        "latency_mean_sec": statistics.fmean(latencies),
        "latency_median_sec": statistics.median(latencies),
        "latency_min_sec": min(latencies),
        "latency_max_sec": max(latencies),
        "timing_repeats": config.timing_repeats,
        "warmup_runs": config.warmup_runs,
        "output": output.detach().cpu(),
    }


def compare_outputs(reference: torch.Tensor, output: torch.Tensor, reference_latency: float, method_latency: float) -> dict[str, Any]:
    image_stats = image_error_stats(output, reference)
    frequency_delta = frequency_error_stats(output, reference)
    return {
        "same_seed_mse": image_stats["mse"],
        "same_seed_mae": image_stats["mae"],
        "same_seed_rel_l2": image_stats["rel_l2"],
        "same_seed_psnr": image_stats["psnr"],
        "frequency_delta_bands": frequency_delta,
        "speedup": reference_latency / method_latency if method_latency > 0 else float("inf"),
    }


def run_no_cache(
    config: DeCoStage3BConfig,
    labels: torch.Tensor,
    uncondition: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    denoiser = load_deco_denoiser(config, device)
    sampler = build_deco_sampler(config, log_diagnostics=False)
    timing = time_deco_sampling(denoiser, sampler, labels, uncondition, noise, config)
    del denoiser
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return timing


def run_cached(
    config: DeCoStage3BConfig,
    labels: torch.Tensor,
    uncondition: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    denoiser = load_deco_denoiser(config, device)
    names = candidate_names(denoiser)
    selected_modules = parse_deco_cache_spec(config.cache_units, names)
    cache_state = RuntimeCacheState(model_name="DeCo", enabled=bool(selected_modules))
    policy = policy_for_modules(config, selected_modules)
    wrapped_modules = wrap_deco_modules(denoiser, cache_state, policy, selected_modules)
    velocity_writer = JsonlWriter(config.run_dir / "velocity_stats.jsonl") if config.save_diagnostics else None
    frequency_writer = JsonlWriter(config.run_dir / "frequency_stats.jsonl") if config.save_diagnostics else None
    step_writer = JsonlWriter(config.run_dir / "step_stats.jsonl") if config.save_diagnostics else None
    try:
        sampler = build_deco_sampler(
            config,
            cache_state=cache_state,
            velocity_writer=velocity_writer,
            frequency_writer=frequency_writer,
            step_writer=step_writer,
            log_diagnostics=config.save_diagnostics,
        )
        timing = time_deco_sampling(denoiser, sampler, labels, uncondition, noise, config, cache_state=cache_state)
        cache_stats = cache_state.summary()
    finally:
        for writer in (velocity_writer, frequency_writer, step_writer):
            if writer is not None:
                writer.close()
    del denoiser
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return timing, cache_stats, wrapped_modules


def json_config(config: DeCoStage3BConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key in ("deco_dir", "ckpt_path", "config_path", "run_dir"):
        raw[key] = str(raw[key])
    return raw


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_common_meta(config: DeCoStage3BConfig, script_name: str, extra: dict[str, Any] | None = None) -> None:
    meta = {
        **collect_git_status(ROOT, ROOT / "third_party/JiT", config.deco_dir),
        "env": collect_env_info(),
        "script": script_name,
        "model_name": "DeCo",
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "checkpoint": str(config.ckpt_path),
        "config": str(config.config_path),
    }
    if extra:
        meta.update(extra)
    write_run_meta(config.run_dir / "meta.json", meta)


def _sample_batches(
    denoiser: torch.nn.Module,
    sampler: CachedDeCoEulerSampler,
    labels: torch.Tensor,
    uncondition: torch.Tensor,
    noise: torch.Tensor,
    config: DeCoStage3BConfig,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for batch_start in range(0, config.num_samples, config.batch_size):
        batch_end = min(batch_start + config.batch_size, config.num_samples)
        batch_noise = noise[batch_start:batch_end].clone()
        batch_labels = labels[batch_start:batch_end]
        batch_uncondition = uncondition[batch_start:batch_end]
        outputs.append(sampler(denoiser, batch_noise, batch_labels, batch_uncondition).detach())
    return torch.cat(outputs, dim=0)
