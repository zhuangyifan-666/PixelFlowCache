#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from pfc.cache.cache_state import RuntimeCacheState  # noqa: E402
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy  # noqa: E402
from pfc.cache.wrap import parse_layer_list, wrap_jit_blocks  # noqa: E402
from pfc.diagnostics.velocity_error import frequency_error_stats, image_error_stats, tensor_error_stats  # noqa: E402
from pfc.profiling.jsonl import JsonlWriter  # noqa: E402
from pfc.profiling.run_meta import collect_env_info, collect_git_status, write_run_meta  # noqa: E402
from pfc.profiling.tensor_stats import l2_norm  # noqa: E402
from pfc.utils.seeding import set_seed  # noqa: E402
from scripts.run_jit_stage2_cache import (  # noqa: E402
    _cfg_enabled,
    _compare_outputs,
    _detect_jit_ckpt_dir,
    _load_jit_model,
    _make_inputs,
    _save_previews,
    _write_json,
)


@dataclass
class Stage2BConfig:
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
    timing_repeats: int = 3
    warmup_runs: int = 1
    diag_full_probe: bool = False
    diag_probe_steps: str = "all"
    log_step_errors: bool = True
    save_previews: bool = True


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _make_run_id(seed: int, steps: int, cache_interval: int, cache_layers: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_layers = "".join(ch if ch.isalnum() else "-" for ch in cache_layers)[:48].strip("-")
    return f"{stamp}_seed{seed}_steps{steps}_i{cache_interval}_{safe_layers or 'layers'}"


def _stage2_compatible_config(config: Stage2BConfig) -> Any:
    # _load_jit_model only reads attributes, so a Stage2BConfig is compatible.
    return config


def _policy_for_config(config: Stage2BConfig, selected_modules: list[str]) -> FixedIntervalCachePolicy:
    branches = {branch.strip() for branch in config.cache_branches.split(",") if branch.strip()}
    return FixedIntervalCachePolicy.from_branches(
        branches,
        enabled=True,
        interval=config.cache_interval,
        cache_modules=set(selected_modules),
        active_t_min=config.active_t_min,
        active_t_max=config.active_t_max,
        active_step_min=config.active_step_min,
        active_step_max=config.active_step_max,
    )


def _predict_v_cfg(
    model: Any,
    z: torch.Tensor,
    labels: torch.Tensor,
    t_scalar: torch.Tensor,
    config: Stage2BConfig,
    cache_state: RuntimeCacheState | None = None,
) -> tuple[torch.Tensor, bool]:
    t_value = float(t_scalar.detach().float().cpu().item())
    t = t_scalar.expand(z.shape[0], 1, 1, 1)
    cfg_active = _cfg_enabled(t_value, config.interval_min, config.interval_max)
    cfg_scale_interval = config.cfg if cfg_active else 1.0
    if cache_state is not None:
        cache_state.set_context(cache_state.current_step_idx, t_value, "cond", solver_stage="euler")
    x_cond = model.net(z, t.flatten(), labels)
    v_cond = (x_cond - z) / (1.0 - t).clamp_min(model.t_eps)
    if cache_state is not None:
        cache_state.set_context(cache_state.current_step_idx, t_value, "uncond", solver_stage="euler")
    null_labels = torch.full_like(labels, model.num_classes)
    x_uncond = model.net(z, t.flatten(), null_labels)
    v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(model.t_eps)
    return v_uncond + cfg_scale_interval * (v_cond - v_uncond), cfg_active


def _sample_plain(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: Stage2BConfig,
    cache_state: RuntimeCacheState | None = None,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    device = noise.device
    timesteps = torch.linspace(0.0, 1.0, config.steps + 1, device=device, dtype=noise.dtype)
    for batch_start in range(0, config.num_samples, config.batch_size):
        batch_end = min(batch_start + config.batch_size, config.num_samples)
        z = noise[batch_start:batch_end].clone()
        batch_labels = labels[batch_start:batch_end]
        if cache_state is not None:
            cache_state.clear_entries()
        for step_idx in range(config.steps):
            t_scalar = timesteps[step_idx]
            t_next_scalar = timesteps[step_idx + 1]
            if cache_state is not None:
                cache_state.set_context(step_idx, float(t_scalar.detach().float().cpu().item()), "cond")
            v_cfg, _cfg_active = _predict_v_cfg(model, z, batch_labels, t_scalar, config, cache_state)
            z = z + (t_next_scalar - t_scalar) * v_cfg
        outputs.append(z.detach())
    return torch.cat(outputs, dim=0)


def _time_repeats(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: Stage2BConfig,
    cache_state: RuntimeCacheState | None = None,
) -> dict[str, Any]:
    device = noise.device
    for _idx in range(config.warmup_runs):
        with torch.no_grad():
            _sample_plain(model, labels, noise, config, cache_state=cache_state)
        if cache_state is not None:
            cache_state.clear_entries()
            cache_state.reset_stats()
    latencies = []
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
            output = _sample_plain(model, labels, noise, config, cache_state=cache_state)
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


def _run_no_cache_with_refs(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: Stage2BConfig,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    outputs: list[torch.Tensor] = []
    refs: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(config.steps)}
    device = noise.device
    timesteps = torch.linspace(0.0, 1.0, config.steps + 1, device=device, dtype=noise.dtype)
    with torch.no_grad():
        for batch_start in range(0, config.num_samples, config.batch_size):
            batch_end = min(batch_start + config.batch_size, config.num_samples)
            z = noise[batch_start:batch_end].clone()
            batch_labels = labels[batch_start:batch_end]
            for step_idx in range(config.steps):
                t_scalar = timesteps[step_idx]
                t_next_scalar = timesteps[step_idx + 1]
                v_cfg, _cfg_active = _predict_v_cfg(model, z, batch_labels, t_scalar, config)
                refs[step_idx].append(v_cfg.detach().to(dtype=torch.float16, device="cpu"))
                z = z + (t_next_scalar - t_scalar) * v_cfg
            outputs.append(z.detach())
    return torch.cat(outputs, dim=0).detach().cpu(), {idx: torch.cat(chunks, dim=0) for idx, chunks in refs.items()}


def _parse_probe_steps(spec: str, steps: int) -> set[int]:
    if spec.strip().lower() == "all":
        return set(range(steps))
    selected = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0 or step >= steps:
            raise ValueError(f"probe step {step} out of range for {steps} steps")
        selected.add(step)
    return selected


def _run_cached_with_errors(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: Stage2BConfig,
    cache_state: RuntimeCacheState,
    policy: FixedIntervalCachePolicy,
    selected_modules: list[str],
    ref_velocities: dict[int, torch.Tensor],
    step_error_path: Path,
    probe_model: Any | None = None,
) -> torch.Tensor:
    writer = JsonlWriter(step_error_path)
    outputs: list[torch.Tensor] = []
    device = noise.device
    timesteps = torch.linspace(0.0, 1.0, config.steps + 1, device=device, dtype=noise.dtype)
    probe_steps = _parse_probe_steps(config.diag_probe_steps, config.steps)
    eps = getattr(model, "t_eps", 5e-2)
    with torch.no_grad():
        for batch_start in range(0, config.num_samples, config.batch_size):
            batch_end = min(batch_start + config.batch_size, config.num_samples)
            z = noise[batch_start:batch_end].clone()
            batch_labels = labels[batch_start:batch_end]
            cache_state.clear_entries()
            for step_idx in range(config.steps):
                t_scalar = timesteps[step_idx]
                t_next_scalar = timesteps[step_idx + 1]
                t_value = float(t_scalar.detach().float().cpu().item())
                t_next_value = float(t_next_scalar.detach().float().cpu().item())
                cache_state.set_context(step_idx, t_value, "cond")
                v_cfg, cfg_active = _predict_v_cfg(model, z, batch_labels, t_scalar, config, cache_state)
                ref_v = ref_velocities[step_idx][batch_start:batch_end].to(device=v_cfg.device, dtype=v_cfg.dtype)
                trajectory_error = tensor_error_stats(v_cfg, ref_v, name="trajectory_velocity")
                frequency_error = frequency_error_stats(v_cfg, ref_v)
                probe_error = None
                if config.diag_full_probe and probe_model is not None and step_idx in probe_steps:
                    v_probe, _ = _predict_v_cfg(probe_model, z, batch_labels, t_scalar, config)
                    probe_error = tensor_error_stats(v_cfg, v_probe, name="probe_velocity")
                summary = cache_state.summary()
                active = any(
                    policy.is_active(step_idx, t_value, module_name, "cond", "euler")
                    for module_name in selected_modules
                )
                writer.write(
                    {
                        "record_type": "stage2b_step_error",
                        "step_idx": step_idx,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "t": t_value,
                        "t_next": t_next_value,
                        "cfg_enabled": cfg_active,
                        "cache_active_window": active,
                        "cache_hit_rate_so_far": summary["hit_rate"],
                        "velocity_norm_cache": l2_norm(v_cfg),
                        "velocity_norm_ref": l2_norm(ref_v),
                        "trajectory_error": trajectory_error,
                        "probe_error": probe_error,
                        "frequency_error": frequency_error,
                        "amplification": 1.0 / max(1.0 - t_value, eps),
                    }
                )
                z = z + (t_next_scalar - t_scalar) * v_cfg
            outputs.append(z.detach())
    writer.close()
    return torch.cat(outputs, dim=0).detach().cpu()


def _json_config(config: Stage2BConfig, selected_layer_ids: list[int], selected_modules: list[str]) -> dict[str, Any]:
    raw = asdict(config)
    for key in ("jit_dir", "ckpt_dir", "run_dir", "preview_dir"):
        raw[key] = str(raw[key])
    raw["selected_layer_ids"] = selected_layer_ids
    raw["selected_modules"] = selected_modules
    return raw


def run_experiment(config: Stage2BConfig) -> dict[str, Any]:
    if config.timing_repeats <= 0:
        raise ValueError("timing_repeats must be positive")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, noise = _make_inputs(_stage2_compatible_config(config), device)

    structure_model = _load_jit_model(_stage2_compatible_config(config), device)
    num_blocks = len(structure_model.net.blocks)
    selected_layer_ids = parse_layer_list(config.cache_layers, num_blocks)
    selected_modules = [f"blocks.{idx}" for idx in selected_layer_ids]
    del structure_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    meta = {
        **collect_git_status(ROOT, config.jit_dir, ROOT / "third_party/DeCo"),
        "env": collect_env_info(),
        "script": "scripts/run_jit_stage2b_cache.py",
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "preview_dir": str(config.preview_dir),
        "selected_layer_ids": selected_layer_ids,
        "selected_modules": selected_modules,
    }
    write_run_meta(config.run_dir / "meta.json", meta)
    _write_json(config.run_dir / "config.json", _json_config(config, selected_layer_ids, selected_modules))

    no_cache_model = _load_jit_model(_stage2_compatible_config(config), device)
    no_cache_timing = _time_repeats(no_cache_model, labels, noise, config)
    no_cache_ref_output, ref_velocities = _run_no_cache_with_refs(no_cache_model, labels, noise, config)
    del no_cache_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cache_state = RuntimeCacheState(model_name="JiT", enabled=True)
    policy = _policy_for_config(config, selected_modules)
    cached_model = _load_jit_model(_stage2_compatible_config(config), device)
    wrapped_modules = wrap_jit_blocks(cached_model, cache_state, policy, selected_layer_ids)
    cache_timing = _time_repeats(cached_model, labels, noise, config, cache_state=cache_state)
    cache_state.clear_entries()
    cache_state.reset_stats()
    probe_model = _load_jit_model(_stage2_compatible_config(config), device) if config.diag_full_probe else None
    cached_output = _run_cached_with_errors(
        cached_model,
        labels,
        noise,
        config,
        cache_state,
        policy,
        selected_modules,
        ref_velocities,
        config.run_dir / "step_error_stats.jsonl",
        probe_model=probe_model,
    )
    cache_stats = cache_state.summary()
    comparison = _compare_outputs(
        no_cache_ref_output,
        cached_output,
        no_cache_timing["latency_median_sec"],
        cache_timing["latency_median_sec"],
    )
    comparison["speedup_median"] = comparison["speedup"]
    comparison["speedup_mean"] = no_cache_timing["latency_mean_sec"] / cache_timing["latency_mean_sec"]

    no_cache_summary = {key: value for key, value in no_cache_timing.items() if key != "output"}
    cache_summary = {key: value for key, value in cache_timing.items() if key != "output"}
    no_cache_summary.update({"mode": "no_cache", "num_samples": config.num_samples, "steps": config.steps})
    cache_summary.update(
        {
            "mode": "cache",
            "num_samples": config.num_samples,
            "steps": config.steps,
            "selected_layer_ids": selected_layer_ids,
            "wrapped_modules": wrapped_modules,
            "cache_policy": policy.to_dict(),
            "cache_hit_rate": cache_stats["hit_rate"],
        }
    )
    _write_json(config.run_dir / "no_cache_summary.json", no_cache_summary)
    _write_json(config.run_dir / "cache_summary.json", cache_summary)
    _write_json(config.run_dir / "comparison.json", comparison)
    _write_json(config.run_dir / "cache_stats.json", cache_stats)
    if config.save_previews:
        _save_previews(no_cache_ref_output, cached_output, config.preview_dir)

    result = {
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "preview_dir": str(config.preview_dir),
        "selected_layer_ids": selected_layer_ids,
        "wrapped_modules": wrapped_modules,
        "no_cache_summary": no_cache_summary,
        "cache_summary": cache_summary,
        "comparison": comparison,
        "cache_stats": cache_stats,
    }
    print(f"JiT Stage 2B run dir: {config.run_dir}")
    print(f"Selected layers: {selected_layer_ids}")
    print(f"Active t window: [{config.active_t_min}, {config.active_t_max})")
    print(f"No-cache median latency: {no_cache_summary['latency_median_sec']:.4f}s")
    print(f"Cached median latency: {cache_summary['latency_median_sec']:.4f}s")
    print(f"Median speedup: {comparison['speedup_median']:.4f}")
    print(f"Cache hit rate: {cache_stats['hit_rate']:.4f}")
    print(f"same_seed_mse: {comparison['same_seed_mse']:.8f}")
    print(f"same_seed_rel_l2: {comparison['same_seed_rel_l2']:.8f}")
    return result


def build_config_from_args(argv: list[str] | None = None) -> Stage2BConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=_env_int("PFC_STAGE2B_NUM_SAMPLES", 8))
    parser.add_argument("--batch-size", type=int, default=_env_int("PFC_STAGE2B_BATCH_SIZE", 4))
    parser.add_argument("--steps", type=int, default=_env_int("PFC_STAGE2B_STEPS", 20))
    parser.add_argument("--seed", type=int, default=_env_int("PFC_STAGE2B_SEED", 0))
    parser.add_argument("--cache-interval", type=int, default=_env_int("PFC_STAGE2B_CACHE_INTERVAL", 2))
    parser.add_argument("--cache-layers", default=os.environ.get("PFC_STAGE2B_CACHE_LAYERS", "all"))
    parser.add_argument("--cfg", type=float, default=_env_float("PFC_STAGE2B_CFG", 3.0))
    parser.add_argument("--save-previews", dest="save_previews", action="store_true")
    parser.add_argument("--no-save-previews", dest="save_previews", action="store_false")
    parser.set_defaults(save_previews=_env_bool("PFC_STAGE2B_SAVE_PREVIEWS", True))
    args = parser.parse_args(argv)

    jit_dir = Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve()
    ckpt_dir = _detect_jit_ckpt_dir()
    run_id = os.environ.get(
        "PFC_STAGE2B_RUN_ID",
        _make_run_id(args.seed, args.steps, args.cache_interval, args.cache_layers),
    )
    run_dir = Path(os.environ.get("PFC_STAGE2B_OUT_DIR", ROOT / "logs/stage2b/jit" / run_id)).resolve()
    preview_dir = Path(
        os.environ.get("PFC_STAGE2B_PREVIEW_DIR", ROOT / "outputs/stage2b/previews/jit" / run_id)
    ).resolve()
    return Stage2BConfig(
        jit_dir=jit_dir,
        ckpt_dir=ckpt_dir,
        run_id=run_id,
        run_dir=run_dir,
        preview_dir=preview_dir,
        model=os.environ.get("PFC_STAGE2B_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE2B_IMG_SIZE", 256),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        interval_min=_env_float("PFC_STAGE2B_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE2B_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=args.cache_interval,
        cache_layers=args.cache_layers,
        cache_branches=os.environ.get("PFC_STAGE2B_CACHE_BRANCHES", "cond,uncond"),
        active_t_min=_env_optional_float("PFC_STAGE2B_ACTIVE_T_MIN", 0.1),
        active_t_max=_env_optional_float("PFC_STAGE2B_ACTIVE_T_MAX", 0.8),
        active_step_min=_env_optional_int("PFC_STAGE2B_ACTIVE_STEP_MIN", None),
        active_step_max=_env_optional_int("PFC_STAGE2B_ACTIVE_STEP_MAX", None),
        timing_repeats=_env_int("PFC_STAGE2B_TIMING_REPEATS", 3),
        warmup_runs=_env_int("PFC_STAGE2B_WARMUP_RUNS", 1),
        diag_full_probe=_env_bool("PFC_STAGE2B_DIAG_FULL_PROBE", False),
        diag_probe_steps=os.environ.get("PFC_STAGE2B_DIAG_PROBE_STEPS", "all"),
        log_step_errors=_env_bool("PFC_STAGE2B_LOG_STEP_ERRORS", True),
        save_previews=args.save_previews,
    )


def main(argv: list[str] | None = None) -> int:
    config = build_config_from_args(argv)
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
