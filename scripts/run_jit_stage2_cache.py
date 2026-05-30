#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.cache.cache_state import RuntimeCacheState  # noqa: E402
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy  # noqa: E402
from pfc.cache.wrap import parse_layer_list, wrap_jit_blocks  # noqa: E402
from pfc.profiling.frequency import frequency_delta_bands  # noqa: E402
from pfc.profiling.jsonl import JsonlWriter  # noqa: E402
from pfc.profiling.run_meta import collect_env_info, collect_git_status, write_run_meta  # noqa: E402
from pfc.profiling.tensor_stats import l2_norm  # noqa: E402
from pfc.utils.seeding import set_seed  # noqa: E402


@dataclass
class Stage2Config:
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
    cache_layers: str = "middle"
    cache_branches: str = "cond,uncond"
    warmup_runs: int = 1
    save_previews: bool = True


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _detect_jit_ckpt_dir() -> Path:
    env_path = os.environ.get("PFC_JIT_CKPT_DIR")
    if env_path and (Path(env_path) / "checkpoint-last.pth").is_file():
        return Path(env_path).resolve()
    expected = ROOT / "ckpts/JiT/JiT-B-16-256"
    if (expected / "checkpoint-last.pth").is_file():
        return expected.resolve()
    matches = sorted((ROOT / "ckpts").glob("**/checkpoint-last.pth"))
    if matches:
        return matches[0].parent.resolve()
    raise FileNotFoundError("JiT checkpoint not found. Expected ckpts/JiT/JiT-B-16-256/checkpoint-last.pth")


def _make_run_id(seed: int, steps: int, cache_interval: int, cache_layers: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_layers = "".join(ch if ch.isalnum() else "-" for ch in cache_layers)[:48].strip("-")
    return f"{stamp}_seed{seed}_steps{steps}_i{cache_interval}_{safe_layers or 'layers'}"


def _build_model_args(config: Stage2Config) -> Namespace:
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


def _load_jit_model(config: Stage2Config, device: torch.device):
    jit_dir = config.jit_dir.resolve()
    if str(jit_dir) not in sys.path:
        sys.path.insert(0, str(jit_dir))
    from denoiser import Denoiser  # type: ignore

    args = _build_model_args(config)
    model = Denoiser(args)
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


def _make_inputs(config: Stage2Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(config.num_samples, dtype=torch.long) % 1000
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    noise = config.noise_scale * torch.randn(
        config.num_samples,
        3,
        config.img_size,
        config.img_size,
        generator=generator,
        dtype=torch.float32,
    )
    return labels.to(device), noise.to(device)


def _cfg_enabled(t_value: float, low: float, high: float) -> bool:
    return (t_value < high) and ((low == 0.0) or (t_value > low))


def _sample_jit(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: Stage2Config,
    mode: str,
    cache_state: RuntimeCacheState | None = None,
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

        for step_idx in range(config.steps):
            t_scalar = timesteps[step_idx]
            t_next_scalar = timesteps[step_idx + 1]
            dt = t_next_scalar - t_scalar
            t_value = float(t_scalar.detach().float().cpu().item())
            t_next_value = float(t_next_scalar.detach().float().cpu().item())
            dt_value = float(dt.detach().float().cpu().item())
            t = t_scalar.expand(z.shape[0], 1, 1, 1)
            cfg_active = _cfg_enabled(t_value, config.interval_min, config.interval_max)
            cfg_scale_interval = config.cfg if cfg_active else 1.0

            if cache_state is not None:
                cache_state.set_context(step_idx, t_value, "cond", solver_stage="euler")
            x_cond = model.net(z, t.flatten(), batch_labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(model.t_eps)

            if cache_state is not None:
                cache_state.set_context(step_idx, t_value, "uncond", solver_stage="euler")
            null_labels = torch.full_like(batch_labels, model.num_classes)
            x_uncond = model.net(z, t.flatten(), null_labels)
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(model.t_eps)

            v_cfg = v_uncond + cfg_scale_interval * (v_cond - v_uncond)
            records.append(
                {
                    "record_type": "stage2_step",
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


def _run_timed(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: Stage2Config,
    mode: str,
    cache_state: RuntimeCacheState | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    device = noise.device
    for _warmup_idx in range(config.warmup_runs):
        with torch.no_grad():
            _sample_jit(model, labels, noise, config, mode=f"{mode}_warmup", cache_state=cache_state)
        if cache_state is not None:
            cache_state.clear_entries()
            cache_state.reset_stats()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        output, records = _sample_jit(model, labels, noise, config, mode=mode, cache_state=cache_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    latency = time.perf_counter() - start
    summary = {
        "mode": mode,
        "latency_sec": latency,
        "samples_per_sec": config.num_samples / latency if latency > 0 else float("inf"),
        "peak_memory_allocated_bytes": peak_memory,
        "num_samples": config.num_samples,
        "batch_size": config.batch_size,
        "steps": config.steps,
        "warmup_runs": config.warmup_runs,
    }
    return output.detach().cpu(), summary, records


def _compare_outputs(no_cache: torch.Tensor, cached: torch.Tensor, no_cache_latency: float, cache_latency: float) -> dict[str, Any]:
    delta = cached.float() - no_cache.float()
    mse = float(torch.mean(delta.square()).item())
    mae = float(torch.mean(torch.abs(delta)).item())
    rmse = math.sqrt(mse)
    rel_l2 = float((torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(no_cache.float()).clamp_min(1e-8)).item())
    psnr = float("inf") if rmse == 0 else 20.0 * math.log10(2.0 / rmse)
    frequency_delta = frequency_delta_bands(cached, no_cache)
    return {
        "same_seed_mse": mse,
        "same_seed_mae": mae,
        "same_seed_rmse": rmse,
        "same_seed_rel_l2": rel_l2,
        "same_seed_psnr": psnr,
        "frequency_delta": frequency_delta,
        "no_cache_latency_sec": no_cache_latency,
        "cache_latency_sec": cache_latency,
        "speedup": no_cache_latency / cache_latency if cache_latency > 0 else float("inf"),
    }


def _to_uint8_image(x: torch.Tensor) -> Image.Image:
    image = ((x.detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    arr = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def _to_uint8_diff(x: torch.Tensor) -> Image.Image:
    image = x.detach().float().cpu().abs()
    denom = float(image.max().item())
    if denom > 0:
        image = image / denom
    arr = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def _save_previews(no_cache: torch.Tensor, cached: torch.Tensor, preview_dir: Path, max_images: int = 4) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(max_images, no_cache.shape[0], cached.shape[0])):
        _to_uint8_image(no_cache[idx]).save(preview_dir / f"no_cache_{idx:03d}.png")
        _to_uint8_image(cached[idx]).save(preview_dir / f"cache_{idx:03d}.png")
        _to_uint8_diff(cached[idx] - no_cache[idx]).save(preview_dir / f"abs_diff_{idx:03d}.png")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def run_experiment(config: Stage2Config) -> dict[str, Any]:
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.steps <= 0:
        raise ValueError("steps must be positive")

    config.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, noise = _make_inputs(config, device)

    no_cache_model = _load_jit_model(config, device)
    num_blocks = len(no_cache_model.net.blocks)
    selected_layer_ids = parse_layer_list(config.cache_layers, num_blocks)
    selected_modules = [f"blocks.{idx}" for idx in selected_layer_ids]

    meta = {
        **collect_git_status(ROOT, config.jit_dir, ROOT / "third_party/DeCo"),
        "env": collect_env_info(),
        "script": "scripts/run_jit_stage2_cache.py",
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "preview_dir": str(config.preview_dir),
        "selected_layer_ids": selected_layer_ids,
        "selected_modules": selected_modules,
    }
    write_run_meta(config.run_dir / "meta.json", meta)
    _write_json(
        config.run_dir / "config.json",
        {
            **asdict(config),
            "jit_dir": str(config.jit_dir),
            "ckpt_dir": str(config.ckpt_dir),
            "run_dir": str(config.run_dir),
            "preview_dir": str(config.preview_dir),
            "selected_layer_ids": selected_layer_ids,
            "selected_modules": selected_modules,
        },
    )

    no_cache_output, no_cache_summary, no_cache_records = _run_timed(
        no_cache_model,
        labels,
        noise,
        config,
        mode="no_cache",
    )
    del no_cache_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cache_state = RuntimeCacheState(model_name="JiT", enabled=True)
    branches = {branch.strip() for branch in config.cache_branches.split(",") if branch.strip()}
    policy = FixedIntervalCachePolicy.from_branches(
        branches,
        enabled=True,
        interval=config.cache_interval,
        cache_modules=set(selected_modules),
    )
    cached_model = _load_jit_model(config, device)
    wrapped_modules = wrap_jit_blocks(cached_model, cache_state, policy, selected_layer_ids)
    cached_output, cache_summary, cache_records = _run_timed(
        cached_model,
        labels,
        noise,
        config,
        mode="cache",
        cache_state=cache_state,
    )

    cache_stats = cache_state.summary()
    cache_summary.update(
        {
            "selected_layer_ids": selected_layer_ids,
            "wrapped_modules": wrapped_modules,
            "cache_policy": policy.to_dict(),
            "cache_hit_rate": cache_stats["hit_rate"],
        }
    )
    comparison = _compare_outputs(
        no_cache_output,
        cached_output,
        no_cache_summary["latency_sec"],
        cache_summary["latency_sec"],
    )

    _write_json(config.run_dir / "no_cache_summary.json", no_cache_summary)
    _write_json(config.run_dir / "cache_summary.json", cache_summary)
    _write_json(config.run_dir / "comparison.json", comparison)
    _write_json(config.run_dir / "cache_stats.json", cache_stats)
    step_writer = JsonlWriter(config.run_dir / "step_stats.jsonl")
    for record in no_cache_records + cache_records:
        step_writer.write(record)
    step_writer.close()
    if config.save_previews:
        _save_previews(no_cache_output, cached_output, config.preview_dir)

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
    print(f"JiT Stage 2 run dir: {config.run_dir}")
    print(f"Selected layers: {selected_layer_ids}")
    print(f"Cache interval: {config.cache_interval}")
    print(f"No-cache latency: {no_cache_summary['latency_sec']:.4f}s")
    print(f"Cached latency: {cache_summary['latency_sec']:.4f}s")
    print(f"Speedup: {comparison['speedup']:.4f}")
    print(f"Cache hit rate: {cache_stats['hit_rate']:.4f}")
    print(f"same_seed_mse: {comparison['same_seed_mse']:.8f}")
    print(f"same_seed_rel_l2: {comparison['same_seed_rel_l2']:.8f}")
    print(f"Preview dir: {config.preview_dir if config.save_previews else 'disabled'}")
    return result


def build_config_from_args(argv: list[str] | None = None) -> Stage2Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=_env_int("PFC_STAGE2_NUM_SAMPLES", 8))
    parser.add_argument("--batch-size", type=int, default=_env_int("PFC_STAGE2_BATCH_SIZE", 4))
    parser.add_argument("--steps", type=int, default=_env_int("PFC_STAGE2_STEPS", 20))
    parser.add_argument("--seed", type=int, default=_env_int("PFC_STAGE2_SEED", 0))
    parser.add_argument("--cache-interval", type=int, default=_env_int("PFC_STAGE2_CACHE_INTERVAL", 2))
    parser.add_argument("--cache-layers", default=os.environ.get("PFC_STAGE2_CACHE_LAYERS", "middle"))
    parser.add_argument("--cfg", type=float, default=_env_float("PFC_STAGE2_CFG", 3.0))
    parser.add_argument("--save-previews", dest="save_previews", action="store_true")
    parser.add_argument("--no-save-previews", dest="save_previews", action="store_false")
    parser.set_defaults(save_previews=_env_bool("PFC_STAGE2_SAVE_PREVIEWS", True))
    args = parser.parse_args(argv)

    jit_dir = Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve()
    ckpt_dir = _detect_jit_ckpt_dir()
    run_id = os.environ.get(
        "PFC_STAGE2_RUN_ID",
        _make_run_id(args.seed, args.steps, args.cache_interval, args.cache_layers),
    )
    run_dir = Path(os.environ.get("PFC_STAGE2_OUT_DIR", ROOT / "logs/stage2/jit" / run_id)).resolve()
    preview_dir = Path(
        os.environ.get("PFC_STAGE2_PREVIEW_DIR", ROOT / "outputs/stage2/previews/jit" / run_id)
    ).resolve()
    return Stage2Config(
        jit_dir=jit_dir,
        ckpt_dir=ckpt_dir,
        run_id=run_id,
        run_dir=run_dir,
        preview_dir=preview_dir,
        model=os.environ.get("PFC_STAGE2_MODEL", "JiT-B/16"),
        img_size=_env_int("PFC_STAGE2_IMG_SIZE", 256),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        interval_min=_env_float("PFC_STAGE2_INTERVAL_MIN", 0.1),
        interval_max=_env_float("PFC_STAGE2_INTERVAL_MAX", 1.0),
        noise_scale=_env_float("PFC_NOISE_SCALE", 1.0),
        cache_interval=args.cache_interval,
        cache_layers=args.cache_layers,
        cache_branches=os.environ.get("PFC_STAGE2_CACHE_BRANCHES", "cond,uncond"),
        warmup_runs=_env_int("PFC_STAGE2_WARMUP_RUNS", 1),
        save_previews=args.save_previews,
    )


def main(argv: list[str] | None = None) -> int:
    config = build_config_from_args(argv)
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
