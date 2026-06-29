#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before importing modules that import torch; otherwise PyTorch may
# initialize CUDA before the visible-device mask is resolved.
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PFC_CUDA_DEVICES", "0")

from pfc.adapters import PixelGenBoundaryAdapter  # noqa: E402
from pfc.eval.generation_io import (  # noqa: E402
    append_generation_manifest,
    prepare_generation_dir,
    save_image_batch_png,
    save_npz_samples,
    write_generation_meta,
)
from pfc.eval.label_schedule import make_imagenet_class_balanced_labels, save_label_schedule  # noqa: E402
from pfc.eval.method_presets import get_pixelgen_stage4a_methods, preset_to_json_dict  # noqa: E402
from pfc.eval.pixelgen_runtime import PIXELGEN_SOLVER_STAGES  # noqa: E402


def _default_run_id(seed: int, num_images: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_n{num_images}"


def _checkpoint_ok(path: Path) -> bool:
    return path.is_file()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _print_dry_run(config: dict[str, Any]) -> None:
    print(json.dumps(_json_ready(config), indent=2, sort_keys=True))


def _dynamic_threshold(args: argparse.Namespace, preset: Any) -> float:
    value = args.dynamic_cache_threshold
    if value is None:
        value = preset.dynamic_cache_threshold
    if value is None:
        value = 0.06
    return float(value)


def _resolved_method_meta(args: argparse.Namespace, preset: Any) -> dict[str, Any]:
    method = preset_to_json_dict(preset)
    if preset.method_type == "cache":
        method.update(
            {
                "cache_units": "pixelgen_jit_blocks",
                "selected_modules": (preset.cache_preset or {}).get("cache_layers", "all"),
                "solver_stages": list(preset.solver_stages or PIXELGEN_SOLVER_STAGES),
            }
        )
    elif preset.method_type == "dynamic_cache":
        threshold = _dynamic_threshold(args, preset)
        method.update(
            {
                "dynamic_cache_threshold": threshold,
                "resolved_dynamic_cache_threshold": threshold,
                "sea_beta": args.sea_beta,
                "sea_proxy_downsample": args.sea_proxy_downsample,
                "cache_units": "pixelgen_jit_blocks",
                "selected_modules": (preset.cache_preset or {}).get("cache_layers", "all"),
                "solver_stages": list(preset.solver_stages or PIXELGEN_SOLVER_STAGES),
            }
        )
    return method


def _pixbfc_static_meta(method_type: str) -> dict[str, Any]:
    adapter = PixelGenBoundaryAdapter()
    boundary_set = None
    if method_type in {"cache", "dynamic_cache"}:
        boundary_set = {
            "name": "pixelgen_jit_style_blocks",
            "description": "Resolved to concrete PixelGen blocks.* names after model construction.",
            "module_names": "all_blocks",
            "resolved_after_model_load": True,
        }
    return {
        "pixbfc_adapter": adapter.describe(),
        "prediction_type": adapter.prediction_type.value,
        "output_to_velocity": "xpred_to_velocity_eps_0.05",
        "boundary_set": boundary_set,
    }


def _dynamic_writer(path: Path | None):
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")

    def write(payload: dict[str, Any]) -> None:
        handle.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")
        handle.flush()

    return handle, write


def _make_noise_for_indices(indices: list[int], seed: int, img_size: int, noise_scale: float, device: Any) -> Any:
    import torch

    chunks = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + index)
        chunks.append(
            noise_scale * torch.randn(1, 3, img_size, img_size, generator=generator, dtype=torch.float32)
        )
    return torch.cat(chunks, dim=0).to(device)


def _autocast_context(device: Any, amp_dtype: str):
    import contextlib
    import torch

    if getattr(device, "type", str(device)) != "cuda" or amp_dtype == "fp32":
        return contextlib.nullcontext()
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return torch.autocast(device_type="cuda", dtype=dtype_by_name[amp_dtype])


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.cache.cache_state import RuntimeCacheState
    from pfc.cache.dynamic_policy_adapter import DynamicPolicyAdapter
    from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy
    from pfc.eval.pixelgen_runtime import (
        PixelGenRuntimeConfig,
        load_pixelgen_model,
        policy_for_pixelgen_modules,
        sample_pixelgen_heun_jit,
    )

    if args.save_npz and args.num_images > 5000:
        raise RuntimeError("--save-npz is intended for small/proxy Stage 4A runs, not large 50k runs")
    preset = get_pixelgen_stage4a_methods()[args.method]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")

    labels = make_imagenet_class_balanced_labels(args.num_images)
    paths = resolved["paths"]
    save_label_schedule(labels, paths["base_dir"])
    config = PixelGenRuntimeConfig(
        pixelgen_dir=args.pixelgen_dir.resolve(),
        ckpt_path=args.pixelgen_ckpt.resolve(),
        run_id=args.run_id,
        run_dir=paths["base_dir"],
        img_size=args.img_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        cfg=args.cfg,
        timeshift=args.timeshift,
        guidance_interval_min=args.guidance_interval_min,
        guidance_interval_max=args.guidance_interval_max,
        t_eps=args.t_eps,
        steps=preset.eval_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_interval=preset.cache_interval or 1,
        active_t_min=preset.active_t_min,
        active_t_max=preset.active_t_max,
        noise_scale=args.noise_scale,
        enable_compile=args.enable_compile,
    )
    denoiser = load_pixelgen_model(config, device)
    resolved["meta"]["checkpoint_weight_source"] = getattr(denoiser, "_pfc_checkpoint_source", None)

    boundary_adapter = PixelGenBoundaryAdapter()
    cache_state: RuntimeCacheState | None = None
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None
    if preset.method_type == "cache":
        boundary_set = boundary_adapter.default_boundary_set(denoiser, args.method)
        selected_modules = list(boundary_set.module_names())
        cache_state = RuntimeCacheState(model_name="PixelGen", enabled=bool(selected_modules))
        boundary_adapter.wrap_boundary_set(
            denoiser,
            boundary_set,
            cache_state,
            policy_for_pixelgen_modules(config, selected_modules),
        )
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "pixelgen_jit_blocks"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        resolved["meta"]["solver_stages"] = list(PIXELGEN_SOLVER_STAGES)
    elif preset.method_type == "dynamic_cache":
        boundary_set = boundary_adapter.default_boundary_set(denoiser, args.method)
        selected_modules = list(boundary_set.module_names())
        threshold = _dynamic_threshold(args, preset)
        policy_kwargs = {
            "threshold": threshold,
            "force_first_n_steps": args.dynamic_force_first_n_steps,
            "min_t": args.sea_min_t,
            "max_t": args.sea_max_t,
            "per_branch": args.dynamic_per_branch,
        }
        if preset.dynamic_cache_type == "sea":
            dynamic_policy = SeaCacheSpectralDistancePolicy(
                beta=args.sea_beta,
                normalize_filter=True,
                time_direction="noise_to_image",
                **policy_kwargs,
            )
        else:
            dynamic_policy = RawAccumulatedDistancePolicy(**policy_kwargs)
        cache_state = RuntimeCacheState(model_name="PixelGen", enabled=bool(selected_modules))
        adapter = DynamicPolicyAdapter(
            dynamic_policy=dynamic_policy,
            cache_modules=set(selected_modules),
            solver_stages=set(PIXELGEN_SOLVER_STAGES),
        )
        boundary_adapter.wrap_boundary_set(denoiser, boundary_set, cache_state, adapter)
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "pixelgen_jit_blocks"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        resolved["meta"]["solver_stages"] = list(PIXELGEN_SOLVER_STAGES)

    samples_for_npz = []
    labels_for_npz: list[int] = []
    generated = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    debug_handle, dynamic_decision_writer = _dynamic_writer(args.dynamic_cache_debug_jsonl)
    try:
        for batch_start in range(0, args.num_images, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_images)
            indices = list(range(batch_start, batch_end))
            if args.resume and args.save_png:
                existing = [paths["image_dir"] / f"{index:06d}.png" for index in indices]
                if all(path.exists() for path in existing):
                    continue
            batch_labels_list = labels[batch_start:batch_end]
            batch_labels = torch.tensor(batch_labels_list, device=device, dtype=torch.long)
            batch_noise = _make_noise_for_indices(indices, args.seed, args.img_size, args.noise_scale, device)
            batch_config = replace(config, batch_size=len(indices))
            if cache_state is not None:
                cache_state.clear_entries()
            if dynamic_policy is not None:
                dynamic_policy.clear_batch()
            with torch.no_grad(), _autocast_context(device, args.amp_dtype):
                output, _records = sample_pixelgen_heun_jit(
                    denoiser,
                    batch_labels,
                    batch_noise,
                    batch_config,
                    cache_state=cache_state,
                    dynamic_policy=dynamic_policy,
                    dynamic_proxy_downsample=args.sea_proxy_downsample,
                    dynamic_decision_writer=dynamic_decision_writer,
                )
            output_cpu = output.detach().cpu()
            if args.save_png:
                records = save_image_batch_png(output_cpu, batch_labels_list, batch_start, paths["image_dir"])
            else:
                records = [{"index": index, "label": int(label)} for index, label in zip(indices, batch_labels_list)]
            append_generation_manifest(paths["manifest"], records)
            if args.save_npz:
                samples_for_npz.append(output_cpu)
                labels_for_npz.extend(batch_labels_list)
            generated += len(indices)
    finally:
        if debug_handle is not None:
            debug_handle.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    latency = time.perf_counter() - start
    if args.save_npz:
        save_npz_samples(torch.cat(samples_for_npz, dim=0), labels_for_npz, paths["samples_npz"])
    cache_stats = cache_state.summary() if cache_state is not None else {"enabled": False, "hit_rate": 0.0}
    if dynamic_policy is not None:
        cache_stats["dynamic_cache"] = dynamic_policy.summary()
        cache_stats["dynamic_cache_threshold"] = dynamic_policy.threshold
        cache_stats["resolved_dynamic_cache_threshold"] = dynamic_policy.threshold
        resolved["meta"]["dynamic_cache_summary"] = dynamic_policy.summary()
    write_generation_meta(
        paths["latency"],
        {
            "latency_sec": latency,
            "images_per_sec": generated / latency if latency > 0 else float("inf"),
            "generated_images": generated,
            "peak_memory_allocated_bytes": peak_memory,
        },
    )
    write_generation_meta(paths["cache_stats"], cache_stats)
    write_generation_meta(paths["generation_meta"], resolved["meta"])
    print(paths["base_dir"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    methods = get_pixelgen_stage4a_methods()
    parser = argparse.ArgumentParser(description="Generate FID-ready PixelGen Stage 4A images.")
    parser.add_argument("--method", required=True, choices=sorted(methods))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/stage4a/full_generation")
    parser.add_argument("--save-png", dest="save_png", action="store_true", default=True)
    parser.add_argument("--no-save-png", dest="save_png", action="store_false")
    parser.add_argument("--save-npz", dest="save_npz", action="store_true", default=False)
    parser.add_argument("--no-save-npz", dest="save_npz", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--pixelgen-dir", type=Path, default=ROOT / "third_party/PixelGen")
    parser.add_argument("--pixelgen-ckpt", type=Path, default=ROOT / "ckpts/PixelGen/PixelGen_XL_160ep.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg", type=float, default=2.25)
    parser.add_argument("--timeshift", type=float, default=2.0)
    parser.add_argument("--guidance-interval-min", type=float, default=0.1)
    parser.add_argument("--guidance-interval-max", type=float, default=0.9)
    parser.add_argument("--t-eps", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--dynamic-cache-threshold", type=float)
    parser.add_argument("--sea-beta", type=float, default=2.0)
    parser.add_argument("--sea-proxy-downsample", type=int, default=64)
    parser.add_argument("--sea-min-t", type=float)
    parser.add_argument("--sea-max-t", type=float)
    parser.add_argument("--dynamic-force-first-n-steps", type=int, default=0)
    parser.add_argument("--dynamic-per-branch", action="store_true")
    parser.add_argument("--dynamic-cache-debug-jsonl", type=Path)
    parser.add_argument("--enable-compile", dest="enable_compile", action="store_true", default=False)
    parser.add_argument("--disable-compile", dest="enable_compile", action="store_false")
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = get_pixelgen_stage4a_methods()[args.method]
    run_id = args.run_id or _default_run_id(args.seed, args.num_images)
    args.run_id = run_id
    paths = prepare_generation_dir(args.output_root, preset.model_name, args.method, run_id, create=not args.dry_run)
    solver_stages = (
        list(preset.solver_stages or PIXELGEN_SOLVER_STAGES)
        if preset.method_type in {"cache", "dynamic_cache"}
        else None
    )
    dynamic_threshold = _dynamic_threshold(args, preset) if preset.method_type == "dynamic_cache" else None
    meta = {
        "model_name": preset.model_name,
        "model": "PixelGen",
        "method_name": preset.method_name,
        "method": _resolved_method_meta(args, preset),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eval_steps": preset.eval_steps,
        "reference_steps": preset.reference_steps,
        "cache_units": "pixelgen_jit_blocks" if preset.method_type in {"cache", "dynamic_cache"} else None,
        "selected_modules": (preset.cache_preset or {}).get("cache_layers") if preset.cache_preset else None,
        "boundary_set": None,
        "solver_stages": solver_stages,
        "dynamic_cache_type": preset.dynamic_cache_type,
        "dynamic_cache_threshold": dynamic_threshold,
        "resolved_dynamic_cache_threshold": dynamic_threshold,
        "sea_beta": args.sea_beta if preset.method_type == "dynamic_cache" else None,
        "sea_proxy_downsample": args.sea_proxy_downsample if preset.method_type == "dynamic_cache" else None,
        "sea_min_t": args.sea_min_t if preset.method_type == "dynamic_cache" else None,
        "sea_max_t": args.sea_max_t if preset.method_type == "dynamic_cache" else None,
        "dynamic_force_first_n_steps": (
            args.dynamic_force_first_n_steps if preset.method_type == "dynamic_cache" else None
        ),
        "dynamic_per_branch": args.dynamic_per_branch if preset.method_type == "dynamic_cache" else None,
        "dynamic_cache_debug_jsonl": (
            str(args.dynamic_cache_debug_jsonl.resolve()) if args.dynamic_cache_debug_jsonl else None
        ),
        "dynamic_cache": {
            "type": preset.dynamic_cache_type,
            "threshold": dynamic_threshold,
            "resolved_dynamic_cache_threshold": dynamic_threshold,
            "sea_beta": args.sea_beta,
            "sea_proxy_downsample": args.sea_proxy_downsample,
            "sea_min_t": args.sea_min_t,
            "sea_max_t": args.sea_max_t,
            "dynamic_force_first_n_steps": args.dynamic_force_first_n_steps,
            "dynamic_per_branch": args.dynamic_per_branch,
            "dynamic_cache_debug_jsonl": str(args.dynamic_cache_debug_jsonl.resolve())
            if args.dynamic_cache_debug_jsonl
            else None,
        }
        if preset.method_type == "dynamic_cache"
        else None,
        "save_png": args.save_png,
        "save_npz": args.save_npz,
        "resume": args.resume,
        "pixelgen_dir": str(args.pixelgen_dir.resolve()),
        "pixelgen_dir_exists": args.pixelgen_dir.resolve().is_dir(),
        "pixelgen_ckpt": str(args.pixelgen_ckpt.resolve()),
        "checkpoint_path": str(args.pixelgen_ckpt.resolve()),
        "checkpoint_exists": _checkpoint_ok(args.pixelgen_ckpt.resolve()),
        "device": args.device,
        "cfg": args.cfg,
        "timeshift": args.timeshift,
        "guidance_interval_min": args.guidance_interval_min,
        "guidance_interval_max": args.guidance_interval_max,
        "t_eps": args.t_eps,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "num_classes": args.num_classes,
        "noise_scale": args.noise_scale,
        "amp_dtype": args.amp_dtype,
        "enable_compile": args.enable_compile,
        "cache_interval": preset.cache_interval,
        "active_t_min": preset.active_t_min,
        "active_t_max": preset.active_t_max,
        "cache_stats": str(paths["cache_stats"]),
        "run_id": run_id,
        **_pixbfc_static_meta(preset.method_type),
    }
    return {"meta": meta, "paths": paths}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.img_size <= 0:
        parser.error("--img-size must be positive")
    if args.t_eps <= 0:
        parser.error("--t-eps must be positive")
    if args.dynamic_cache_threshold is not None and args.dynamic_cache_threshold <= 0:
        parser.error("--dynamic-cache-threshold must be positive")
    if args.sea_beta <= 0:
        parser.error("--sea-beta must be positive")
    if args.sea_proxy_downsample < 0:
        parser.error("--sea-proxy-downsample must be non-negative")
    if args.dynamic_force_first_n_steps < 0:
        parser.error("--dynamic-force-first-n-steps must be non-negative")
    resolved = resolve_config(args)
    if args.dry_run:
        _print_dry_run({"meta": resolved["meta"], "paths": resolved["paths"]})
        return 0
    if not resolved["meta"]["checkpoint_exists"]:
        raise FileNotFoundError(f"Missing PixelGen checkpoint: {args.pixelgen_ckpt}. Pass --pixelgen-ckpt.")
    if not resolved["meta"]["pixelgen_dir_exists"]:
        raise FileNotFoundError(f"Missing PixelGen directory: {args.pixelgen_dir}. Initialize third_party/PixelGen first.")
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
