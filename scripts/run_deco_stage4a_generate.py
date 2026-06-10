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

from pfc.eval.generation_io import (  # noqa: E402
    append_generation_manifest,
    prepare_generation_dir,
    save_image_batch_png,
    save_npz_samples,
    write_generation_meta,
)
from pfc.eval.label_schedule import make_imagenet_class_balanced_labels, save_label_schedule  # noqa: E402
from pfc.eval.method_presets import get_deco_stage4a_methods, preset_to_json_dict  # noqa: E402


def _default_run_id(seed: int, num_images: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_n{num_images}"


def _checkpoint_ok(path: Path) -> bool:
    return path.is_file()


def _json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, dict):
            output[key] = _json_ready(value)
        else:
            output[key] = value
    return output


def _make_noise_for_indices(indices: list[int], seed: int, resolution: int, device: Any) -> Any:
    import torch

    chunks = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + index)
        chunks.append(torch.randn(1, 3, resolution, resolution, generator=generator, dtype=torch.float32))
    return torch.cat(chunks, dim=0).to(device)


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.cache.cache_state import RuntimeCacheState
    from pfc.cache.deco_wrap import parse_deco_cache_spec, wrap_deco_modules
    from pfc.eval.deco_runtime import (
        DeCoRuntimeConfig,
        build_deco_sampler,
        candidate_names,
        load_deco_denoiser,
        policy_for_modules,
    )

    if args.save_npz and args.num_images > 5000:
        raise RuntimeError("--save-npz is intended for small/proxy Stage 4A runs, not large 50k runs")
    preset = get_deco_stage4a_methods()[args.method]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")
    labels = make_imagenet_class_balanced_labels(args.num_images)
    paths = resolved["paths"]
    save_label_schedule(labels, paths["base_dir"])
    config = DeCoRuntimeConfig(
        deco_dir=args.deco_dir.resolve(),
        ckpt_path=args.deco_ckpt.resolve(),
        config_path=args.deco_config.resolve(),
        run_id=args.run_id,
        run_dir=paths["base_dir"],
        num_samples=args.batch_size,
        batch_size=args.batch_size,
        steps=preset.eval_steps,
        seed=args.seed,
        cfg=args.cfg,
        cfg_interval_min=args.cfg_interval_min,
        cfg_interval_max=args.cfg_interval_max,
        cache_interval=preset.cache_interval or 1,
        active_t_min=preset.active_t_min,
        active_t_max=preset.active_t_max,
        cache_units=preset.deco_cache_units or "none",
        resolution=args.resolution,
    )
    denoiser = load_deco_denoiser(config, device)
    cache_state: RuntimeCacheState | None = None
    if preset.method_type == "cache":
        names = candidate_names(denoiser)
        selected_modules = parse_deco_cache_spec(config.cache_units, names)
        cache_state = RuntimeCacheState(model_name="DeCo", enabled=bool(selected_modules))
        wrap_deco_modules(denoiser, cache_state, policy_for_modules(config, selected_modules), selected_modules)
    sampler = build_deco_sampler(config, cache_state=cache_state, log_diagnostics=False)
    samples_for_npz = []
    labels_for_npz: list[int] = []
    generated = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    for batch_start in range(0, args.num_images, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.num_images)
        indices = list(range(batch_start, batch_end))
        if args.resume and args.save_png:
            existing = [paths["image_dir"] / f"{index:06d}.png" for index in indices]
            if all(path.exists() for path in existing):
                continue
        batch_labels_list = labels[batch_start:batch_end]
        batch_labels = torch.tensor(batch_labels_list, device=device, dtype=torch.long)
        batch_uncondition = torch.full_like(batch_labels, 1000)
        batch_noise = _make_noise_for_indices(indices, args.seed, args.resolution, device)
        batch_config = replace(config, num_samples=len(indices), batch_size=len(indices))
        sampler.num_steps = batch_config.steps
        if cache_state is not None:
            cache_state.clear_entries()
        with torch.no_grad():
            output = sampler(denoiser, batch_noise, batch_labels, batch_uncondition).detach().cpu()
        if args.save_png:
            records = save_image_batch_png(output, batch_labels_list, batch_start, paths["image_dir"])
        else:
            records = [{"index": index, "label": int(label)} for index, label in zip(indices, batch_labels_list)]
        append_generation_manifest(paths["manifest"], records)
        if args.save_npz:
            samples_for_npz.append(output)
            labels_for_npz.extend(batch_labels_list)
        generated += len(indices)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    latency = time.perf_counter() - start
    if args.save_npz:
        save_npz_samples(torch.cat(samples_for_npz, dim=0), labels_for_npz, paths["samples_npz"])
    cache_stats = cache_state.summary() if cache_state is not None else {"enabled": False, "hit_rate": 0.0}
    write_generation_meta(paths["latency"], {
        "latency_sec": latency,
        "images_per_sec": generated / latency if latency > 0 else float("inf"),
        "generated_images": generated,
        "peak_memory_allocated_bytes": peak_memory,
    })
    write_generation_meta(paths["cache_stats"], cache_stats)
    write_generation_meta(paths["generation_meta"], resolved["meta"])
    print(paths["base_dir"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    methods = get_deco_stage4a_methods()
    parser = argparse.ArgumentParser(description="Generate FID-ready DeCo Stage 4A images.")
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
    parser.add_argument("--deco-dir", type=Path, default=ROOT / "third_party/DeCo")
    parser.add_argument("--deco-ckpt", type=Path, default=ROOT / "ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt")
    parser.add_argument("--deco-config", type=Path, default=ROOT / "third_party/DeCo/configs_c2i/DeCo_XL.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg", type=float, default=3.2)
    parser.add_argument("--cfg-interval-min", type=float, default=0.1)
    parser.add_argument("--cfg-interval-max", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=256)
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = get_deco_stage4a_methods()[args.method]
    run_id = args.run_id or _default_run_id(args.seed, args.num_images)
    args.run_id = run_id
    paths = prepare_generation_dir(args.output_root, preset.model_name, args.method, run_id, create=not args.dry_run)
    meta = {
        "model": "DeCo",
        "method": preset_to_json_dict(preset),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "save_png": args.save_png,
        "save_npz": args.save_npz,
        "resume": args.resume,
        "deco_dir": str(args.deco_dir.resolve()),
        "deco_ckpt": str(args.deco_ckpt.resolve()),
        "checkpoint_exists": _checkpoint_ok(args.deco_ckpt.resolve()),
        "deco_config": str(args.deco_config.resolve()),
        "config_exists": args.deco_config.resolve().exists(),
        "device": args.device,
        "cfg": args.cfg,
        "resolution": args.resolution,
        "run_id": run_id,
    }
    return {"meta": meta, "paths": paths}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PFC_CUDA_DEVICES", "0"))
    resolved = resolve_config(args)
    if args.dry_run:
        print(json.dumps(_json_ready({"meta": resolved["meta"], "paths": resolved["paths"]}), indent=2, sort_keys=True))
        if not resolved["meta"]["checkpoint_exists"]:
            print(f"Missing DeCo checkpoint: {args.deco_ckpt}")
            return 2
        if not resolved["meta"]["config_exists"]:
            print(f"Missing DeCo config: {args.deco_config}")
            return 2
        return 0
    if not resolved["meta"]["checkpoint_exists"]:
        raise FileNotFoundError(f"Missing DeCo checkpoint: {args.deco_ckpt}")
    if not resolved["meta"]["config_exists"]:
        raise FileNotFoundError(f"Missing DeCo config: {args.deco_config}")
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
