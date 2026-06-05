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
from pfc.eval.method_presets import get_jit_stage4a_methods, preset_to_json_dict  # noqa: E402


def _default_run_id(seed: int, num_images: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_n{num_images}"


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


def _checkpoint_ok(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "checkpoint-last.pth").is_file()


def _print_dry_run(config: dict[str, Any]) -> None:
    print(json.dumps(_json_ready(config), indent=2, sort_keys=True))


def _make_noise_for_indices(indices: list[int], seed: int, img_size: int, noise_scale: float, device: Any) -> Any:
    import torch

    chunks = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + index)
        chunks.append(
            noise_scale
            * torch.randn(1, 3, img_size, img_size, generator=generator, dtype=torch.float32)
        )
    return torch.cat(chunks, dim=0).to(device)


def load_jit_runtime_helpers() -> tuple[Any, Any, Any]:
    from scripts.run_jit_stage2b_cache import Stage2BConfig, _load_jit_model
    from scripts.run_jit_stage2_cache import _sample_jit

    return Stage2BConfig, _load_jit_model, _sample_jit


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.cache.cache_state import RuntimeCacheState
    from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
    from pfc.cache.wrap import parse_layer_list, wrap_jit_blocks

    Stage2BConfig, _load_jit_model, sample_jit = load_jit_runtime_helpers()

    if args.save_npz and args.num_images > 5000:
        raise RuntimeError("--save-npz is intended for small/proxy Stage 4A runs, not large 50k runs")
    preset = get_jit_stage4a_methods()[args.method]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")
    labels = make_imagenet_class_balanced_labels(args.num_images)
    paths = resolved["paths"]
    save_label_schedule(labels, paths["base_dir"])
    config = Stage2BConfig(
        jit_dir=args.jit_dir.resolve(),
        ckpt_dir=args.jit_ckpt_dir.resolve(),
        run_id=args.run_id,
        run_dir=paths["base_dir"],
        preview_dir=paths["base_dir"] / "previews",
        model=args.jit_model,
        img_size=args.img_size,
        num_samples=args.batch_size,
        batch_size=args.batch_size,
        steps=preset.eval_steps,
        seed=args.seed,
        cfg=args.cfg,
        interval_min=0.1,
        interval_max=1.0,
        noise_scale=args.noise_scale,
        cache_interval=preset.cache_interval or 1,
        cache_layers=(preset.cache_preset or {}).get("cache_layers", "none"),
        cache_branches="cond,uncond",
        active_t_min=preset.active_t_min,
        active_t_max=preset.active_t_max,
        active_window_warmup_refreshes=preset.active_window_warmup_refreshes,
        warmup_runs=0,
        save_previews=False,
    )
    model = _load_jit_model(config, device)
    cache_state: RuntimeCacheState | None = None
    if preset.method_type == "cache":
        num_blocks = len(model.net.blocks)
        selected_layer_ids = parse_layer_list(config.cache_layers, num_blocks)
        selected_modules = [f"blocks.{idx}" for idx in selected_layer_ids]
        cache_state = RuntimeCacheState(model_name="JiT", enabled=True)
        policy = FixedIntervalCachePolicy.from_branches(
            {"cond", "uncond"},
            enabled=True,
            interval=config.cache_interval,
            cache_modules=set(selected_modules),
            active_t_min=config.active_t_min,
            active_t_max=config.active_t_max,
            active_window_warmup_refreshes=config.active_window_warmup_refreshes,
        )
        wrap_jit_blocks(model, cache_state, policy, selected_layer_ids)

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
        noise = _make_noise_for_indices(indices, args.seed, args.img_size, args.noise_scale, device)
        batch_config = replace(config, num_samples=len(indices), batch_size=len(indices))
        if cache_state is not None:
            cache_state.clear_entries()
        with torch.no_grad():
            output, _records = sample_jit(model, batch_labels, noise, batch_config, mode=args.method, cache_state=cache_state)
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
    methods = get_jit_stage4a_methods()
    parser = argparse.ArgumentParser(description="Generate FID-ready JiT Stage 4A images.")
    parser.add_argument("--method", required=True, choices=sorted(methods))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/stage4a/full_generation")
    parser.add_argument("--save-png", dest="save_png", action="store_true", default=True)
    parser.add_argument("--no-save-png", dest="save_png", action="store_false")
    parser.add_argument("--save-npz", dest="save_npz", action="store_true", default=False)
    parser.add_argument("--no-save-npz", dest="save_npz", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--jit-dir", type=Path, default=ROOT / "third_party/JiT")
    parser.add_argument("--jit-ckpt-dir", type=Path, default=ROOT / "ckpts/JiT/JiT-B-16-256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--jit-model", default="JiT-B/16")
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = get_jit_stage4a_methods()[args.method]
    run_id = args.run_id or _default_run_id(args.seed, args.num_images)
    args.run_id = run_id
    paths = prepare_generation_dir(args.output_root, preset.model_name, args.method, run_id, create=not args.dry_run)
    meta = {
        "model": "JiT",
        "method": preset_to_json_dict(preset),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "save_png": args.save_png,
        "save_npz": args.save_npz,
        "resume": args.resume,
        "jit_dir": str(args.jit_dir.resolve()),
        "jit_ckpt_dir": str(args.jit_ckpt_dir.resolve()),
        "checkpoint_exists": _checkpoint_ok(args.jit_ckpt_dir.resolve()),
        "device": args.device,
        "cfg": args.cfg,
        "img_size": args.img_size,
        "noise_scale": args.noise_scale,
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
        _print_dry_run({"meta": resolved["meta"], "paths": resolved["paths"]})
        if not resolved["meta"]["checkpoint_exists"]:
            print(f"Missing JiT checkpoint: {args.jit_ckpt_dir / 'checkpoint-last.pth'}")
            return 2
        return 0
    if not resolved["meta"]["checkpoint_exists"]:
        raise FileNotFoundError(f"Missing JiT checkpoint: {args.jit_ckpt_dir / 'checkpoint-last.pth'}")
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
