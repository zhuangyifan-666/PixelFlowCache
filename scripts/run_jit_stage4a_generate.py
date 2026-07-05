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
    count_images,
    prepare_generation_dir,
    save_image_batch_png,
    save_npz_samples,
    write_generation_meta,
)
from pfc.adapters import JiTBoundaryAdapter  # noqa: E402
from pfc.cache.safe_map_policy import compute_safe_map_density  # noqa: E402
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
        elif isinstance(value, (list, tuple)):
            output[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            output[key] = value
    return output


def _checkpoint_ok(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "checkpoint-last.pth").is_file()


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
    if preset.method_type == "dynamic_cache":
        threshold = _dynamic_threshold(args, preset)
        method.update(
            {
                "dynamic_cache_threshold": threshold,
                "resolved_dynamic_cache_threshold": threshold,
                "sea_beta": args.sea_beta,
                "sea_proxy_downsample": args.sea_proxy_downsample,
                "cache_units": "jit_blocks",
                "selected_modules": (preset.cache_preset or {}).get("cache_layers", "all"),
            }
        )
    elif preset.method_type == "safe_cache":
        cache_preset = preset.cache_preset or {}
        method.update(
            {
                "cache_units": cache_preset.get("cache_units", "jit_safe_whole_backbone"),
                "selected_modules": cache_preset.get("cache_layers", "all"),
                "safe_map_path": str(args.safe_map.resolve()) if args.safe_map else None,
                "safe_map_exists": bool(args.safe_map and args.safe_map.is_file()),
                "safe_map_mode": args.safe_map_mode,
                "safe_max_age_override": args.safe_max_age,
                "safe_fallback_global_branch": args.safe_fallback_global_branch,
            }
        )
    return method


def _pixbfc_static_meta(method_type: str) -> dict[str, Any]:
    adapter = JiTBoundaryAdapter()
    boundary_set = None
    if method_type in {"cache", "safe_cache", "dynamic_cache"}:
        boundary_set = {
            "name": "jit_whole_backbone",
            "description": "Resolved to concrete JiT block names after model construction.",
            "module_names": "all_blocks",
            "resolved_after_model_load": True,
        }
    return {
        "pixbfc_adapter": adapter.describe(),
        "prediction_type": adapter.prediction_type.value,
        "output_to_velocity": "xpred_to_velocity",
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
            noise_scale
            * torch.randn(1, 3, img_size, img_size, generator=generator, dtype=torch.float32)
        )
    return torch.cat(chunks, dim=0).to(device)


def load_jit_runtime_helpers() -> tuple[Any, Any, Any]:
    from pfc.eval.jit_runtime import JiTRuntimeConfig, load_jit_model, sample_jit

    return JiTRuntimeConfig, load_jit_model, sample_jit


def compute_shard_indices(num_images: int, num_shards: int, shard_index: int, shard_mode: str) -> list[int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if shard_mode == "strided":
        return [idx for idx in range(num_images) if idx % num_shards == shard_index]
    if shard_mode == "contiguous":
        base, extra = divmod(num_images, num_shards)
        start = shard_index * base + min(shard_index, extra)
        end = start + base + (1 if shard_index < extra else 0)
        return list(range(start, end))
    raise ValueError(f"Unsupported shard_mode: {shard_mode}")


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[start : start + chunk_size] for start in range(0, len(values), chunk_size)]


def _apply_shard_paths(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Path]:
    if args.num_shards <= 1:
        return paths
    suffix = args.manifest_suffix or f"_shard{args.shard_index}"
    base = paths["base_dir"]
    paths = dict(paths)
    paths["manifest"] = base / f"manifest{suffix}.jsonl"
    paths["generation_meta"] = base / f"generation_meta{suffix}.json"
    paths["latency"] = base / f"latency{suffix}.json"
    paths["cache_stats"] = base / f"cache_stats{suffix}.json"
    paths["labels_json_shard"] = base / f"labels{suffix}.json"
    paths["labels_csv_shard"] = base / f"labels{suffix}.csv"
    return paths


def _write_label_schedule_if_same(labels: list[int], base_dir: Path) -> None:
    json_path = base_dir / "labels.json"
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8")).get("labels", [])
        if [int(item) for item in existing] != [int(item) for item in labels]:
            raise RuntimeError(f"Existing label schedule differs: {json_path}")
        return
    save_label_schedule(labels, base_dir)


def _write_shard_label_schedule(labels: list[int], indices: list[int], paths: dict[str, Path]) -> None:
    shard_labels = [int(labels[index]) for index in indices]
    if "labels_json_shard" in paths:
        save_label_schedule(shard_labels, paths["labels_json_shard"])
        save_label_schedule(shard_labels, paths["labels_csv_shard"])


def _safe_map_density_from_path(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return compute_safe_map_density(json.loads(path.read_text(encoding="utf-8")))


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.cache.cache_state import RuntimeCacheState
    from pfc.cache.dynamic_policy_adapter import DynamicPolicyAdapter
    from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
    from pfc.cache.safe_map_policy import SafeMapCachePolicy
    from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy

    JiTRuntimeConfig, load_jit_model, sample_jit = load_jit_runtime_helpers()

    if args.save_npz and args.num_images > 5000:
        raise RuntimeError("--save-npz is intended for small/proxy Stage 4A runs, not large 50k runs")
    preset = get_jit_stage4a_methods()[args.method]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")
    labels = make_imagenet_class_balanced_labels(args.num_images)
    paths = resolved["paths"]
    shard_indices = compute_shard_indices(args.num_images, args.num_shards, args.shard_index, args.shard_mode)
    if args.num_shards == 1 or args.shard_index == 0:
        _write_label_schedule_if_same(labels, paths["base_dir"])
    _write_shard_label_schedule(labels, shard_indices, paths)
    config = JiTRuntimeConfig(
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
    model = load_jit_model(config, device)
    boundary_adapter = JiTBoundaryAdapter()
    cache_state: RuntimeCacheState | None = None
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None
    safe_policy: SafeMapCachePolicy | None = None
    if preset.method_type == "cache":
        boundary_set = boundary_adapter.default_boundary_set(model, args.method)
        selected_modules = list(boundary_set.module_names())
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
        boundary_adapter.wrap_boundary_set(model, boundary_set, cache_state, policy)
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "jit_blocks"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
    elif preset.method_type == "safe_cache":
        if args.safe_map is None:
            raise ValueError("--safe-map is required for Safe-BFC generation")
        boundary_set = boundary_adapter.default_boundary_set(model, args.method)
        selected_modules = list(boundary_set.module_names())
        cache_state = RuntimeCacheState(model_name="JiT", enabled=bool(selected_modules))
        safe_policy = SafeMapCachePolicy.from_path(
            args.safe_map,
            enabled=bool(selected_modules),
            model_name="JiT",
            max_age=args.safe_max_age,
            fallback_to_global_branch=args.safe_fallback_global_branch,
            debug_jsonl_path=args.safe_debug_jsonl,
        )
        boundary_adapter.wrap_boundary_set(model, boundary_set, cache_state, safe_policy)
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "jit_safe_whole_backbone"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        resolved["meta"]["safe_policy"] = safe_policy.to_dict()
        resolved["meta"]["safe_map_density"] = safe_policy.safe_density
        resolved["meta"]["safe_lambda"] = safe_policy.safe_lambda
        resolved["meta"]["safe_quantile"] = safe_policy.quantile
        resolved["meta"]["safe_max_age"] = safe_policy.max_age
        resolved["meta"]["lte_floor"] = safe_policy.lte_floor
        resolved["meta"]["boundary_groups"] = safe_policy.boundary_groups
        resolved["meta"]["module_to_boundary"] = safe_policy.module_to_boundary
    elif preset.method_type == "dynamic_cache":
        boundary_set = boundary_adapter.default_boundary_set(model, args.method)
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
        cache_state = RuntimeCacheState(model_name="JiT", enabled=bool(selected_modules))
        adapter = DynamicPolicyAdapter.from_branches(
            dynamic_policy,
            {"cond", "uncond"},
            cache_modules=set(selected_modules),
            solver_stages={"euler"},
        )
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "jit_blocks"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        boundary_adapter.wrap_boundary_set(model, boundary_set, cache_state, adapter)

    samples_for_npz = []
    labels_for_npz: list[int] = []
    generated = 0
    existing_images_skipped = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    debug_handle, dynamic_decision_writer = _dynamic_writer(args.dynamic_cache_debug_jsonl)
    try:
        for indices in _chunked(shard_indices, args.batch_size):
            if args.resume and args.save_png:
                pending_indices = [index for index in indices if not (paths["image_dir"] / f"{index:06d}.png").exists()]
                existing_images_skipped += len(indices) - len(pending_indices)
                indices = pending_indices
                if not indices:
                    continue
            batch_labels_list = [labels[index] for index in indices]
            batch_labels = torch.tensor(batch_labels_list, device=device, dtype=torch.long)
            noise = _make_noise_for_indices(indices, args.seed, args.img_size, args.noise_scale, device)
            batch_config = replace(
                config,
                num_samples=len(indices),
                batch_size=len(indices),
                dynamic_proxy_downsample=args.sea_proxy_downsample,
            )
            if cache_state is not None:
                cache_state.clear_entries()
            with torch.no_grad():
                output, _records = sample_jit(
                    model,
                    batch_labels,
                    noise,
                    batch_config,
                    mode=args.method,
                    cache_state=cache_state,
                    dynamic_policy=dynamic_policy,
                    dynamic_decision_writer=dynamic_decision_writer,
                )
            output_cpu = output.detach().cpu()
            if args.save_png:
                records = save_image_batch_png(output_cpu, batch_labels_list, indices, paths["image_dir"])
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
    if safe_policy is not None:
        cache_stats["safe_policy"] = safe_policy.summary()
        resolved["meta"]["safe_policy_summary"] = safe_policy.summary()
    write_generation_meta(paths["latency"], {
        "latency_sec": latency,
        "images_per_sec": generated / latency if latency > 0 else float("inf"),
        "generated_images": generated,
        "requested_images": args.num_images,
        "generated_images_this_run": generated,
        "existing_images_skipped": existing_images_skipped,
        "total_shard_images": len(shard_indices),
        "total_images_available": count_images(paths["image_dir"]),
        "resume": args.resume,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_mode": args.shard_mode,
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
    parser.add_argument("--dynamic-cache-threshold", type=float)
    parser.add_argument("--sea-beta", type=float, default=2.0)
    parser.add_argument("--sea-proxy-downsample", type=int, default=64)
    parser.add_argument("--sea-min-t", type=float)
    parser.add_argument("--sea-max-t", type=float)
    parser.add_argument("--dynamic-force-first-n-steps", type=int, default=0)
    parser.add_argument("--dynamic-per-branch", action="store_true")
    parser.add_argument("--dynamic-cache-debug-jsonl", type=Path)
    parser.add_argument("--safe-map", type=Path)
    parser.add_argument("--safe-map-mode", choices=("quality", "speed", "custom"), default="custom")
    parser.add_argument("--safe-debug-jsonl", type=Path)
    parser.add_argument("--safe-max-age", type=int)
    parser.add_argument("--safe-fallback-global-branch", dest="safe_fallback_global_branch", action="store_true", default=True)
    parser.add_argument("--no-safe-fallback-global-branch", dest="safe_fallback_global_branch", action="store_false")
    parser.add_argument("--allow-empty-safe-map", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-mode", choices=("strided", "contiguous"), default="strided")
    parser.add_argument("--manifest-suffix")
    parser.add_argument("--shard-output-meta", action="store_true", default=True)
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = get_jit_stage4a_methods()[args.method]
    run_id = args.run_id or _default_run_id(args.seed, args.num_images)
    args.run_id = run_id
    paths = prepare_generation_dir(args.output_root, preset.model_name, args.method, run_id, create=not args.dry_run)
    paths = _apply_shard_paths(paths, args)
    dynamic_threshold = _dynamic_threshold(args, preset) if preset.method_type == "dynamic_cache" else None
    safe_density = _safe_map_density_from_path(args.safe_map) if preset.method_type == "safe_cache" else None
    meta = {
        "model_name": preset.model_name,
        "model": "JiT",
        "method_name": preset.method_name,
        "method_type": preset.method_type,
        "method": _resolved_method_meta(args, preset),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eval_steps": preset.eval_steps,
        "reference_steps": preset.reference_steps,
        "cache_units": (
            "jit_safe_whole_backbone"
            if preset.method_type == "safe_cache"
            else "jit_blocks"
            if preset.method_type == "dynamic_cache"
            else None
        ),
        "selected_modules": (preset.cache_preset or {}).get("cache_layers") if preset.cache_preset else None,
        "dynamic_cache_type": preset.dynamic_cache_type,
        "dynamic_cache_threshold": dynamic_threshold,
        "resolved_dynamic_cache_threshold": dynamic_threshold,
        "sea_beta": args.sea_beta if preset.method_type == "dynamic_cache" else None,
        "sea_proxy_downsample": args.sea_proxy_downsample if preset.method_type == "dynamic_cache" else None,
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
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_mode": args.shard_mode,
        "manifest_suffix": args.manifest_suffix,
        "shard_indices": compute_shard_indices(args.num_images, args.num_shards, args.shard_index, args.shard_mode),
        "dynamic_cache": {
            "threshold": dynamic_threshold,
            "resolved_dynamic_cache_threshold": dynamic_threshold,
            "sea_beta": args.sea_beta,
            "sea_proxy_downsample": args.sea_proxy_downsample,
            "sea_min_t": args.sea_min_t,
            "sea_max_t": args.sea_max_t,
            "dynamic_force_first_n_steps": args.dynamic_force_first_n_steps,
            "dynamic_per_branch": args.dynamic_per_branch,
            "dynamic_cache_debug_jsonl": str(args.dynamic_cache_debug_jsonl.resolve()) if args.dynamic_cache_debug_jsonl else None,
        },
        "safe_cache": {
            "safe_map_path": str(args.safe_map.resolve()) if args.safe_map else None,
            "safe_map_exists": bool(args.safe_map and args.safe_map.is_file()),
            "safe_map_mode": args.safe_map_mode if preset.method_type == "safe_cache" else None,
            "safe_map_density": safe_density,
            "safe_max_age_override": args.safe_max_age,
            "safe_fallback_global_branch": args.safe_fallback_global_branch,
            "safe_debug_jsonl": str(args.safe_debug_jsonl.resolve()) if args.safe_debug_jsonl else None,
            "allow_empty_safe_map": args.allow_empty_safe_map,
        },
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
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1 and args.save_npz:
        parser.error("--save-npz is not supported with --num-shards > 1")
    if args.dynamic_force_first_n_steps < 0:
        parser.error("--dynamic-force-first-n-steps must be non-negative")
    if args.sea_proxy_downsample < 0:
        parser.error("--sea-proxy-downsample must be non-negative")
    if args.safe_max_age is not None and args.safe_max_age <= 0:
        parser.error("--safe-max-age must be positive")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PFC_CUDA_DEVICES", "0"))
    resolved = resolve_config(args)
    if args.dry_run:
        _print_dry_run({"meta": resolved["meta"], "paths": resolved["paths"]})
        safe_density = resolved["meta"].get("safe_cache", {}).get("safe_map_density")
        if safe_density and safe_density.get("safe_total", 0) > 0 and safe_density.get("safe_true", 0) == 0:
            print("Warning: Safe map contains zero reusable positions; generation would degenerate to no-cache.")
        if not resolved["meta"]["checkpoint_exists"]:
            print(f"Missing JiT checkpoint: {args.jit_ckpt_dir / 'checkpoint-last.pth'}")
            return 2
        return 0
    if get_jit_stage4a_methods()[args.method].method_type == "safe_cache":
        if args.safe_map is None:
            parser.error("--safe-map is required for method_type=safe_cache")
        if not args.safe_map.is_file():
            raise FileNotFoundError(f"Missing Safe-BFC safe map: {args.safe_map}")
        safe_density = resolved["meta"].get("safe_cache", {}).get("safe_map_density")
        if (
            safe_density
            and safe_density.get("safe_total", 0) > 0
            and safe_density.get("safe_true", 0) == 0
            and not args.allow_empty_safe_map
        ):
            raise RuntimeError(
                "Safe map contains zero reusable positions; refusing to run because this will degenerate to no-cache. "
                "Pass --allow-empty-safe-map to override explicitly."
            )
    if not resolved["meta"]["checkpoint_exists"]:
        raise FileNotFoundError(f"Missing JiT checkpoint: {args.jit_ckpt_dir / 'checkpoint-last.pth'}")
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
