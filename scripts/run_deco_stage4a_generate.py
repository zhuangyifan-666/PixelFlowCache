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
    reconcile_resume_state,
    save_image_batch_png,
    save_npz_samples,
    write_generation_meta,
)
from pfc.eval.provenance import collect_generation_provenance  # noqa: E402
from pfc.eval.sharding import apply_shard_paths, compute_shard_indices  # noqa: E402
from pfc.eval.timing import GenerationTiming  # noqa: E402
from pfc.adapters import DeCoBoundaryAdapter  # noqa: E402
from pfc.eval.label_schedule import ensure_label_schedule, make_imagenet_class_balanced_labels  # noqa: E402
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
                "cache_units": preset.deco_cache_units or "all_candidates",
                "selected_modules": preset.deco_cache_units or "all_candidates",
            }
        )
    return method


def _pixbfc_static_meta(method_type: str, preset_name: str) -> dict[str, Any]:
    adapter = DeCoBoundaryAdapter()
    boundary_set = None
    if method_type in {"cache", "dynamic_cache"}:
        boundary_name = "deco_backbone_plus_final" if preset_name == "bfc_backbone_plus_final_t02_10" else "deco_all_candidates"
        boundary_set = {
            "name": boundary_name,
            "description": "Resolved to concrete DeCo module names after model construction.",
            "module_names": "deco_cache_candidates",
            "resolved_after_model_load": True,
        }
    return {
        "pixbfc_adapter": adapter.describe(),
        "prediction_type": adapter.prediction_type.value,
        "output_to_velocity": "identity",
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


def _make_noise_for_indices(indices: list[int], seed: int, resolution: int, device: Any) -> Any:
    import torch

    chunks = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + index)
        chunks.append(torch.randn(1, 3, resolution, resolution, generator=generator, dtype=torch.float32))
    return torch.cat(chunks, dim=0).to(device)


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[start : start + chunk_size] for start in range(0, len(values), chunk_size)]


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.cache.cache_state import RuntimeCacheState
    from pfc.cache.dynamic_policy_adapter import DynamicPolicyAdapter
    from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy
    from pfc.eval.deco_runtime import (
        DeCoRuntimeConfig,
        build_deco_sampler,
        load_deco_denoiser,
        policy_for_modules,
    )

    generation_started_utc = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    timing = GenerationTiming(
        requested_images=args.num_images,
        resume=args.resume,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if args.save_npz and args.num_images > 5000:
        raise RuntimeError("--save-npz is intended for small/proxy Stage 4A runs, not large 50k runs")
    preset = get_deco_stage4a_methods()[args.method]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")
    labels = make_imagenet_class_balanced_labels(args.num_images)
    paths = resolved["paths"]
    all_shard_indices = compute_shard_indices(
        args.num_images,
        args.num_shards,
        args.shard_index,
        args.shard_mode,
    )
    ensure_label_schedule(labels, paths["base_dir"])
    reconciliation = reconcile_resume_state(
        all_shard_indices,
        labels,
        paths["image_dir"],
        paths["manifest"],
        resume=args.resume,
        save_png=args.save_png,
    )
    shard_indices = reconciliation.pending_indices
    resolved["meta"]["resume_reconciliation"] = reconciliation.to_dict()
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
        dynamic_proxy_downsample=args.sea_proxy_downsample,
    )
    with timing.measure("model_load_latency_sec"):
        denoiser = load_deco_denoiser(config, device)
    resolved["meta"]["checkpoint_load_summary"] = getattr(
        denoiser,
        "_pfc_checkpoint_load_summary",
        None,
    )
    boundary_adapter = DeCoBoundaryAdapter()
    cache_state: RuntimeCacheState | None = None
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None
    if preset.method_type == "cache":
        boundary_set = boundary_adapter.default_boundary_set(denoiser, args.method)
        selected_modules = list(boundary_set.module_names())
        cache_state = RuntimeCacheState(
            model_name="DeCo",
            enabled=bool(selected_modules),
            clone_on_store=args.clone_cache_on_store,
        )
        boundary_adapter.wrap_boundary_set(denoiser, boundary_set, cache_state, policy_for_modules(config, selected_modules))
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = config.cache_units
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
    elif preset.method_type == "dynamic_cache":
        boundary_set = boundary_adapter.default_boundary_set(denoiser, args.method)
        selected_modules = list(boundary_set.module_names())
        threshold = _dynamic_threshold(args, preset)
        policy_kwargs = {
            "threshold": threshold,
            "force_first_n_steps": args.dynamic_force_first_n_steps,
            "min_t": args.sea_min_t,
            "max_t": args.sea_max_t,
            "per_branch": False,
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
        cache_state = RuntimeCacheState(
            model_name="DeCo",
            enabled=bool(selected_modules),
            clone_on_store=args.clone_cache_on_store,
        )
        adapter = DynamicPolicyAdapter(
            dynamic_policy=dynamic_policy,
            cache_modules=set(selected_modules),
            solver_stages={"euler"},
        )
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = config.cache_units
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        boundary_adapter.wrap_boundary_set(denoiser, boundary_set, cache_state, adapter)
    debug_handle, dynamic_decision_writer = _dynamic_writer(args.dynamic_cache_debug_jsonl)
    sampler = build_deco_sampler(
        config,
        cache_state=cache_state,
        dynamic_policy=dynamic_policy,
        dynamic_decision_writer=dynamic_decision_writer,
        log_diagnostics=False,
    )
    samples_for_npz = []
    labels_for_npz: list[int] = []
    generated = 0
    existing_images_skipped = len(reconciliation.complete_indices)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for indices in _chunked(shard_indices, args.batch_size):
            with timing.measure("input_prepare_latency_sec"):
                batch_labels_list = [labels[index] for index in indices]
                batch_labels = torch.tensor(batch_labels_list, device=device, dtype=torch.long)
                batch_uncondition = torch.full_like(batch_labels, 1000)
                batch_noise = _make_noise_for_indices(indices, args.seed, args.resolution, device)
                batch_config = replace(config, num_samples=len(indices), batch_size=len(indices))
            sampler.num_steps = batch_config.steps
            if cache_state is not None:
                cache_state.begin_batch(session_id=f"{args.run_id}:{indices[0]}")
            with timing.measure("sampling_latency_sec", device=device, synchronize=True):
                with torch.no_grad():
                    output = sampler(denoiser, batch_noise, batch_labels, batch_uncondition)
            with timing.measure("postprocess_latency_sec"):
                output = output.detach().cpu()
            if args.save_png:
                with timing.measure("png_save_latency_sec"):
                    records = save_image_batch_png(output, batch_labels_list, indices, paths["image_dir"])
            else:
                records = [{"index": index, "label": int(label)} for index, label in zip(indices, batch_labels_list)]
            with timing.measure("manifest_latency_sec"):
                append_generation_manifest(paths["manifest"], records)
            if args.save_npz:
                samples_for_npz.append(output)
                labels_for_npz.extend(batch_labels_list)
            generated += len(indices)
    finally:
        if debug_handle is not None:
            debug_handle.close()
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if args.save_npz and samples_for_npz:
        with timing.measure("npz_save_latency_sec"):
            save_npz_samples(torch.cat(samples_for_npz, dim=0), labels_for_npz, paths["samples_npz"])
    cache_stats = cache_state.summary() if cache_state is not None else {"enabled": False, "hit_rate": 0.0}
    if dynamic_policy is not None:
        cache_stats["dynamic_cache"] = dynamic_policy.summary()
        cache_stats["dynamic_cache_threshold"] = dynamic_policy.threshold
        cache_stats["resolved_dynamic_cache_threshold"] = dynamic_policy.threshold
        resolved["meta"]["dynamic_cache_summary"] = dynamic_policy.summary()
    timing.generated_images_this_run = generated
    timing.existing_images_skipped = existing_images_skipped
    timing.total_images_available = count_images(paths["image_dir"])
    timing.peak_memory_allocated_bytes = peak_memory
    timing.end_to_end_latency_sec = time.perf_counter() - end_to_end_started
    write_generation_meta(
        paths["latency"],
        {
            **timing.to_dict(),
            "generated_images": generated,
            "total_shard_images": len(all_shard_indices),
            "shard_mode": args.shard_mode,
        },
    )
    write_generation_meta(paths["cache_stats"], cache_stats)
    resolved["meta"].update(
        {
            "generation_start_utc": generation_started_utc,
            "generation_end_utc": datetime.now(timezone.utc).isoformat(),
            "timing_scope": timing.timing_scope,
            "timing_schema_version": 2,
            "clone_cache_on_store": args.clone_cache_on_store,
            "provenance": collect_generation_provenance(
                ROOT,
                checkpoint_path=args.deco_ckpt,
                hash_checkpoint=args.hash_checkpoints,
            ),
        }
    )
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
    parser.add_argument("--allow-partial-npz", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument("--clone-cache-on-store", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--deco-dir", type=Path, default=ROOT / "third_party/DeCo")
    parser.add_argument("--deco-ckpt", type=Path, default=ROOT / "ckpts/DeCo/DeCo_XL.ckpt")
    parser.add_argument("--deco-config", type=Path, default=ROOT / "third_party/DeCo/configs_c2i/DeCo_XL.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg", type=float, default=3.2)
    parser.add_argument("--cfg-interval-min", type=float, default=0.1)
    parser.add_argument("--cfg-interval-max", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--dynamic-cache-threshold", type=float)
    parser.add_argument("--sea-beta", type=float, default=2.0)
    parser.add_argument("--sea-proxy-downsample", type=int, default=64)
    parser.add_argument("--sea-min-t", type=float)
    parser.add_argument("--sea-max-t", type=float)
    parser.add_argument("--dynamic-force-first-n-steps", type=int, default=0)
    parser.add_argument("--dynamic-cache-debug-jsonl", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-mode", choices=("strided", "contiguous"), default="strided")
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = get_deco_stage4a_methods()[args.method]
    run_id = args.run_id or _default_run_id(args.seed, args.num_images)
    args.run_id = run_id
    paths = prepare_generation_dir(args.output_root, preset.model_name, args.method, run_id, create=not args.dry_run)
    paths = apply_shard_paths(
        paths,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    shard_indices = compute_shard_indices(
        args.num_images,
        args.num_shards,
        args.shard_index,
        args.shard_mode,
    )
    dynamic_threshold = _dynamic_threshold(args, preset) if preset.method_type == "dynamic_cache" else None
    meta = {
        "model_name": preset.model_name,
        "model": "DeCo",
        "method_name": preset.method_name,
        "method": _resolved_method_meta(args, preset),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eval_steps": preset.eval_steps,
        "reference_steps": preset.reference_steps,
        "cache_units": preset.deco_cache_units if preset.method_type == "dynamic_cache" else None,
        "selected_modules": preset.deco_cache_units if preset.method_type == "dynamic_cache" else None,
        "dynamic_cache_type": preset.dynamic_cache_type,
        "dynamic_cache_threshold": dynamic_threshold,
        "resolved_dynamic_cache_threshold": dynamic_threshold,
        "sea_beta": args.sea_beta if preset.method_type == "dynamic_cache" else None,
        "sea_proxy_downsample": args.sea_proxy_downsample if preset.method_type == "dynamic_cache" else None,
        "save_png": args.save_png,
        "save_npz": args.save_npz,
        "resume": args.resume,
        "allow_partial_npz": args.allow_partial_npz,
        "partial_npz": bool(args.resume and args.save_npz and args.allow_partial_npz),
        "warmup_batches": args.warmup_batches,
        "clone_cache_on_store": args.clone_cache_on_store,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_mode": args.shard_mode,
        "shard_indices": shard_indices,
        "deco_dir": str(args.deco_dir.resolve()),
        "deco_ckpt": str(args.deco_ckpt.resolve()),
        "checkpoint_exists": _checkpoint_ok(args.deco_ckpt.resolve()),
        "deco_config": str(args.deco_config.resolve()),
        "config_exists": args.deco_config.resolve().exists(),
        "device": args.device,
        "cfg": args.cfg,
        "resolution": args.resolution,
        "run_id": run_id,
        "dynamic_cache": {
            "threshold": dynamic_threshold,
            "resolved_dynamic_cache_threshold": dynamic_threshold,
            "sea_beta": args.sea_beta,
            "sea_proxy_downsample": args.sea_proxy_downsample,
            "sea_min_t": args.sea_min_t,
            "sea_max_t": args.sea_max_t,
            "dynamic_force_first_n_steps": args.dynamic_force_first_n_steps,
            "dynamic_cache_debug_jsonl": str(args.dynamic_cache_debug_jsonl.resolve()) if args.dynamic_cache_debug_jsonl else None,
        },
        **_pixbfc_static_meta(preset.method_type, preset.method_name),
    }
    return {"meta": meta, "paths": paths}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.resume and not args.save_png:
        raise ValueError("Resume requires PNG completion markers in the current implementation.")
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be non-negative")
    if args.warmup_batches:
        parser.error("--warmup-batches is supported only by the JiT single-GPU timing suite")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")
    if args.dynamic_force_first_n_steps < 0:
        parser.error("--dynamic-force-first-n-steps must be non-negative")
    if args.sea_proxy_downsample < 0:
        parser.error("--sea-proxy-downsample must be non-negative")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PFC_CUDA_DEVICES", "0"))
    resolved = resolve_config(args)
    if args.dry_run:
        print(json.dumps(_json_ready({"meta": resolved["meta"], "paths": resolved["paths"]}), indent=2, sort_keys=True))
        if not resolved["meta"]["checkpoint_exists"]:
            print(f"Missing DeCo checkpoint: {args.deco_ckpt}")
        if not resolved["meta"]["config_exists"]:
            print(f"Missing DeCo config: {args.deco_config}")
        return 0
    if args.resume and args.save_npz and not args.allow_partial_npz:
        raise ValueError(
            "NPZ resume is not supported because the in-memory tensor set may be incomplete."
        )
    if not resolved["meta"]["checkpoint_exists"]:
        raise FileNotFoundError(f"Missing DeCo checkpoint: {args.deco_ckpt}")
    if not resolved["meta"]["config_exists"]:
        raise FileNotFoundError(f"Missing DeCo config: {args.deco_config}")
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
