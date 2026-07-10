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
from pfc.eval.sharding import (  # noqa: E402
    apply_shard_paths,
    compute_shard_indices as _compute_shard_indices,
)
from pfc.eval.timing import GenerationTiming  # noqa: E402
from pfc.adapters import JiTBoundaryAdapter  # noqa: E402
from pfc.cache.safe_map_policy import compute_safe_map_density  # noqa: E402
from pfc.cache.speca_policy import resolve_verifier_module  # noqa: E402
from pfc.eval.label_schedule import ensure_label_schedule, make_imagenet_class_balanced_labels, save_label_schedule  # noqa: E402
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


def _taylorseer_interval(args: argparse.Namespace, preset: Any) -> int:
    return int(args.taylorseer_interval or preset.taylorseer_interval or 4)


def _taylorseer_max_order(args: argparse.Namespace, preset: Any) -> int:
    return int(args.taylorseer_max_order or preset.taylorseer_max_order or 4)


def _taylorseer_refresh_first_n_steps(args: argparse.Namespace, preset: Any) -> int:
    value = args.taylorseer_refresh_first_n_steps
    if value is None:
        value = preset.taylorseer_refresh_first_n_steps
    if value is None:
        value = 1
    return int(value)


def _taylorseer_refresh_last_n_steps(args: argparse.Namespace, preset: Any) -> int:
    value = args.taylorseer_refresh_last_n_steps
    if value is None:
        value = preset.taylorseer_refresh_last_n_steps
    if value is None:
        value = 0
    return int(value)


def _taylorseer_config(args: argparse.Namespace, preset: Any) -> dict[str, Any]:
    return {
        "baseline_name": "TaylorSeer-style",
        "official_reference": "TaylorSeer adapted baseline, not official reproduction",
        "taylorseer_interval": _taylorseer_interval(args, preset),
        "taylorseer_max_order": _taylorseer_max_order(args, preset),
        "taylorseer_min_history": args.taylorseer_min_history,
        "taylorseer_refresh_first_n_steps": _taylorseer_refresh_first_n_steps(args, preset),
        "taylorseer_refresh_last_n_steps": _taylorseer_refresh_last_n_steps(args, preset),
        "taylorseer_clone_forecast": args.taylorseer_clone_forecast,
        "taylorseer_debug_jsonl": str(args.taylorseer_debug_jsonl.resolve()) if args.taylorseer_debug_jsonl else None,
    }


def _speca_value(args: argparse.Namespace, preset: Any, name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is None:
        value = getattr(preset, name, None)
    return default if value is None else value


def _dicache_value(args: argparse.Namespace, preset: Any, name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is None:
        value = getattr(preset, name, None)
    return default if value is None else value


def _declared_jit_modules(model_name: str) -> list[str]:
    declared_depths = {
        "JiT-B/16": 12,
    }
    depth = declared_depths.get(str(model_name))
    return [f"blocks.{idx}" for idx in range(depth)] if depth is not None else []


def _declared_jit_structure(model_name: str) -> dict[str, int | None]:
    structures = {
        "JiT-B/16": {"total_blocks": 12, "in_context_start": 4, "in_context_len": 32},
    }
    return structures.get(
        str(model_name),
        {"total_blocks": None, "in_context_start": None, "in_context_len": None},
    )


def _speca_config(
    args: argparse.Namespace,
    preset: Any,
    selected_modules: list[str] | None = None,
) -> dict[str, Any]:
    requested = str(_speca_value(args, preset, "speca_verifier_module", "auto"))
    selected = selected_modules if selected_modules is not None else _declared_jit_modules(args.jit_model)
    resolved = resolve_verifier_module(selected, requested) if selected else None
    return {
        "baseline_name": "adapted SpeCa-style",
        "official_reproduction": False,
        "draft_type": "adapted TaylorSeer block-output forecasting",
        "verification_type": "last JiT transformer block",
        "decision_timing": "next_step",
        "sample_adaptivity": "batch-level",
        "speca_max_order": int(_speca_value(args, preset, "speca_max_order", 4)),
        "speca_first_full_steps": int(_speca_value(args, preset, "speca_first_full_steps", 3)),
        "speca_base_threshold": float(_speca_value(args, preset, "speca_base_threshold", 0.1)),
        "speca_decay_rate": float(_speca_value(args, preset, "speca_decay_rate", 0.01)),
        "speca_min_threshold": float(_speca_value(args, preset, "speca_min_threshold", 0.01)),
        "speca_min_forecast_steps": int(_speca_value(args, preset, "speca_min_forecast_steps", 2)),
        "speca_max_forecast_steps": int(_speca_value(args, preset, "speca_max_forecast_steps", 5)),
        "speca_error_metric": str(_speca_value(args, preset, "speca_error_metric", "relative_l1")),
        "speca_branch_aggregation": str(_speca_value(args, preset, "speca_branch_aggregation", "mean")),
        "speca_verifier_module_requested": requested,
        "speca_verifier_module_resolved": resolved,
        "speca_min_history": int(_speca_value(args, preset, "speca_min_history", 2)),
        "speca_clone_forecast": bool(args.speca_clone_forecast),
        "speca_eps": float(args.speca_eps),
        "speca_max_error_samples": int(args.speca_max_error_samples),
        "speca_debug_jsonl": str(args.speca_debug_jsonl.resolve()) if args.speca_debug_jsonl else None,
        "selected_modules": selected or "resolved_after_model_load",
    }


def _dicache_config(args: argparse.Namespace, preset: Any) -> dict[str, Any]:
    structure = _declared_jit_structure(args.jit_model)
    total_steps = int(preset.eval_steps)
    ret_ratio = float(_dicache_value(args, preset, "dicache_ret_ratio", 0.2))
    share_cfg_prefix = bool(
        _dicache_value(args, preset, "dicache_share_cfg_prefix", False)
    )
    schedule_variant = str(
        _dicache_value(
            args,
            preset,
            "dicache_schedule_variant",
            "released_flux_compat",
        )
    )
    retention_last = min(total_steps - 1, int(ret_ratio * total_steps))
    return {
        "baseline_name": "adapted DiCache-style",
        "official_reproduction": False,
        "official_reference_backend": "FLUX released example",
        "model_name": "JiT",
        "model_space": "pixel",
        "prediction_type": "xpred",
        "solver_stage": "euler",
        "schedule_granularity": "batch_level_shared_cfg",
        "residual_space": "image_token_block_stack",
        "probe_depth": int(_dicache_value(args, preset, "dicache_probe_depth", 1)),
        "reuse_threshold": float(_dicache_value(args, preset, "dicache_reuse_threshold", 0.4)),
        "error_choice": str(_dicache_value(args, preset, "dicache_error_choice", "delta_y")),
        "branch_aggregation": str(
            _dicache_value(args, preset, "dicache_branch_aggregation", "mean")
        ),
        "ret_ratio": ret_ratio,
        "force_last_step_full": bool(
            _dicache_value(args, preset, "dicache_force_last_step_full", True)
        ),
        "dcta_enabled": bool(_dicache_value(args, preset, "dicache_dcta_enabled", True)),
        "gamma_min": float(_dicache_value(args, preset, "dicache_gamma_min", 1.0)),
        "gamma_max": float(_dicache_value(args, preset, "dicache_gamma_max", 1.5)),
        "eps": float(_dicache_value(args, preset, "dicache_eps", 1e-10)),
        "max_stat_samples": int(
            _dicache_value(args, preset, "dicache_max_stat_samples", 4096)
        ),
        "share_cfg_prefix": share_cfg_prefix,
        "schedule_variant": schedule_variant,
        "dicache_share_cfg_prefix": share_cfg_prefix,
        "dicache_schedule_variant": schedule_variant,
        "cfg_prefix_fairness_mode": (
            "shared_prefix_ablation"
            if share_cfg_prefix
            else "strict_no_cache_matched"
        ),
        "cfg_prefix_ablation_name": (
            "dicache_style_shared_cfg_prefix" if share_cfg_prefix else None
        ),
        "decision_rule": "strict_lt",
        "clone_history": bool(args.dicache_clone_history),
        "force_full": bool(args.dicache_force_full),
        "debug_jsonl": str(args.dicache_debug_jsonl.resolve()) if args.dicache_debug_jsonl else None,
        "debug_sync_overhead_enabled": args.dicache_debug_jsonl is not None,
        "retention_full_last_step_idx": retention_last,
        "retention_full_step_count": retention_last + 1,
        **structure,
    }


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
    elif preset.method_type == "forecast_cache":
        cache_preset = preset.cache_preset or {}
        method.update(
            {
                "cache_units": cache_preset.get("cache_units", "jit_blocks"),
                "selected_modules": cache_preset.get("cache_layers", "all"),
                **_taylorseer_config(args, preset),
            }
        )
    elif preset.method_type == "speculative_cache":
        cache_preset = preset.cache_preset or {}
        method.update(
            {
                "cache_units": cache_preset.get("cache_units", "jit_blocks"),
                **_speca_config(args, preset),
            }
        )
    elif preset.method_type == "probe_cache":
        method.update(
            {
                "cache_units": "jit_block_stack_residual",
                **_dicache_config(args, preset),
            }
        )
    return method


def _pixbfc_static_meta(method_type: str) -> dict[str, Any]:
    adapter = JiTBoundaryAdapter()
    boundary_set = None
    if method_type in {"cache", "safe_cache", "dynamic_cache", "forecast_cache", "speculative_cache"}:
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
    return _compute_shard_indices(num_images, num_shards, shard_index, shard_mode)


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[start : start + chunk_size] for start in range(0, len(values), chunk_size)]


def _apply_shard_paths(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Path]:
    paths = apply_shard_paths(
        paths,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        suffix=args.manifest_suffix,
    )
    if args.num_shards <= 1:
        return paths
    suffix = args.manifest_suffix or f"_shard{args.shard_index}"
    base = paths["base_dir"]
    paths["labels_json_shard"] = base / f"labels{suffix}.json"
    paths["labels_csv_shard"] = base / f"labels{suffix}.csv"
    return paths


def _write_label_schedule_if_same(labels: list[int], base_dir: Path) -> None:
    ensure_label_schedule(labels, base_dir)


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
    from pfc.cache.dicache_policy import DiCachePolicy
    from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy
    from pfc.cache.safe_map_policy import SafeMapCachePolicy
    from pfc.cache.speca_policy import SpeCaCachePolicy
    from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy, SeaCacheSpectralDistancePolicy
    from pfc.cache.taylorseer_policy import TaylorSeerCachePolicy
    from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor, sample_jit_dicache

    JiTRuntimeConfig, load_jit_model, sample_jit = load_jit_runtime_helpers()

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
    preset = get_jit_stage4a_methods()[args.method]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")
    labels = make_imagenet_class_balanced_labels(args.num_images)
    paths = resolved["paths"]
    all_shard_indices = compute_shard_indices(args.num_images, args.num_shards, args.shard_index, args.shard_mode)
    _write_label_schedule_if_same(labels, paths["base_dir"])
    _write_shard_label_schedule(labels, all_shard_indices, paths)
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
    with timing.measure("model_load_latency_sec"):
        model = load_jit_model(config, device)
    boundary_adapter = JiTBoundaryAdapter()
    cache_state: RuntimeCacheState | None = None
    dynamic_policy: RawAccumulatedDistancePolicy | SeaCacheSpectralDistancePolicy | None = None
    safe_policy: SafeMapCachePolicy | None = None
    taylorseer_policy: TaylorSeerCachePolicy | None = None
    speca_policy: SpeCaCachePolicy | None = None
    dicache_policy: DiCachePolicy | None = None
    dicache_executor: JiTDiCacheExecutor | None = None
    if preset.method_type == "cache":
        boundary_set = boundary_adapter.default_boundary_set(model, args.method)
        selected_modules = list(boundary_set.module_names())
        cache_state = RuntimeCacheState(model_name="JiT", enabled=True, clone_on_store=args.clone_cache_on_store)
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
        cache_state = RuntimeCacheState(model_name="JiT", enabled=bool(selected_modules), clone_on_store=args.clone_cache_on_store)
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
    elif preset.method_type == "forecast_cache":
        boundary_set = boundary_adapter.default_boundary_set(model, args.method)
        selected_modules = list(boundary_set.module_names())
        cache_state = RuntimeCacheState(model_name="JiT", enabled=bool(selected_modules), clone_on_store=args.clone_cache_on_store)
        taylorseer_policy = TaylorSeerCachePolicy(
            enabled=bool(selected_modules),
            model_name="JiT",
            cache_modules=set(selected_modules),
            interval=_taylorseer_interval(args, preset),
            max_order=_taylorseer_max_order(args, preset),
            min_history=args.taylorseer_min_history,
            solver_stages={"euler"},
            branches={"cond", "uncond", "global"},
            fallback_to_global_branch=True,
            refresh_first_n_steps=_taylorseer_refresh_first_n_steps(args, preset),
            refresh_last_n_steps=_taylorseer_refresh_last_n_steps(args, preset),
            clone_forecast=args.taylorseer_clone_forecast,
            debug_jsonl_path=args.taylorseer_debug_jsonl,
            total_steps=preset.eval_steps,
        )
        boundary_adapter.wrap_boundary_set(model, boundary_set, cache_state, taylorseer_policy)
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "jit_blocks"
        resolved["meta"]["baseline_name"] = "TaylorSeer-style"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        resolved["meta"]["taylorseer_policy"] = taylorseer_policy.to_dict()
        resolved["meta"].update(_taylorseer_config(args, preset))
    elif preset.method_type == "speculative_cache":
        boundary_set = boundary_adapter.default_boundary_set(model, args.method)
        selected_modules = list(boundary_set.module_names())
        verifier_requested = str(_speca_value(args, preset, "speca_verifier_module", "auto"))
        cache_state = RuntimeCacheState(model_name="JiT", enabled=bool(selected_modules), clone_on_store=args.clone_cache_on_store)
        speca_policy = SpeCaCachePolicy(
            enabled=bool(selected_modules),
            model_name="JiT",
            cache_modules=set(selected_modules),
            max_order=int(_speca_value(args, preset, "speca_max_order", 4)),
            first_full_steps=int(_speca_value(args, preset, "speca_first_full_steps", 3)),
            base_threshold=float(_speca_value(args, preset, "speca_base_threshold", 0.1)),
            decay_rate=float(_speca_value(args, preset, "speca_decay_rate", 0.01)),
            min_threshold=float(_speca_value(args, preset, "speca_min_threshold", 0.01)),
            min_forecast_steps=int(_speca_value(args, preset, "speca_min_forecast_steps", 2)),
            max_forecast_steps=int(_speca_value(args, preset, "speca_max_forecast_steps", 5)),
            error_metric=str(_speca_value(args, preset, "speca_error_metric", "relative_l1")),
            branch_aggregation=str(_speca_value(args, preset, "speca_branch_aggregation", "mean")),
            min_history=int(_speca_value(args, preset, "speca_min_history", 2)),
            verifier_module=verifier_requested,
            solver_stages={"euler"},
            branches={"cond", "uncond", "global"},
            verification_branches=("cond", "uncond"),
            fallback_to_global_branch=True,
            clone_forecast=args.speca_clone_forecast,
            debug_jsonl_path=args.speca_debug_jsonl,
            total_steps=preset.eval_steps,
            eps=args.speca_eps,
            max_verification_error_samples=args.speca_max_error_samples,
        )
        boundary_adapter.wrap_boundary_set(model, boundary_set, cache_state, speca_policy)
        resolved["meta"].update(_speca_config(args, preset, selected_modules))
        resolved["meta"]["method_type"] = "speculative_cache"
        resolved["meta"]["selected_modules"] = selected_modules
        resolved["meta"]["cache_units"] = "jit_blocks"
        resolved["meta"]["boundary_set"] = boundary_set.to_dict()
        resolved["meta"]["speca_policy"] = speca_policy.to_dict()
        resolved["meta"]["speca_verifier_module_requested"] = speca_policy.verifier_module_requested
        resolved["meta"]["speca_verifier_module_resolved"] = speca_policy.verifier_module_resolved
    elif preset.method_type == "probe_cache":
        settings = _dicache_config(args, preset)
        dicache_executor = JiTDiCacheExecutor(model.net)
        dicache_policy = DiCachePolicy(
            total_blocks=dicache_executor.total_blocks,
            total_steps=preset.eval_steps,
            probe_depth=settings["probe_depth"],
            reuse_threshold=settings["reuse_threshold"],
            error_choice=settings["error_choice"],
            branch_aggregation=settings["branch_aggregation"],
            ret_ratio=settings["ret_ratio"],
            force_last_step_full=settings["force_last_step_full"],
            dcta_enabled=settings["dcta_enabled"],
            gamma_min=settings["gamma_min"],
            gamma_max=settings["gamma_max"],
            eps=settings["eps"],
            clone_history=settings["clone_history"],
            debug_jsonl=args.dicache_debug_jsonl,
            max_error_samples=settings["max_stat_samples"],
            schedule_variant=settings["schedule_variant"],
            share_cfg_prefix=settings["share_cfg_prefix"],
            force_full=settings["force_full"],
        )
        settings.update(
            {
                "total_blocks": dicache_executor.total_blocks,
                "in_context_start": int(model.net.in_context_start),
                "in_context_len": int(model.net.in_context_len),
            }
        )
        resolved["meta"].update(settings)
        resolved["meta"]["dicache_cache"] = settings
        resolved["meta"]["cache_units"] = "jit_block_stack_residual"
        resolved["meta"]["selected_modules"] = []
        resolved["meta"]["boundary_set"] = None
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
        cache_state = RuntimeCacheState(model_name="JiT", enabled=bool(selected_modules), clone_on_store=args.clone_cache_on_store)
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

    if args.warmup_batches and shard_indices:
        warm_indices = shard_indices[: min(args.batch_size, len(shard_indices))]
        with timing.measure("warmup_latency_sec", device=device, synchronize=True):
            with torch.no_grad():
                for warmup_index in range(args.warmup_batches):
                    warm_labels = torch.tensor(
                        [labels[index] for index in warm_indices],
                        device=device,
                        dtype=torch.long,
                    )
                    warm_noise = _make_noise_for_indices(
                        warm_indices,
                        args.seed + 1_000_000 + warmup_index,
                        args.img_size,
                        args.noise_scale,
                        device,
                    )
                    warm_config = replace(
                        config,
                        num_samples=len(warm_indices),
                        batch_size=len(warm_indices),
                        dynamic_proxy_downsample=args.sea_proxy_downsample,
                    )
                    if cache_state is not None:
                        cache_state.begin_batch(session_id=f"{args.run_id}:warmup:{warmup_index}")
                    if taylorseer_policy is not None:
                        taylorseer_policy.clear_batch()
                    if speca_policy is not None:
                        speca_policy.clear_batch()
                    if dicache_policy is not None and dicache_executor is not None:
                        sample_jit_dicache(
                            model,
                            warm_labels,
                            warm_noise,
                            warm_config,
                            executor=dicache_executor,
                            policy=dicache_policy,
                            mode=args.method,
                        )
                    else:
                        sample_jit(
                            model,
                            warm_labels,
                            warm_noise,
                            warm_config,
                            mode=args.method,
                            cache_state=cache_state,
                            dynamic_policy=dynamic_policy,
                        )
        if cache_state is not None:
            cache_state.clear()
            cache_state.reset_stats()
        if dynamic_policy is not None:
            dynamic_policy.reset()
        if safe_policy is not None:
            safe_policy.reset_runtime_state()
            safe_policy.reset_stats()
        if taylorseer_policy is not None:
            taylorseer_policy.reset_runtime_state()
            taylorseer_policy.reset_stats()
        if speca_policy is not None:
            speca_policy.reset_runtime_state()
            speca_policy.reset_stats()
        if dicache_policy is not None:
            dicache_policy.reset_runtime_state()
            dicache_policy.reset_stats()

    samples_for_npz = []
    labels_for_npz: list[int] = []
    generated = 0
    existing_images_skipped = len(reconciliation.complete_indices)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    debug_handle, dynamic_decision_writer = _dynamic_writer(args.dynamic_cache_debug_jsonl)
    try:
        for indices in _chunked(shard_indices, args.batch_size):
            with timing.measure("input_prepare_latency_sec"):
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
                cache_state.begin_batch(session_id=f"{args.run_id}:{indices[0]}")
            if taylorseer_policy is not None:
                taylorseer_policy.clear_batch()
            if speca_policy is not None:
                speca_policy.clear_batch()
            with timing.measure("sampling_latency_sec", device=device, synchronize=True):
                with torch.no_grad():
                    if dicache_policy is not None and dicache_executor is not None:
                        output, _records = sample_jit_dicache(
                            model,
                            batch_labels,
                            noise,
                            batch_config,
                            executor=dicache_executor,
                            policy=dicache_policy,
                            mode=args.method,
                        )
                    else:
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
            with timing.measure("postprocess_latency_sec"):
                output_cpu = output.detach().cpu()
            if args.save_png:
                with timing.measure("png_save_latency_sec"):
                    records = save_image_batch_png(output_cpu, batch_labels_list, indices, paths["image_dir"])
            else:
                records = [{"index": index, "label": int(label)} for index, label in zip(indices, batch_labels_list)]
            with timing.measure("manifest_latency_sec"):
                append_generation_manifest(paths["manifest"], records)
            if args.save_npz:
                samples_for_npz.append(output_cpu)
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
    if safe_policy is not None:
        cache_stats["safe_policy"] = safe_policy.summary()
        resolved["meta"]["safe_policy_summary"] = safe_policy.summary()
    if taylorseer_policy is not None:
        cache_stats["taylorseer_policy"] = taylorseer_policy.summary()
        resolved["meta"]["taylorseer_policy_summary"] = taylorseer_policy.summary()
    if speca_policy is not None:
        speca_summary = speca_policy.summary()
        runtime_summary = cache_state.summary() if cache_state is not None else {"enabled": False}
        cache_stats["runtime_cache"] = runtime_summary
        cache_stats["speca_policy"] = speca_summary
        cache_stats["verification_overhead_stats"] = speca_policy.verification_overhead_stats()
        resolved["meta"]["speca_policy_summary"] = speca_summary
    if dicache_policy is not None:
        dicache_summary = dicache_policy.summary()
        cache_stats["dicache_policy"] = dicache_summary
        resolved["meta"]["dicache_policy_summary"] = dicache_summary
    timing.generated_images_this_run = generated
    timing.existing_images_skipped = existing_images_skipped
    timing.total_images_available = count_images(paths["image_dir"])
    timing.peak_memory_allocated_bytes = peak_memory
    timing.end_to_end_latency_sec = time.perf_counter() - end_to_end_started
    latency_payload = {
        **timing.to_dict(),
        "generated_images": generated,
        "total_shard_images": len(all_shard_indices),
        "shard_mode": args.shard_mode,
    }
    write_generation_meta(paths["latency"], latency_payload)
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
                checkpoint_path=args.jit_ckpt_dir / "checkpoint-last.pth",
                hash_checkpoint=args.hash_checkpoints,
            ),
        }
    )
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
    parser.add_argument("--allow-partial-npz", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument("--clone-cache-on-store", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=0)
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
    parser.add_argument("--taylorseer-interval", type=int)
    parser.add_argument("--taylorseer-max-order", type=int)
    parser.add_argument("--taylorseer-refresh-first-n-steps", type=int, default=1)
    parser.add_argument("--taylorseer-refresh-last-n-steps", type=int, default=0)
    parser.add_argument("--taylorseer-debug-jsonl", type=Path)
    parser.add_argument("--taylorseer-clone-forecast", action="store_true")
    parser.add_argument("--taylorseer-min-history", type=int, default=2)
    parser.add_argument("--speca-max-order", type=int)
    parser.add_argument("--speca-first-full-steps", type=int)
    parser.add_argument("--speca-base-threshold", type=float)
    parser.add_argument("--speca-decay-rate", type=float)
    parser.add_argument("--speca-min-threshold", type=float)
    parser.add_argument("--speca-min-forecast-steps", type=int)
    parser.add_argument("--speca-max-forecast-steps", type=int)
    parser.add_argument(
        "--speca-error-metric",
        choices=("l1", "l2", "relative_l1", "relative_l2", "cosine_error"),
    )
    parser.add_argument("--speca-branch-aggregation", choices=("mean", "max"))
    parser.add_argument("--speca-verifier-module")
    parser.add_argument("--speca-min-history", type=int)
    parser.add_argument("--speca-debug-jsonl", type=Path)
    parser.add_argument("--speca-clone-forecast", action="store_true")
    parser.add_argument("--speca-eps", type=float, default=1e-10)
    parser.add_argument("--speca-max-error-samples", type=int, default=4096)
    parser.add_argument("--dicache-probe-depth", type=int)
    parser.add_argument("--dicache-reuse-threshold", type=float)
    parser.add_argument("--dicache-error-choice", choices=("delta_y", "delta_minus"))
    parser.add_argument("--dicache-branch-aggregation", choices=("mean", "max"))
    parser.add_argument("--dicache-ret-ratio", type=float)
    parser.add_argument(
        "--dicache-force-last-step-full",
        dest="dicache_force_last_step_full",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-dicache-force-last-step-full",
        dest="dicache_force_last_step_full",
        action="store_false",
    )
    parser.add_argument("--dicache-dcta", dest="dicache_dcta_enabled", action="store_true", default=None)
    parser.add_argument("--no-dicache-dcta", dest="dicache_dcta_enabled", action="store_false")
    parser.add_argument("--dicache-gamma-min", type=float)
    parser.add_argument("--dicache-gamma-max", type=float)
    parser.add_argument("--dicache-eps", type=float)
    parser.add_argument("--dicache-max-stat-samples", type=int)
    parser.add_argument("--dicache-debug-jsonl", type=Path)
    parser.add_argument("--dicache-clone-history", action="store_true")
    parser.add_argument("--dicache-force-full", action="store_true")
    parser.add_argument(
        "--dicache-share-cfg-prefix",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--dicache-schedule-variant",
        choices=("released_flux_compat",),
    )
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
    taylorseer_settings = _taylorseer_config(args, preset) if preset.method_type == "forecast_cache" else None
    speca_settings = _speca_config(args, preset) if preset.method_type == "speculative_cache" else None
    dicache_settings = _dicache_config(args, preset) if preset.method_type == "probe_cache" else None
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
            if preset.method_type in {"dynamic_cache", "forecast_cache", "speculative_cache"}
            else "jit_block_stack_residual"
            if preset.method_type == "probe_cache"
            else None
        ),
        "selected_modules": (
            []
            if preset.method_type == "probe_cache"
            else (preset.cache_preset or {}).get("cache_layers")
            if preset.cache_preset
            else None
        ),
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
        "jit_dir": str(args.jit_dir.resolve()),
        "jit_ckpt_dir": str(args.jit_ckpt_dir.resolve()),
        "checkpoint_exists": _checkpoint_ok(args.jit_ckpt_dir.resolve()),
        "device": args.device,
        "device_type": str(args.device).split(":", 1)[0],
        "dtype": "float32",
        "amp_enabled": False,
        "autocast_enabled": False,
        "compile_enabled": False,
        "cfg": args.cfg,
        "cfg_interval": [0.1, 1.0],
        "sampler": "euler",
        "solver": "euler",
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
        "taylorseer_cache": taylorseer_settings,
        "speca_cache": speca_settings,
        "dicache_cache": dicache_settings,
        **_pixbfc_static_meta(preset.method_type),
    }
    if speca_settings is not None:
        meta.update(speca_settings)
    if dicache_settings is not None:
        meta.update(dicache_settings)
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
    if args.taylorseer_interval is not None and args.taylorseer_interval <= 0:
        parser.error("--taylorseer-interval must be positive")
    if args.taylorseer_max_order is not None and args.taylorseer_max_order < 0:
        parser.error("--taylorseer-max-order must be non-negative")
    if args.taylorseer_min_history <= 0:
        parser.error("--taylorseer-min-history must be positive")
    if args.taylorseer_refresh_first_n_steps is not None and args.taylorseer_refresh_first_n_steps < 0:
        parser.error("--taylorseer-refresh-first-n-steps must be non-negative")
    if args.taylorseer_refresh_last_n_steps is not None and args.taylorseer_refresh_last_n_steps < 0:
        parser.error("--taylorseer-refresh-last-n-steps must be non-negative")
    if args.speca_max_order is not None and args.speca_max_order < 0:
        parser.error("--speca-max-order must be non-negative")
    if args.speca_first_full_steps is not None and args.speca_first_full_steps < 0:
        parser.error("--speca-first-full-steps must be non-negative")
    if args.speca_base_threshold is not None and args.speca_base_threshold <= 0.0:
        parser.error("--speca-base-threshold must be positive")
    if args.speca_decay_rate is not None and not 0.0 < args.speca_decay_rate <= 1.0:
        parser.error("--speca-decay-rate must satisfy 0 < value <= 1")
    if args.speca_min_threshold is not None and args.speca_min_threshold <= 0.0:
        parser.error("--speca-min-threshold must be positive")
    if args.speca_min_forecast_steps is not None and args.speca_min_forecast_steps <= 0:
        parser.error("--speca-min-forecast-steps must be positive")
    if args.speca_max_forecast_steps is not None and args.speca_max_forecast_steps <= 0:
        parser.error("--speca-max-forecast-steps must be positive")
    if args.speca_min_history is not None and args.speca_min_history <= 0:
        parser.error("--speca-min-history must be positive")
    if args.speca_eps <= 0.0:
        parser.error("--speca-eps must be positive")
    if args.speca_max_error_samples <= 0:
        parser.error("--speca-max-error-samples must be positive")
    if args.dicache_probe_depth is not None and args.dicache_probe_depth <= 0:
        parser.error("--dicache-probe-depth must be positive")
    if args.dicache_reuse_threshold is not None and args.dicache_reuse_threshold <= 0.0:
        parser.error("--dicache-reuse-threshold must be positive")
    if args.dicache_ret_ratio is not None and not 0.0 <= args.dicache_ret_ratio < 1.0:
        parser.error("--dicache-ret-ratio must satisfy 0 <= value < 1")
    if args.dicache_gamma_min is not None and args.dicache_gamma_min < 0.0:
        parser.error("--dicache-gamma-min must be non-negative")
    if args.dicache_eps is not None and args.dicache_eps <= 0.0:
        parser.error("--dicache-eps must be positive")
    if args.dicache_max_stat_samples is not None and args.dicache_max_stat_samples <= 0:
        parser.error("--dicache-max-stat-samples must be positive")
    resolved = resolve_config(args)
    speca_settings = resolved["meta"].get("speca_cache")
    if speca_settings is not None:
        if speca_settings["speca_min_threshold"] > speca_settings["speca_base_threshold"]:
            parser.error("--speca-min-threshold must not exceed --speca-base-threshold")
        if speca_settings["speca_max_forecast_steps"] < speca_settings["speca_min_forecast_steps"]:
            parser.error("--speca-max-forecast-steps must be >= --speca-min-forecast-steps")
    dicache_settings = resolved["meta"].get("dicache_cache")
    if dicache_settings is not None:
        total_blocks = dicache_settings.get("total_blocks")
        if total_blocks is not None and not 1 <= dicache_settings["probe_depth"] < total_blocks:
            parser.error("--dicache-probe-depth must be smaller than the JiT block count")
        if dicache_settings["gamma_min"] > dicache_settings["gamma_max"]:
            parser.error("--dicache-gamma-min must not exceed --dicache-gamma-max")
    if args.dry_run:
        _print_dry_run({"meta": resolved["meta"], "paths": resolved["paths"]})
        safe_density = resolved["meta"].get("safe_cache", {}).get("safe_map_density")
        if safe_density and safe_density.get("safe_total", 0) > 0 and safe_density.get("safe_true", 0) == 0:
            print("Warning: Safe map contains zero reusable positions; generation would degenerate to no-cache.")
        if not resolved["meta"]["checkpoint_exists"]:
            print(f"Missing JiT checkpoint: {args.jit_ckpt_dir / 'checkpoint-last.pth'}")
        return 0
    if args.resume and args.save_npz and not args.allow_partial_npz:
        raise ValueError(
            "NPZ resume is not supported because the in-memory tensor set may be incomplete."
        )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PFC_CUDA_DEVICES", "0"))
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
