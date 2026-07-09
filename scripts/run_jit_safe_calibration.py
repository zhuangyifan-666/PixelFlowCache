#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_run_id(seed: int, num_images: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_jit_safe_calib_seed{seed}_n{num_images}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _planned_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "calibration_meta": out_dir / "calibration_meta.json",
        "u_table": out_dir / "u_table.json",
        "safe_map_quality": out_dir / "safe_map_quality.json",
        "safe_map_speed": out_dir / "safe_map_speed.json",
        "calibration_summary_csv": out_dir / "calibration_summary.csv",
        "safe_map_summary_md": out_dir / "safe_map_summary.md",
    }


def _make_noise_for_indices(indices: list[int], seed: int, img_size: int, noise_scale: float, device: Any) -> Any:
    import torch

    chunks = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + index)
        chunks.append(noise_scale * torch.randn(1, 3, img_size, img_size, generator=generator, dtype=torch.float32))
    return torch.cat(chunks, dim=0).to(device)


def _batch_rel_l2(numerator: Any, denominator: Any, eps: float) -> Any:
    import torch

    num = torch.linalg.vector_norm(numerator.flatten(1), dim=1)
    den = torch.linalg.vector_norm(denominator.flatten(1), dim=1).clamp_min(eps)
    return num / den


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _velocity_from_xpred(x_pred: Any, x: Any, t: Any, t_eps: float) -> Any:
    return (x_pred - x) / (1.0 - t).clamp_min(t_eps)


def _prepare_controller(
    controller: Any | None,
    *,
    capture_store: dict[str, Any] | None = None,
    replay_store: dict[str, Any] | None = None,
) -> None:
    if controller is None:
        return
    if replay_store is not None:
        controller.replay_from(replay_store)
    elif capture_store is not None:
        controller.capture_into(capture_store)
    else:
        controller.disable()


def _cfg_velocity(
    *,
    model: Any,
    z: Any,
    t_scalar: Any,
    labels: Any,
    cfg_scale: float,
    controller: Any | None = None,
    capture_cond: dict[str, Any] | None = None,
    capture_uncond: dict[str, Any] | None = None,
    replay_cond: dict[str, Any] | None = None,
    replay_uncond: dict[str, Any] | None = None,
) -> Any:
    import torch

    t = t_scalar.expand(z.shape[0], 1, 1, 1)
    flat_t = t.flatten()
    try:
        _prepare_controller(controller, capture_store=capture_cond, replay_store=replay_cond)
        x_cond = model.net(z, flat_t, labels)
        v_cond = _velocity_from_xpred(x_cond, z, t, getattr(model, "t_eps", 5e-2))

        null_labels = torch.full_like(labels, model.num_classes)
        _prepare_controller(controller, capture_store=capture_uncond, replay_store=replay_uncond)
        x_uncond = model.net(z, flat_t, null_labels)
        v_uncond = _velocity_from_xpred(x_uncond, z, t, getattr(model, "t_eps", 5e-2))

        return v_uncond + cfg_scale * (v_cond - v_uncond)
    finally:
        if controller is not None:
            controller.disable()


def _build_boundary_groups(num_blocks: int, requested: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    block_names = [f"blocks.{idx}" for idx in range(num_blocks)]
    if "whole_backbone" in requested:
        groups["jit_whole_backbone"] = block_names
    if "thirds" in requested:
        split_a = max(1, num_blocks // 3)
        split_b = max(split_a, (2 * num_blocks) // 3)
        groups["early_blocks"] = block_names[:split_a]
        groups["middle_blocks"] = block_names[split_a:split_b]
        groups["late_blocks"] = block_names[split_b:]
    if not groups:
        raise ValueError("No supported boundary groups selected")
    return {name: modules for name, modules in groups.items() if modules}


def _module_to_boundary(boundary_groups: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for boundary, modules in boundary_groups.items():
        for module_name in modules:
            mapping.setdefault(module_name, boundary)
    return mapping


def _make_safe_map(
    *,
    map_name: str,
    lambda_value: float,
    map_max_age: int,
    ratios: dict[str, dict[int, dict[int, list[float]]]],
    boundary_groups: dict[str, list[str]],
    module_to_boundary: dict[str, str],
    args: argparse.Namespace,
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    safe_global: dict[str, dict[str, dict[str, bool]]] = {}
    u_global: dict[str, dict[str, dict[str, float | None]]] = {}
    rows: list[dict[str, Any]] = []
    for boundary in boundary_groups:
        safe_global[boundary] = {}
        u_global[boundary] = {}
        for step_idx in range(steps):
            safe_global[boundary][str(step_idx)] = {}
            u_global[boundary][str(step_idx)] = {}
            for age in range(1, map_max_age + 1):
                values = ratios.get(boundary, {}).get(step_idx, {}).get(age, [])
                u_value = _quantile(values, args.quantile)
                is_safe = bool(u_value is not None and u_value <= lambda_value)
                safe_global[boundary][str(step_idx)][str(age)] = is_safe
                u_global[boundary][str(step_idx)][str(age)] = u_value
                rows.append(
                    {
                        "map": map_name,
                        "boundary": boundary,
                        "step_idx": step_idx,
                        "age": age,
                        "num_values": len(values),
                        "u_ratio": "" if u_value is None else u_value,
                        "safe": int(is_safe),
                        "lambda": lambda_value,
                        "quantile": args.quantile,
                    }
                )

    safe_by_branch = {
        "global": safe_global,
        "cond": json.loads(json.dumps(safe_global)),
        "uncond": json.loads(json.dumps(safe_global)),
    }
    u_by_branch = {
        "global": u_global,
        "cond": json.loads(json.dumps(u_global)),
        "uncond": json.loads(json.dumps(u_global)),
    }
    safe_map = {
        "policy_name": "SafeMapCachePolicy",
        "model_name": "JiT",
        "model": args.jit_model,
        "steps": steps,
        "solver_stages": ["euler"],
        "branches": ["global", "cond", "uncond"],
        "branch_note": "cond/uncond tables are copied from the CFG-combined global calibration.",
        "boundary_groups": boundary_groups,
        "module_to_boundary": module_to_boundary,
        "max_age": map_max_age,
        "quantile": args.quantile,
        "lambda": lambda_value,
        "eps": args.eps,
        "lte_floor": args.lte_floor,
        "safe": {"euler": safe_by_branch},
        "u_ratio": {"euler": u_by_branch},
        "cache_age_candidates": list(range(1, map_max_age + 1)),
        "calibration_num_images": args.num_calibration_images,
        "seed": args.seed,
    }
    return safe_map, rows


def _write_outputs(
    *,
    paths: dict[str, Path],
    meta: dict[str, Any],
    u_table: dict[str, Any],
    quality_map: dict[str, Any],
    speed_map: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["calibration_meta"].write_text(json.dumps(_json_ready(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["u_table"].write_text(json.dumps(_json_ready(u_table), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["safe_map_quality"].write_text(
        json.dumps(_json_ready(quality_map), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["safe_map_speed"].write_text(
        json.dumps(_json_ready(speed_map), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["calibration_summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["map", "boundary", "step_idx", "age", "num_values", "u_ratio", "safe", "lambda", "quantile"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    quality_safe = sum(int(row["safe"]) for row in rows if row["map"] == "quality")
    speed_safe = sum(int(row["safe"]) for row in rows if row["map"] == "speed")
    paths["safe_map_summary_md"].write_text(
        "\n".join(
            [
                "# JiT Safe-BFC Calibration Summary",
                "",
                "This file describes calibration artifacts only; it is not a generation or FID result.",
                "",
                f"- calibration images: {meta['num_calibration_images']}",
                f"- steps: {meta['steps']}",
                f"- quantile: {meta['quantile']}",
                f"- quality lambda: {meta['quality_lambda']}",
                f"- speed lambda: {meta['speed_lambda']}",
                f"- quality safe cells: {quality_safe}",
                f"- speed safe cells: {speed_safe}",
                "",
            ]
        ),
        encoding="utf-8",
    )


class _BlockReplayController:
    def __init__(self, model: Any, module_names: list[str]) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.capture_store: dict[str, Any] | None = None
        self.replay_store: dict[str, Any] | None = None
        self._originals: dict[int, Any] = {}

    def __enter__(self) -> "_BlockReplayController":
        import torch.nn as nn

        blocks = self.model.net.blocks
        controller = self

        class WrappedBlock(nn.Module):
            def __init__(self, name: str, original: nn.Module) -> None:
                super().__init__()
                self.name = name
                self.original = original

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                replay_store = controller.replay_store
                if replay_store is not None and self.name in replay_store:
                    cached = replay_store[self.name]
                    first_tensor = next((arg for arg in args if hasattr(arg, "device") and hasattr(arg, "dtype")), None)
                    if first_tensor is not None:
                        cached = cached.to(device=first_tensor.device, dtype=first_tensor.dtype)
                    return cached
                output = self.original(*args, **kwargs)
                capture_store = controller.capture_store
                if capture_store is not None and self.name in controller.module_names and hasattr(output, "detach"):
                    capture_store[self.name] = output.detach().clone()
                return output

        for module_name in self.module_names:
            prefix, _, suffix = module_name.partition(".")
            if prefix != "blocks" or not suffix.isdigit():
                raise ValueError(f"Unsupported JiT module name for calibration: {module_name}")
            idx = int(suffix)
            if idx in self._originals:
                continue
            self._originals[idx] = blocks[idx]
            blocks[idx] = WrappedBlock(module_name, blocks[idx])
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        blocks = self.model.net.blocks
        for idx, original in self._originals.items():
            blocks[idx] = original
        self.disable()

    def capture_into(self, store: dict[str, Any]) -> None:
        self.capture_store = store
        self.replay_store = None

    def replay_from(self, store: dict[str, Any] | None) -> None:
        self.capture_store = None
        self.replay_store = store

    def disable(self) -> None:
        self.capture_store = None
        self.replay_store = None


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.eval.jit_runtime import JiTRuntimeConfig, cfg_enabled, load_jit_model
    from pfc.eval.label_schedule import make_imagenet_class_balanced_labels

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in this process")

    config = JiTRuntimeConfig(
        jit_dir=args.jit_dir.resolve(),
        ckpt_dir=args.jit_ckpt_dir.resolve(),
        run_id=args.run_id,
        run_dir=resolved["paths"]["out_dir"],
        preview_dir=resolved["paths"]["out_dir"] / "previews",
        model=args.jit_model,
        img_size=args.img_size,
        num_samples=args.batch_size,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        interval_min=0.1,
        interval_max=1.0,
        noise_scale=args.noise_scale,
        cache_interval=1,
        cache_layers="none",
        cache_branches="cond,uncond",
        active_t_min=None,
        active_t_max=None,
        warmup_runs=0,
        save_previews=False,
    )
    model = load_jit_model(config, device)
    model.eval()
    num_blocks = len(model.net.blocks)
    boundary_groups = _build_boundary_groups(num_blocks, _split_csv(args.boundary_groups))
    module_to_boundary = _module_to_boundary(boundary_groups)
    all_modules = sorted({module for modules in boundary_groups.values() for module in modules})
    labels_all = make_imagenet_class_balanced_labels(args.num_calibration_images)
    timesteps = torch.linspace(0.0, 1.0, args.steps + 1, device=device, dtype=torch.float32)
    ratios: dict[str, dict[int, dict[int, list[float]]]] = {
        boundary: defaultdict(lambda: defaultdict(list)) for boundary in boundary_groups
    }
    lte_values: dict[int, list[float]] = defaultdict(list)

    # Calibration algorithm:
    # 1. Run the fresh JiT step and capture boundary activations for cond/uncond forwards.
    # 2. Replay activations captured at step i-a for each candidate age a and boundary b.
    # 3. Convert JiT x-pred outputs to velocity before comparing Euler solver updates.
    # 4. Estimate Euler relative LTE by step doubling: one h step vs two h/2 steps.
    # 5. Compute R = E / max(LTE_rel, lte_floor), then quantile over calibration samples.
    with torch.no_grad(), _BlockReplayController(model, all_modules) as controller:
        for batch_start in range(0, args.num_calibration_images, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_calibration_images)
            indices = list(range(batch_start, batch_end))
            labels = torch.tensor(labels_all[batch_start:batch_end], device=device, dtype=torch.long)
            z = _make_noise_for_indices(indices, args.seed, args.img_size, args.noise_scale, device)
            history: dict[int, dict[str, dict[str, Any]]] = {}

            for step_idx in range(args.steps):
                t_scalar = timesteps[step_idx]
                t_next_scalar = timesteps[step_idx + 1]
                dt = t_next_scalar - t_scalar
                half_dt = dt * 0.5
                t_value = float(t_scalar.detach().cpu().item())
                cfg_scale = args.cfg if cfg_enabled(t_value, config.interval_min, config.interval_max) else 1.0

                capture_cond: dict[str, Any] = {}
                capture_uncond: dict[str, Any] = {}
                v_fresh = _cfg_velocity(
                    model=model,
                    z=z,
                    t_scalar=t_scalar,
                    labels=labels,
                    cfg_scale=cfg_scale,
                    controller=controller,
                    capture_cond=capture_cond,
                    capture_uncond=capture_uncond,
                )
                dx_fresh = dt * v_fresh
                x_h = z + dx_fresh

                v_half_1 = _cfg_velocity(model=model, z=z, t_scalar=t_scalar, labels=labels, cfg_scale=cfg_scale)
                z_mid = z + half_dt * v_half_1
                t_half = t_scalar + half_dt
                cfg_scale_half = args.cfg if cfg_enabled(float(t_half.detach().cpu().item()), config.interval_min, config.interval_max) else 1.0
                v_half_2 = _cfg_velocity(model=model, z=z_mid, t_scalar=t_half, labels=labels, cfg_scale=cfg_scale_half)
                x_h2 = z_mid + half_dt * v_half_2

                lte_rel = _batch_rel_l2(x_h2 - x_h, dx_fresh, args.eps)
                lte_safe = lte_rel.clamp_min(args.lte_floor)
                lte_values[step_idx].extend(float(value) for value in lte_rel.detach().cpu().tolist())

                for age in range(1, args.max_age + 1):
                    source = history.get(step_idx - age)
                    if source is None:
                        continue
                    for boundary, modules in boundary_groups.items():
                        replay_cond = {name: source["cond"][name] for name in modules if name in source["cond"]}
                        replay_uncond = {name: source["uncond"][name] for name in modules if name in source["uncond"]}
                        if not replay_cond and not replay_uncond:
                            continue
                        v_cached = _cfg_velocity(
                            model=model,
                            z=z,
                            t_scalar=t_scalar,
                            labels=labels,
                            cfg_scale=cfg_scale,
                            controller=controller,
                            replay_cond=replay_cond,
                            replay_uncond=replay_uncond,
                        )
                        dx_cached = dt * v_cached
                        e_rel = _batch_rel_l2(dx_cached - dx_fresh, dx_fresh, args.eps)
                        ratio = e_rel / lte_safe
                        ratios[boundary][step_idx][age].extend(float(value) for value in ratio.detach().cpu().tolist())

                history[step_idx] = {"cond": capture_cond, "uncond": capture_uncond}
                for old_step in list(history):
                    if old_step < step_idx - args.max_age:
                        del history[old_step]
                z = x_h.detach()

    quality_max_age = min(args.quality_max_age, args.max_age)
    speed_max_age = min(args.speed_max_age or args.max_age, args.max_age)
    quality_map, quality_rows = _make_safe_map(
        map_name="quality",
        lambda_value=args.quality_lambda,
        map_max_age=quality_max_age,
        ratios=ratios,
        boundary_groups=boundary_groups,
        module_to_boundary=module_to_boundary,
        args=args,
        steps=args.steps,
    )
    speed_map, speed_rows = _make_safe_map(
        map_name="speed",
        lambda_value=args.speed_lambda,
        map_max_age=speed_max_age,
        ratios=ratios,
        boundary_groups=boundary_groups,
        module_to_boundary=module_to_boundary,
        args=args,
        steps=args.steps,
    )
    u_table = {
        "steps": args.steps,
        "max_age": args.max_age,
        "quantile": args.quantile,
        "lte_rel_by_step": {
            str(step): {
                "num_values": len(values),
                "quantile": _quantile(values, args.quantile),
                "mean": sum(values) / len(values) if values else None,
            }
            for step, values in sorted(lte_values.items())
        },
    }
    _write_outputs(
        paths=resolved["paths"],
        meta=resolved["meta"],
        u_table=u_table,
        quality_map=quality_map,
        speed_map=speed_map,
        rows=quality_rows + speed_rows,
    )
    print(resolved["paths"]["out_dir"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate JiT Safe-BFC solver-perturbation safe maps.")
    parser.add_argument("--num-calibration-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--jit-dir", type=Path, default=ROOT / "third_party/JiT")
    parser.add_argument("--jit-ckpt-dir", type=Path, default=ROOT / "ckpts/JiT/JiT-B-16-256")
    parser.add_argument("--jit-model", default="JiT-B/16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--boundary-groups", default="whole_backbone")
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--quality-max-age", type=int, default=2)
    parser.add_argument("--speed-max-age", type=int)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--quality-lambda", type=float, default=0.5)
    parser.add_argument("--speed-lambda", type=float, default=1.0)
    parser.add_argument("--lte-floor", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or _default_run_id(args.seed, args.num_calibration_images)
    args.run_id = run_id
    out_dir = args.out_dir or ROOT / "calibrations/jit_safe" / run_id
    paths = _planned_paths(out_dir)
    quality_max_age = min(args.quality_max_age, args.max_age)
    speed_max_age = min(args.speed_max_age or args.max_age, args.max_age)
    meta = {
        "run_id": run_id,
        "policy_name": "SafeMapCachePolicy",
        "model_name": "JiT",
        "model": args.jit_model,
        "num_calibration_images": args.num_calibration_images,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "img_size": args.img_size,
        "noise_scale": args.noise_scale,
        "boundary_groups_arg": args.boundary_groups,
        "max_age": args.max_age,
        "quality_max_age": quality_max_age,
        "speed_max_age": speed_max_age,
        "quantile": args.quantile,
        "quality_lambda": args.quality_lambda,
        "speed_lambda": args.speed_lambda,
        "lte_floor": args.lte_floor,
        "eps": args.eps,
        "jit_dir": str(args.jit_dir.resolve()),
        "jit_ckpt_dir": str(args.jit_ckpt_dir.resolve()),
        "checkpoint_exists": (args.jit_ckpt_dir.resolve() / "checkpoint-last.pth").is_file(),
        "dry_run": args.dry_run,
        "algorithm": [
            "Run fresh JiT step to obtain CFG-combined Delta x_fresh.",
            "Replay boundary activations from i-a for each candidate cache age.",
            "Convert JiT x-pred to velocity before comparing solver updates.",
            "Estimate Euler relative LTE with step doubling.",
            "Use R = E / max(LTE_rel, lte_floor), then quantile over calibration samples.",
        ],
    }
    return {"meta": meta, "paths": paths}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_calibration_images <= 0:
        parser.error("--num-calibration-images must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.max_age <= 0:
        parser.error("--max-age must be positive")
    if args.quality_max_age <= 0:
        parser.error("--quality-max-age must be positive")
    if args.speed_max_age is not None and args.speed_max_age <= 0:
        parser.error("--speed-max-age must be positive")
    if not 0.0 < args.quantile <= 1.0:
        parser.error("--quantile must be in (0, 1]")
    if args.lte_floor <= 0.0:
        parser.error("--lte-floor must be positive")
    if args.eps <= 0.0:
        parser.error("--eps must be positive")
    resolved = resolve_config(args)
    if args.dry_run:
        payload = {
            "dry_run": True,
            "dry_run_note": "No checkpoint is loaded, no model is instantiated, and no sampling/calibration is run.",
            "meta": resolved["meta"],
            "expected_outputs": resolved["paths"],
        }
        print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return 0
    if not resolved["meta"]["checkpoint_exists"]:
        raise FileNotFoundError(f"Missing JiT checkpoint: {args.jit_ckpt_dir / 'checkpoint-last.pth'}")
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
