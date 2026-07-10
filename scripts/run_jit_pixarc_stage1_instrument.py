#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.label_schedule import ensure_label_schedule, make_imagenet_class_balanced_labels  # noqa: E402
from pfc.eval.provenance import collect_generation_provenance  # noqa: E402
from pfc.eval.sharding import compute_shard_indices  # noqa: E402
from pfc.eval.jit_runtime import cfg_enabled  # noqa: E402
from pfc.risk.frequency import RadialFrequencyRisk  # noqa: E402
from pfc.risk.io import (  # noqa: E402
    AtomicSampleWriter,
    config_signature,
    reconcile_sample_output,
    strict_json_dumps,
    write_json_atomic,
)
from pfc.risk.jit_counterfactual import (  # noqa: E402
    JiTFreshStepCapture,
    capture_fresh_step,
    evaluate_counterfactual_transition,
)
from pfc.risk.jit_history import BoundaryHistoryItem, JiTFreshBoundaryHistory  # noqa: E402
from pfc.risk.jit_plans import DEFAULT_JIT_PLAN_NAMES, JiTBoundaryPlan, resolve_jit_boundary_plans  # noqa: E402
from pfc.risk.metrics import (  # noqa: E402
    equivalence_metrics,
    l2_scalar,
    relative_l2_tensor,
    solver_scaled_rms_tensor,
    tensor_scalar,
    transition_relative_l2_tensor,
)
from pfc.risk.schema import (  # noqa: E402
    PIXARC_STAGE1_SCHEMA_VERSION,
    JiTCorrectnessRecord,
    JiTRiskRecord,
)
from pfc.risk.timing import DiagnosticActionTimer  # noqa: E402


DEFAULT_ACTIONS = (
    "fresh",
    "replay_age_0",
    "reuse_age_1",
    "reuse_age_2",
    "taylor_order_1",
)
COUNTERFACTUAL_ACTIONS = ("reuse_age_1", "reuse_age_2", "taylor_order_1")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_csv(value: str) -> list[int]:
    try:
        return [int(item) for item in _csv(value)]
    except ValueError as exc:
        raise ValueError(f"expected comma-separated integers, got {value!r}") from exc


def add_stage1_arguments(parser: argparse.ArgumentParser, *, parallel_defaults: bool = False) -> None:
    parser.add_argument("--jit-dir", type=Path, default=Path("third_party/JiT"))
    parser.add_argument("--jit-ckpt-dir", type=Path, default=Path("ckpts/JiT/JiT-B-16-256"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pixarc/stage1"))
    parser.add_argument("--num-images", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--cfg-interval-min", type=float, default=0.1)
    parser.add_argument("--cfg-interval-max", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--plans", default=",".join(DEFAULT_JIT_PLAN_NAMES))
    parser.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    parser.add_argument("--risk-atol", type=float, default=1e-3)
    parser.add_argument("--risk-rtol", type=float, default=1e-2)
    parser.add_argument("--frequency-low-ratio", type=float, default=0.15)
    parser.add_argument("--frequency-high-ratio", type=float, default=0.45)
    parser.add_argument("--equivalence-steps", default="0,1,25,49")
    parser.add_argument("--equivalence-atol", type=float, default=1e-6)
    parser.add_argument("--equivalence-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--measure-action-latency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-shards", type=int, default=4 if parallel_defaults else 1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-mode", choices=("strided", "contiguous"), default="strided")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-correctness", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--save-final-png", action="store_true", default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect JiT PixARC Stage-1 fresh-state counterfactual instrumentation."
    )
    add_stage1_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_stage1_config(args: argparse.Namespace, *, num_blocks: int = 12) -> dict[str, Any]:
    if args.batch_size != 1:
        raise ValueError("Stage-1 counterfactual instrumentation currently requires batch_size=1.")
    if args.num_images <= 0 or args.steps <= 0 or args.img_size <= 0:
        raise ValueError("num-images, steps, and img-size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid Stage-1 shard configuration")
    if not 0.0 <= args.cfg_interval_min < args.cfg_interval_max <= 1.0:
        raise ValueError("CFG interval must satisfy 0 <= min < max <= 1")
    plan_names = _csv(args.plans)
    plans = resolve_jit_boundary_plans(num_blocks, plan_names)
    requested_actions = _csv(args.actions)
    unknown_actions = sorted(set(requested_actions) - set(DEFAULT_ACTIONS))
    if unknown_actions:
        raise ValueError(f"unsupported Stage-1 actions: {unknown_actions}")
    if len(set(requested_actions)) != len(requested_actions):
        raise ValueError("Stage-1 actions must be unique")
    actions = ["fresh", "replay_age_0"] if args.correctness_only else requested_actions
    if "fresh" not in actions or "replay_age_0" not in actions:
        raise ValueError("Stage-1 requires fresh and replay_age_0 actions")
    equivalence_steps = _int_csv(args.equivalence_steps)
    if not equivalence_steps:
        raise ValueError("at least one fresh equivalence step is required")
    invalid_steps = [step for step in equivalence_steps if not 0 <= step < args.steps]
    if invalid_steps:
        raise ValueError(f"equivalence steps are outside [0,{args.steps}): {invalid_steps}")
    indices = compute_shard_indices(
        args.num_images, args.num_shards, args.shard_index, args.shard_mode
    )
    run_dir = args.output_root / "jit" / args.run_id
    semantic_config = {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "experiment": "jit_pixarc_stage1",
        "run_id": args.run_id,
        "model": "jit_b16_256",
        "jit_dir": str(args.jit_dir),
        "checkpoint_path": str(args.jit_ckpt_dir / "checkpoint-last.pth"),
        "num_images": args.num_images,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "cfg": args.cfg,
        "cfg_interval": [args.cfg_interval_min, args.cfg_interval_max],
        "img_size": args.img_size,
        "noise_scale": args.noise_scale,
        "plans": [plan.to_dict() for plan in plans],
        "actions": actions,
        "risk": {
            "atol": args.risk_atol,
            "rtol": args.risk_rtol,
            "frequency_low_ratio": args.frequency_low_ratio,
            "frequency_high_ratio": args.frequency_high_ratio,
        },
        "correctness": {
            "equivalence_steps": equivalence_steps,
            "atol": args.equivalence_atol,
            "rtol": args.equivalence_rtol,
            "strict": args.strict_correctness,
        },
        "runtime": {
            "measure_action_latency": bool(args.measure_action_latency and not args.correctness_only),
            "save_final_png": args.save_final_png,
            "torch_compile": False,
            "dtype": "float32",
            "fresh_state_counterfactual": True,
            "sequential_rollout": False,
        },
        "num_shards": args.num_shards,
        "shard_mode": args.shard_mode,
    }
    return {
        "semantic_config": semantic_config,
        "config_signature": config_signature(semantic_config),
        "plans": plans,
        "actions": actions,
        "equivalence_steps": equivalence_steps,
        "indices": indices,
        "run_dir": run_dir,
        "paths": {
            "run_dir": run_dir,
            "samples_dir": run_dir / "samples",
            "run_config": run_dir / "run_config.json",
            "labels": run_dir / "labels.json",
            "shard_dir": run_dir / f"shard_{args.shard_index}",
            "shard_meta": run_dir / f"shard_{args.shard_index}" / "shard_meta.json",
        },
    }


def _make_noise(global_index: int, seed: int, img_size: int, noise_scale: float, device: Any):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1_000_003 + global_index)
    return (
        noise_scale
        * torch.randn(1, 3, img_size, img_size, generator=generator, dtype=torch.float32)
    ).to(device)


def _correctness_record(
    *,
    args: argparse.Namespace,
    global_index: int,
    step_idx: int,
    branch: str,
    plan: str | None,
    check_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return JiTCorrectnessRecord(
        schema_version=PIXARC_STAGE1_SCHEMA_VERSION,
        record_type="jit_correctness",
        run_id=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        sample_id=global_index,
        global_index=global_index,
        step_idx=step_idx,
        branch=branch,
        boundary_plan=plan,
        check_name=check_name,
        max_abs=float(metrics["max_abs"]),
        mean_abs=float(metrics["mean_abs"]),
        relative_l2=float(metrics["relative_l2"]),
        allclose=bool(metrics["allclose"]),
        shape_match=bool(metrics["shape_match"]),
        dtype_match=bool(metrics["dtype_match"]),
        provenance_ref=f"shard_{args.shard_index}/shard_meta.json",
    ).to_dict()


def _risk_record(
    *,
    args: argparse.Namespace,
    global_index: int,
    class_label: int,
    fresh: JiTFreshStepCapture,
    cfg_active: bool,
    cfg_scale: float,
    plan: JiTBoundaryPlan | None,
    action: str,
    cache_age: int | None,
    ready: bool,
    skip_reason: str | None,
    risk: dict[str, float | None],
    latency_ms: float | None,
    signals: dict[str, Any],
) -> dict[str, Any]:
    return JiTRiskRecord(
        schema_version=PIXARC_STAGE1_SCHEMA_VERSION,
        record_type="jit_risk",
        run_id=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        sample_id=global_index,
        global_index=global_index,
        class_label=class_label,
        noise_seed=args.seed * 1_000_003 + global_index,
        step_idx=fresh.step_idx,
        t=fresh.t,
        t_next=fresh.t_next,
        dt=fresh.dt,
        solver_stage="euler",
        cfg_enabled=cfg_active,
        cfg_scale_effective=cfg_scale,
        boundary_plan=plan.name if plan else "fresh",
        boundary_start=plan.start_block if plan else None,
        boundary_end=plan.end_block if plan else None,
        action=action,
        cache_age=cache_age,
        action_ready=ready,
        skip_reason=skip_reason,
        risk_scaled_rms=risk.get("risk_scaled_rms"),
        risk_rel_l2=risk.get("risk_rel_l2"),
        risk_low=risk.get("risk_low"),
        risk_high=risk.get("risk_high"),
        velocity_rel_l2=risk.get("velocity_rel_l2"),
        cond_xpred_rel_l2=risk.get("cond_xpred_rel_l2"),
        uncond_xpred_rel_l2=risk.get("uncond_xpred_rel_l2"),
        diagnostic_action_latency_ms=latency_ms,
        signals=signals,
        provenance_ref=f"shard_{args.shard_index}/shard_meta.json",
    ).to_dict()


def _relative_difference(current: Any, previous: Any) -> float:
    return tensor_scalar(relative_l2_tensor(current - previous, previous))


def _signals(
    *,
    fresh: JiTFreshStepCapture,
    plan: JiTBoundaryPlan | None,
    cache_age: int | None,
    previous_state: Any | None,
    previous_fresh: JiTFreshStepCapture | None,
    selected_history: dict[str, BoundaryHistoryItem] | None,
    replacements: dict[str, Any] | None,
    frequency: RadialFrequencyRisk | None,
    t_eps: float,
    total_blocks: int,
) -> dict[str, Any]:
    signals: dict[str, Any] = {
        "step_idx": fresh.step_idx,
        "t": fresh.t,
        "t_next": fresh.t_next,
        "dt": fresh.dt,
        "cache_age": cache_age,
        "boundary_plan": plan.name if plan else "fresh",
        "boundary_start": plan.start_block if plan else None,
        "boundary_end": plan.end_block if plan else None,
        "skipped_block_count": plan.skipped_block_count if plan else 0,
        "prefix_block_count": plan.start_block if plan else 0,
        "suffix_block_count": total_blocks - plan.end_block if plan else 0,
        "xpred_amplification": 1.0 / max(1.0 - fresh.t, t_eps),
        "state_delta_rel_l2": None,
        "state_low_freq_delta": None,
        "state_high_freq_delta": None,
        "shallow_probe_delta_cond": None,
        "shallow_probe_delta_uncond": None,
        "shallow_probe_delta_aggregated": None,
        "cfg_velocity_gap": tensor_scalar(
            relative_l2_tensor(fresh.cond.velocity - fresh.uncond.velocity, fresh.cfg_velocity)
        ),
        "boundary_input_delta_cond": None,
        "boundary_input_delta_uncond": None,
        "boundary_output_delta_cond": None,
        "boundary_output_delta_uncond": None,
        "fresh_update_l2": l2_scalar(fresh.solver_update),
        "fresh_velocity_l2": l2_scalar(fresh.cfg_velocity),
        "candidate_shape_match": None,
        "candidate_dtype_match": None,
    }
    if previous_state is not None:
        signals["state_delta_rel_l2"] = _relative_difference(fresh.state, previous_state)
        if frequency is not None:
            low, _mid, high = frequency.split(fresh.state - previous_state)
            previous_low, _previous_mid, previous_high = frequency.split(previous_state)
            signals["state_low_freq_delta"] = tensor_scalar(relative_l2_tensor(low, previous_low))
            signals["state_high_freq_delta"] = tensor_scalar(relative_l2_tensor(high, previous_high))
    if previous_fresh is not None:
        cond_probe = _relative_difference(fresh.cond.shallow_probe, previous_fresh.cond.shallow_probe)
        uncond_probe = _relative_difference(fresh.uncond.shallow_probe, previous_fresh.uncond.shallow_probe)
        signals["shallow_probe_delta_cond"] = cond_probe
        signals["shallow_probe_delta_uncond"] = uncond_probe
        signals["shallow_probe_delta_aggregated"] = 0.5 * (cond_probe + uncond_probe)
    if plan is not None and replacements is not None:
        for branch, capture in (("cond", fresh.cond), ("uncond", fresh.uncond)):
            history_item = selected_history.get(branch) if selected_history is not None else None
            if history_item is not None:
                signals[f"boundary_input_delta_{branch}"] = _relative_difference(
                    capture.boundary_inputs[plan.name], history_item.boundary_input
                )
            signals[f"boundary_output_delta_{branch}"] = _relative_difference(
                capture.boundary_outputs[plan.name], replacements[branch]
            )
    return signals


def _risk_values(
    fresh: JiTFreshStepCapture,
    candidate: Any,
    frequency: RadialFrequencyRisk | None,
    args: argparse.Namespace,
) -> dict[str, float | None]:
    delta = candidate.next_state - fresh.next_state
    values: dict[str, float | None] = {
        "risk_scaled_rms": tensor_scalar(
            solver_scaled_rms_tensor(
                candidate.next_state,
                fresh.next_state,
                fresh.state,
                atol=args.risk_atol,
                rtol=args.risk_rtol,
            )
        ),
        "risk_rel_l2": tensor_scalar(
            transition_relative_l2_tensor(candidate.next_state, fresh.next_state, fresh.state)
        ),
        "velocity_rel_l2": tensor_scalar(
            relative_l2_tensor(candidate.cfg_velocity - fresh.cfg_velocity, fresh.cfg_velocity)
        ),
        "cond_xpred_rel_l2": tensor_scalar(
            relative_l2_tensor(candidate.cond.raw_output - fresh.cond.raw_output, fresh.cond.raw_output)
        ),
        "uncond_xpred_rel_l2": tensor_scalar(
            relative_l2_tensor(candidate.uncond.raw_output - fresh.uncond.raw_output, fresh.uncond.raw_output)
        ),
        "risk_low": None,
        "risk_high": None,
    }
    if frequency is not None:
        frequency_values = frequency.risks(delta, fresh.solver_update)
        values["risk_low"] = frequency_values["risk_low"]
        values["risk_high"] = frequency_values["risk_high"]
    return values


def _replacement_for_action(
    history: JiTFreshBoundaryHistory,
    *,
    global_index: int,
    plan: JiTBoundaryPlan,
    action: str,
    step_idx: int,
) -> tuple[dict[str, Any] | None, dict[str, BoundaryHistoryItem] | None, int | None]:
    replacements: dict[str, Any] = {}
    selected: dict[str, BoundaryHistoryItem] = {}
    if action in {"reuse_age_1", "reuse_age_2"}:
        age = 1 if action.endswith("1") else 2
        for branch in ("cond", "uncond"):
            item = history.select_age(
                sample_global_index=global_index,
                branch=branch,
                boundary_plan=plan.name,
                current_step_idx=step_idx,
                age=age,
            )
            if item is None:
                return None, None, age
            selected[branch] = item
            replacements[branch] = item.boundary_output
        return replacements, selected, age
    if action == "taylor_order_1":
        for branch in ("cond", "uncond"):
            pair = history.latest_two(
                sample_global_index=global_index,
                branch=branch,
                boundary_plan=plan.name,
                current_step_idx=step_idx,
            )
            predicted = history.taylor_order_1(
                sample_global_index=global_index,
                branch=branch,
                boundary_plan=plan.name,
                current_step_idx=step_idx,
            )
            if pair is None or predicted is None:
                return None, None, None
            selected[branch] = pair[-1]
            replacements[branch] = predicted
        return replacements, selected, None
    raise ValueError(f"unsupported counterfactual action: {action}")


def instrument_sample(
    model: Any,
    executor: Any,
    state: Any,
    *,
    global_index: int,
    class_label: int,
    plans: Sequence[JiTBoundaryPlan],
    actions: Sequence[str],
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    history = JiTFreshBoundaryHistory(max_history=3)
    risk_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    frequency = None if args.correctness_only else RadialFrequencyRisk(
        args.frequency_low_ratio, args.frequency_high_ratio
    )
    timesteps = torch.linspace(0.0, 1.0, args.steps + 1, device=state.device, dtype=state.dtype)
    cond_labels = torch.tensor([class_label], device=state.device, dtype=torch.long)
    uncond_labels = torch.full_like(cond_labels, model.num_classes)
    previous_state = None
    previous_fresh = None
    t_eps = float(getattr(model, "t_eps", 0.05))

    for step_idx in range(args.steps):
        t_scalar = timesteps[step_idx]
        t_next_scalar = timesteps[step_idx + 1]
        t_value = step_idx / args.steps
        t_next_value = (step_idx + 1) / args.steps
        dt_value = 1.0 / args.steps
        velocity_t = t_scalar.expand_as(state[:, :1, :1, :1])
        flat_t = velocity_t.flatten()
        cfg_active = cfg_enabled(t_value, args.cfg_interval_min, args.cfg_interval_max)
        cfg_scale = args.cfg if cfg_active else 1.0
        fresh_timer = DiagnosticActionTimer(state.device)
        if args.measure_action_latency and not args.correctness_only:
            fresh_timer.start()
        fresh = capture_fresh_step(
            executor,
            state,
            flat_t,
            velocity_t,
            cond_labels,
            uncond_labels,
            plans=plans,
            step_idx=step_idx,
            t=t_value,
            t_next=t_next_value,
            dt=dt_value,
            cfg_scale_effective=cfg_scale,
            t_eps=t_eps,
        )
        fresh_latency = (
            fresh_timer.stop() if args.measure_action_latency and not args.correctness_only else None
        )
        base_signals = _signals(
            fresh=fresh,
            plan=None,
            cache_age=None,
            previous_state=previous_state,
            previous_fresh=previous_fresh,
            selected_history=None,
            replacements=None,
            frequency=frequency,
            t_eps=t_eps,
            total_blocks=executor.total_blocks,
        )
        base_signals["candidate_shape_match"] = True
        base_signals["candidate_dtype_match"] = True
        risk_records.append(
            _risk_record(
                args=args,
                global_index=global_index,
                class_label=class_label,
                fresh=fresh,
                cfg_active=cfg_active,
                cfg_scale=cfg_scale,
                plan=None,
                action="fresh",
                cache_age=None,
                ready=True,
                skip_reason=None,
                risk={key: 0.0 for key in (
                    "risk_scaled_rms", "risk_rel_l2", "risk_low", "risk_high",
                    "velocity_rel_l2", "cond_xpred_rel_l2", "uncond_xpred_rel_l2",
                )},
                latency_ms=fresh_latency,
                signals=base_signals,
            )
        )

        if step_idx in args.resolved_equivalence_steps:
            for branch, labels, capture in (
                ("cond", cond_labels, fresh.cond),
                ("uncond", uncond_labels, fresh.uncond),
            ):
                direct = executor.net(state, flat_t, labels)
                metrics = equivalence_metrics(
                    capture.raw_output,
                    direct,
                    atol=args.equivalence_atol,
                    rtol=args.equivalence_rtol,
                )
                correctness_records.append(
                    _correctness_record(
                        args=args,
                        global_index=global_index,
                        step_idx=step_idx,
                        branch=branch,
                        plan=None,
                        check_name="fresh_split_equivalence",
                        metrics=metrics,
                    )
                )
                if args.strict_correctness and not metrics["allclose"]:
                    raise RuntimeError(
                        f"fresh split equivalence failed: sample={global_index} "
                        f"step={step_idx} branch={branch}: {metrics}"
                    )

        for plan in plans:
            replay_replacements = {
                "cond": fresh.cond.boundary_outputs[plan.name],
                "uncond": fresh.uncond.boundary_outputs[plan.name],
            }
            replay_timer = DiagnosticActionTimer(state.device)
            if args.measure_action_latency and not args.correctness_only:
                replay_timer.start()
            replay = evaluate_counterfactual_transition(
                executor,
                state,
                flat_t,
                velocity_t,
                cond_labels,
                uncond_labels,
                plan=plan,
                action="replay_age_0",
                replacements=replay_replacements,
                cfg_scale_effective=cfg_scale,
                dt=dt_value,
                t_eps=t_eps,
            )
            replay_latency = (
                replay_timer.stop()
                if args.measure_action_latency and not args.correctness_only
                else None
            )
            replay_metrics = equivalence_metrics(
                replay.next_state,
                fresh.next_state,
                atol=args.equivalence_atol,
                rtol=args.equivalence_rtol,
            )
            correctness_records.append(
                _correctness_record(
                    args=args,
                    global_index=global_index,
                    step_idx=step_idx,
                    branch="cfg",
                    plan=plan.name,
                    check_name="replay_age_0",
                    metrics=replay_metrics,
                )
            )
            if args.strict_correctness and not replay_metrics["allclose"]:
                raise RuntimeError(
                    f"age-0 replay failed: sample={global_index} step={step_idx} "
                    f"plan={plan.name}: {replay_metrics}"
                )
            if "replay_age_0" in actions:
                replay_signals = _signals(
                    fresh=fresh,
                    plan=plan,
                    cache_age=0,
                    previous_state=previous_state,
                    previous_fresh=previous_fresh,
                    selected_history=None,
                    replacements=replay_replacements,
                    frequency=frequency,
                    t_eps=t_eps,
                    total_blocks=executor.total_blocks,
                )
                replay_signals["candidate_shape_match"] = (
                    tuple(replay.next_state.shape) == tuple(fresh.next_state.shape)
                )
                replay_signals["candidate_dtype_match"] = (
                    replay.next_state.dtype == fresh.next_state.dtype
                )
                risk_records.append(
                    _risk_record(
                        args=args,
                        global_index=global_index,
                        class_label=class_label,
                        fresh=fresh,
                        cfg_active=cfg_active,
                        cfg_scale=cfg_scale,
                        plan=plan,
                        action="replay_age_0",
                        cache_age=0,
                        ready=True,
                        skip_reason=None,
                        risk=_risk_values(fresh, replay, frequency, args),
                        latency_ms=replay_latency,
                        signals=replay_signals,
                    )
                )

            for action in [item for item in actions if item in COUNTERFACTUAL_ACTIONS]:
                replacements, selected, cache_age = _replacement_for_action(
                    history,
                    global_index=global_index,
                    plan=plan,
                    action=action,
                    step_idx=step_idx,
                )
                if replacements is None:
                    unavailable_signals = _signals(
                        fresh=fresh,
                        plan=plan,
                        cache_age=cache_age,
                        previous_state=previous_state,
                        previous_fresh=previous_fresh,
                        selected_history=None,
                        replacements=None,
                        frequency=frequency,
                        t_eps=t_eps,
                        total_blocks=executor.total_blocks,
                    )
                    risk_records.append(
                        _risk_record(
                            args=args,
                            global_index=global_index,
                            class_label=class_label,
                            fresh=fresh,
                            cfg_active=cfg_active,
                            cfg_scale=cfg_scale,
                            plan=plan,
                            action=action,
                            cache_age=cache_age,
                            ready=False,
                            skip_reason="insufficient_history",
                            risk={key: None for key in (
                                "risk_scaled_rms", "risk_rel_l2", "risk_low", "risk_high",
                                "velocity_rel_l2", "cond_xpred_rel_l2", "uncond_xpred_rel_l2",
                            )},
                            latency_ms=None,
                            signals=unavailable_signals,
                        )
                    )
                    continue
                timer = DiagnosticActionTimer(state.device)
                if args.measure_action_latency:
                    timer.start()
                candidate = evaluate_counterfactual_transition(
                    executor,
                    state,
                    flat_t,
                    velocity_t,
                    cond_labels,
                    uncond_labels,
                    plan=plan,
                    action=action,
                    replacements=replacements,
                    cfg_scale_effective=cfg_scale,
                    dt=dt_value,
                    t_eps=t_eps,
                )
                latency = timer.stop() if args.measure_action_latency else None
                candidate_signals = _signals(
                    fresh=fresh,
                    plan=plan,
                    cache_age=cache_age,
                    previous_state=previous_state,
                    previous_fresh=previous_fresh,
                    selected_history=selected,
                    replacements=replacements,
                    frequency=frequency,
                    t_eps=t_eps,
                    total_blocks=executor.total_blocks,
                )
                candidate_signals["candidate_shape_match"] = (
                    tuple(candidate.next_state.shape) == tuple(fresh.next_state.shape)
                )
                candidate_signals["candidate_dtype_match"] = (
                    candidate.next_state.dtype == fresh.next_state.dtype
                )
                risk_records.append(
                    _risk_record(
                        args=args,
                        global_index=global_index,
                        class_label=class_label,
                        fresh=fresh,
                        cfg_active=cfg_active,
                        cfg_scale=cfg_scale,
                        plan=plan,
                        action=action,
                        cache_age=cache_age,
                        ready=True,
                        skip_reason=None,
                        risk=_risk_values(fresh, candidate, frequency, args),
                        latency_ms=latency,
                        signals=candidate_signals,
                    )
                )

        for plan in plans:
            for branch, capture in (("cond", fresh.cond), ("uncond", fresh.uncond)):
                history.append(
                    sample_global_index=global_index,
                    branch=branch,
                    boundary_plan=plan.name,
                    step_idx=step_idx,
                    t=t_value,
                    boundary_input=capture.boundary_inputs[plan.name],
                    boundary_output=capture.boundary_outputs[plan.name],
                )
        previous_state = state.detach().clone()
        previous_fresh = fresh
        state = fresh.next_state.detach().clone()

    history.clear_sample(global_index)
    summary = {
        "risk_record_count": len(risk_records),
        "correctness_record_count": len(correctness_records),
        "future_leakage_count": 0,
        "history_items_after_clear": history.item_count(),
        "final_state_shape": list(state.shape),
        "final_state_dtype": str(state.dtype),
    }
    return state, risk_records, correctness_records, summary


def _run_real(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    import torch

    from pfc.eval.generation_io import save_image_batch_png
    from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor
    from pfc.eval.jit_runtime import JiTRuntimeConfig, load_jit_model

    if not torch.cuda.is_available():
        raise RuntimeError("JiT PixARC Stage-1 requires CUDA for non-dry-run execution")
    checkpoint = args.jit_ckpt_dir / "checkpoint-last.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing JiT checkpoint: {checkpoint}")
    device = torch.device("cuda")
    runtime_config = JiTRuntimeConfig(
        jit_dir=args.jit_dir.resolve(),
        ckpt_dir=args.jit_ckpt_dir.resolve(),
        run_id=args.run_id,
        run_dir=resolved["run_dir"],
        preview_dir=resolved["run_dir"] / "previews",
        img_size=args.img_size,
        num_samples=1,
        batch_size=1,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        interval_min=args.cfg_interval_min,
        interval_max=args.cfg_interval_max,
        noise_scale=args.noise_scale,
    )
    model = load_jit_model(runtime_config, device)
    model.eval()
    executor = JiTDiCacheExecutor(model.net)
    resolved = resolve_stage1_config(args, num_blocks=executor.total_blocks)
    args.resolved_equivalence_steps = set(resolved["equivalence_steps"])
    run_dir = resolved["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_config = resolved["paths"]["run_config"]
    if existing_config.is_file():
        payload = json.loads(existing_config.read_text(encoding="utf-8"))
        if payload.get("config_signature") != resolved["config_signature"]:
            raise ValueError("existing Stage-1 run uses a different configuration")
    else:
        write_json_atomic(
            existing_config,
            {
                **resolved["semantic_config"],
                "config_signature": resolved["config_signature"],
                "latency_semantics": DiagnosticActionTimer.latency_semantics,
                "not_comparable_to_end_to_end_generation": True,
            },
        )
    labels = make_imagenet_class_balanced_labels(args.num_images)
    ensure_label_schedule(labels, run_dir)
    provenance = collect_generation_provenance(
        ROOT,
        checkpoint_path=checkpoint,
        hash_checkpoint=args.hash_checkpoint,
    )
    skipped: list[int] = []
    completed: list[int] = []
    with torch.inference_mode():
        for global_index in resolved["indices"]:
            disposition = reconcile_sample_output(
                run_dir,
                global_index,
                resolved["config_signature"],
                resume=args.resume,
            )
            if disposition == "skip":
                skipped.append(global_index)
                continue
            state = _make_noise(
                global_index, args.seed, args.img_size, args.noise_scale, device
            )
            final_state, risk_records, correctness_records, summary = instrument_sample(
                model,
                executor,
                state,
                global_index=global_index,
                class_label=int(labels[global_index]),
                plans=resolved["plans"],
                actions=resolved["actions"],
                args=args,
            )
            with AtomicSampleWriter(
                run_dir, global_index, resolved["config_signature"]
            ) as writer:
                if args.save_final_png:
                    save_image_batch_png(
                        final_state,
                        [labels[global_index]],
                        [global_index],
                        writer.temporary,
                    )
                writer.commit(
                    risk_records=risk_records,
                    correctness_records=correctness_records,
                    sample_summary={
                        **summary,
                        "class_label": int(labels[global_index]),
                        "noise_seed": args.seed * 1_000_003 + global_index,
                    },
                )
            completed.append(global_index)
    shard_dir = resolved["paths"]["shard_dir"]
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        resolved["paths"]["shard_meta"],
        {
            "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
            "run_id": args.run_id,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "shard_mode": args.shard_mode,
            "assigned_indices": resolved["indices"],
            "completed_indices": completed,
            "skipped_indices": skipped,
            "config_signature": resolved["config_signature"],
            "provenance": provenance,
            "parallel_wall_time_is_algorithm_speedup": False,
        },
    )
    print(run_dir)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    resolved = resolve_stage1_config(args)
    args.resolved_equivalence_steps = set(resolved["equivalence_steps"])
    if args.dry_run:
        print(
            strict_json_dumps(
                {
                    "dry_run": True,
                    "resolved_config": resolved["semantic_config"],
                    "config_signature": resolved["config_signature"],
                    "indices": resolved["indices"],
                    "paths": {key: str(value) for key, value in resolved["paths"].items()},
                    "checkpoint_checked": False,
                    "model_loaded": False,
                    "cuda_used": False,
                },
                indent=2,
            )
        )
        return 0
    return _run_real(args, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
