from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pfc.risk.io import AtomicSampleWriter, config_signature, write_json_atomic
from pfc.risk.schema import PIXARC_STAGE1_SCHEMA_VERSION


DEFAULT_ACTIONS = (
    "fresh",
    "replay_age_0",
    "reuse_age_1",
    "reuse_age_2",
    "taylor_order_1",
)


def make_stage1_run(
    root: Path,
    *,
    num_images: int = 2,
    steps: int = 3,
    plans: Iterable[str] = ("early", "late"),
    actions: Iterable[str] = DEFAULT_ACTIONS,
    sample_indices: Iterable[int] | None = None,
    positive_reuse: bool = True,
) -> Path:
    plan_names = list(plans)
    action_names = list(actions)
    semantic_config: dict[str, Any] = {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "experiment": "jit_pixarc_stage1",
        "run_id": "synthetic",
        "model": "jit_b16_256",
        "num_images": num_images,
        "steps": steps,
        "batch_size": 1,
        "seed": 21,
        "plans": [
            {"name": name, "start_block": index, "end_block": index + 1, "description": name}
            for index, name in enumerate(plan_names)
        ],
        "actions": action_names,
        "correctness": {
            "equivalence_steps": [0],
            "atol": 1e-6,
            "rtol": 1e-5,
            "strict": True,
        },
    }
    signature = config_signature(semantic_config)
    write_json_atomic(root / "run_config.json", {**semantic_config, "config_signature": signature})
    indices = range(num_images) if sample_indices is None else sample_indices
    for global_index in indices:
        risk_records: list[dict[str, Any]] = []
        correctness_records: list[dict[str, Any]] = []
        for step_idx in range(steps):
            risk_records.append(_risk_row(global_index, step_idx, "fresh", "fresh", True, 0.0))
            if step_idx == 0:
                for branch in ("cond", "uncond"):
                    correctness_records.append(
                        _correctness_row(
                            global_index,
                            step_idx,
                            None,
                            "fresh_split_equivalence",
                            branch=branch,
                        )
                    )
            for plan in plan_names:
                correctness_records.append(
                    _correctness_row(global_index, step_idx, plan, "replay_age_0")
                )
                if "replay_age_0" in action_names:
                    risk_records.append(
                        _risk_row(global_index, step_idx, plan, "replay_age_0", True, 0.0)
                    )
                for action in ("reuse_age_1", "reuse_age_2", "taylor_order_1"):
                    if action not in action_names:
                        continue
                    ready_after = 1 if action == "reuse_age_1" else 2
                    ready = step_idx >= ready_after
                    value = 1.0 + step_idx if ready and positive_reuse and action.startswith("reuse") else 0.0
                    risk_records.append(
                        _risk_row(global_index, step_idx, plan, action, ready, value)
                    )
        with AtomicSampleWriter(root, global_index, signature) as writer:
            writer.commit(
                risk_records=list(reversed(risk_records)),
                correctness_records=list(reversed(correctness_records)),
                sample_summary={
                    "risk_record_count": len(risk_records),
                    "correctness_record_count": len(correctness_records),
                    "future_leakage_count": 0,
                    "history_items_after_clear": 0,
                    "final_state_shape": [1, 2, 2, 2],
                    "final_state_dtype": "torch.float32",
                },
            )
    return root


def _risk_row(
    global_index: int,
    step_idx: int,
    plan: str,
    action: str,
    ready: bool,
    risk: float,
) -> dict[str, Any]:
    return {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "record_type": "jit_risk",
        "run_id": "synthetic",
        "shard_index": global_index % 2,
        "num_shards": 2,
        "sample_id": global_index,
        "global_index": global_index,
        "class_label": global_index,
        "noise_seed": 21_000_063 + global_index,
        "step_idx": step_idx,
        "t": step_idx / 3,
        "t_next": (step_idx + 1) / 3,
        "dt": 1 / 3,
        "solver_stage": "euler",
        "cfg_enabled": True,
        "cfg_scale_effective": 3.0,
        "boundary_plan": plan,
        "boundary_start": None if action == "fresh" else 0,
        "boundary_end": None if action == "fresh" else 1,
        "action": action,
        "cache_age": {"replay_age_0": 0, "reuse_age_1": 1, "reuse_age_2": 2}.get(action),
        "action_ready": ready,
        "skip_reason": None if ready else "insufficient_history",
        "risk_scaled_rms": risk if ready else None,
        "risk_rel_l2": risk / 10 if ready else None,
        "risk_low": risk / 20 if ready else None,
        "risk_high": risk / 5 if ready else None,
        "velocity_rel_l2": risk / 10 if ready else None,
        "cond_xpred_rel_l2": risk / 10 if ready else None,
        "uncond_xpred_rel_l2": risk / 10 if ready else None,
        "diagnostic_action_latency_ms": 1.0 + step_idx if ready else None,
        "signals": {
            "candidate_shape_match": True if ready else None,
            "candidate_dtype_match": True if ready else None,
            "state_delta_rel_l2": None if step_idx == 0 else 0.1,
        },
        "provenance_ref": f"shard_{global_index % 2}/shard_meta.json",
    }


def _correctness_row(
    global_index: int,
    step_idx: int,
    plan: str | None,
    check_name: str,
    *,
    branch: str = "cfg",
) -> dict[str, Any]:
    return {
        "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
        "record_type": "jit_correctness",
        "run_id": "synthetic",
        "shard_index": global_index % 2,
        "num_shards": 2,
        "sample_id": global_index,
        "global_index": global_index,
        "step_idx": step_idx,
        "branch": branch,
        "boundary_plan": plan,
        "check_name": check_name,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "relative_l2": 0.0,
        "allclose": True,
        "shape_match": True,
        "dtype_match": True,
        "future_leakage_count": 0,
        "details": {},
        "provenance_ref": f"shard_{global_index % 2}/shard_meta.json",
    }
