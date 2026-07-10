from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from pfc.eval.jit_dicache_runtime import JiTDiCacheExecutor
from pfc.eval.jit_runtime import combine_cfg_velocity, xpred_to_velocity
from pfc.risk.jit_plans import JiTBoundaryPlan


@dataclass
class JiTBranchFreshCapture:
    branch: str
    raw_output: torch.Tensor
    velocity: torch.Tensor
    boundary_outputs: dict[str, torch.Tensor]
    boundary_inputs: dict[str, torch.Tensor]
    shallow_probe: torch.Tensor
    condition: torch.Tensor
    num_image_tokens: int


@dataclass
class JiTFreshStepCapture:
    step_idx: int
    t: float
    t_next: float
    dt: float
    state: torch.Tensor
    cond: JiTBranchFreshCapture
    uncond: JiTBranchFreshCapture
    cfg_velocity: torch.Tensor
    solver_update: torch.Tensor
    next_state: torch.Tensor


@dataclass
class JiTBranchCandidate:
    branch: str
    raw_output: torch.Tensor
    velocity: torch.Tensor
    condition: torch.Tensor


@dataclass
class JiTCounterfactualTransition:
    plan: JiTBoundaryPlan
    action: str
    cond: JiTBranchCandidate
    uncond: JiTBranchCandidate
    cfg_velocity: torch.Tensor
    solver_update: torch.Tensor
    next_state: torch.Tensor


def capture_fresh_branch(
    executor: JiTDiCacheExecutor,
    state: torch.Tensor,
    flat_t: torch.Tensor,
    velocity_t: torch.Tensor,
    labels: torch.Tensor,
    *,
    branch: str,
    plans: Sequence[JiTBoundaryPlan],
    t_eps: float,
) -> JiTBranchFreshCapture:
    for plan in plans:
        plan.validate(executor.total_blocks)
    hidden, t_embedding, num_image_tokens = executor.prepare_common_input(state, flat_t)
    label_embedding, condition = executor.prepare_branch_condition(t_embedding, labels)
    starts: dict[int, list[str]] = {}
    ends: dict[int, list[str]] = {}
    for plan in plans:
        starts.setdefault(plan.start_block, []).append(plan.name)
        ends.setdefault(plan.end_block, []).append(plan.name)
    boundary_inputs: dict[str, torch.Tensor] = {}
    boundary_outputs: dict[str, torch.Tensor] = {}
    shallow_probe: torch.Tensor | None = None

    for index in range(executor.total_blocks):
        for name in starts.get(index, ()):
            boundary_inputs[name] = hidden.detach().clone()
        hidden = executor.run_blocks_range(
            hidden,
            condition,
            label_embedding,
            start=index,
            end=index + 1,
            num_image_tokens=num_image_tokens,
        )
        if index == 0:
            shallow_probe = executor.extract_image_tokens(hidden, num_image_tokens).detach().clone()
        for name in ends.get(index + 1, ()):
            boundary_outputs[name] = hidden.detach().clone()

    missing_inputs = sorted(set(plan.name for plan in plans) - set(boundary_inputs))
    missing_outputs = sorted(set(plan.name for plan in plans) - set(boundary_outputs))
    if missing_inputs or missing_outputs:
        raise RuntimeError(
            f"fresh boundary capture is incomplete: inputs={missing_inputs}, outputs={missing_outputs}"
        )
    if shallow_probe is None:
        raise RuntimeError("JiT fresh capture did not execute block 0")
    raw_output = executor.finalize_output(
        executor.extract_image_tokens(hidden, num_image_tokens),
        condition,
    )
    velocity = xpred_to_velocity(raw_output, state, velocity_t, t_eps)
    return JiTBranchFreshCapture(
        branch=str(branch),
        raw_output=raw_output.detach().clone(),
        velocity=velocity.detach().clone(),
        boundary_outputs=boundary_outputs,
        boundary_inputs=boundary_inputs,
        shallow_probe=shallow_probe,
        condition=condition.detach().clone(),
        num_image_tokens=num_image_tokens,
    )


def capture_fresh_step(
    executor: JiTDiCacheExecutor,
    state: torch.Tensor,
    flat_t: torch.Tensor,
    velocity_t: torch.Tensor,
    cond_labels: torch.Tensor,
    uncond_labels: torch.Tensor,
    *,
    plans: Sequence[JiTBoundaryPlan],
    step_idx: int,
    t: float,
    t_next: float,
    dt: float,
    cfg_scale_effective: float,
    t_eps: float,
) -> JiTFreshStepCapture:
    cond = capture_fresh_branch(
        executor,
        state,
        flat_t,
        velocity_t,
        cond_labels,
        branch="cond",
        plans=plans,
        t_eps=t_eps,
    )
    uncond = capture_fresh_branch(
        executor,
        state,
        flat_t,
        velocity_t,
        uncond_labels,
        branch="uncond",
        plans=plans,
        t_eps=t_eps,
    )
    cfg_velocity = combine_cfg_velocity(cond.velocity, uncond.velocity, cfg_scale_effective)
    solver_update = float(dt) * cfg_velocity
    next_state = state + solver_update
    return JiTFreshStepCapture(
        step_idx=int(step_idx),
        t=float(t),
        t_next=float(t_next),
        dt=float(dt),
        state=state.detach().clone(),
        cond=cond,
        uncond=uncond,
        cfg_velocity=cfg_velocity.detach().clone(),
        solver_update=solver_update.detach().clone(),
        next_state=next_state.detach().clone(),
    )


def evaluate_branch_candidate(
    executor: JiTDiCacheExecutor,
    state: torch.Tensor,
    flat_t: torch.Tensor,
    velocity_t: torch.Tensor,
    labels: torch.Tensor,
    plan: JiTBoundaryPlan,
    replacement_hidden: torch.Tensor,
    *,
    branch: str,
    t_eps: float,
) -> JiTBranchCandidate:
    plan.validate(executor.total_blocks)
    prefix, t_embedding, num_image_tokens = executor.prepare_common_input(state, flat_t)
    label_embedding, condition = executor.prepare_branch_condition(t_embedding, labels)
    if plan.start_block > 0:
        prefix = executor.run_blocks_range(
            prefix,
            condition,
            label_embedding,
            start=0,
            end=plan.start_block,
            num_image_tokens=num_image_tokens,
        )
    del prefix
    _validate_replacement_shape(executor, replacement_hidden, plan, num_image_tokens)
    hidden = replacement_hidden.detach().clone()
    if plan.end_block < executor.total_blocks:
        hidden = executor.run_blocks_range(
            hidden,
            condition,
            label_embedding,
            start=plan.end_block,
            end=executor.total_blocks,
            num_image_tokens=num_image_tokens,
        )
    raw_output = executor.finalize_output(
        executor.extract_image_tokens(hidden, num_image_tokens),
        condition,
    )
    velocity = xpred_to_velocity(raw_output, state, velocity_t, t_eps)
    return JiTBranchCandidate(
        branch=str(branch),
        raw_output=raw_output.detach().clone(),
        velocity=velocity.detach().clone(),
        condition=condition.detach().clone(),
    )


def evaluate_counterfactual_transition(
    executor: JiTDiCacheExecutor,
    state: torch.Tensor,
    flat_t: torch.Tensor,
    velocity_t: torch.Tensor,
    cond_labels: torch.Tensor,
    uncond_labels: torch.Tensor,
    *,
    plan: JiTBoundaryPlan,
    action: str,
    replacements: Mapping[str, torch.Tensor],
    cfg_scale_effective: float,
    dt: float,
    t_eps: float,
) -> JiTCounterfactualTransition:
    if set(replacements) != {"cond", "uncond"}:
        raise ValueError("counterfactual replacements must contain cond and uncond")
    state_snapshot = state.detach().clone()
    cond = evaluate_branch_candidate(
        executor,
        state,
        flat_t,
        velocity_t,
        cond_labels,
        plan,
        replacements["cond"],
        branch="cond",
        t_eps=t_eps,
    )
    uncond = evaluate_branch_candidate(
        executor,
        state,
        flat_t,
        velocity_t,
        uncond_labels,
        plan,
        replacements["uncond"],
        branch="uncond",
        t_eps=t_eps,
    )
    cfg_velocity = combine_cfg_velocity(cond.velocity, uncond.velocity, cfg_scale_effective)
    solver_update = float(dt) * cfg_velocity
    next_state = state + solver_update
    if not torch.equal(state, state_snapshot):
        raise AssertionError("counterfactual evaluation modified the fresh state")
    return JiTCounterfactualTransition(
        plan=plan,
        action=str(action),
        cond=cond,
        uncond=uncond,
        cfg_velocity=cfg_velocity.detach().clone(),
        solver_update=solver_update.detach().clone(),
        next_state=next_state.detach().clone(),
    )


def _validate_replacement_shape(
    executor: JiTDiCacheExecutor,
    hidden: torch.Tensor,
    plan: JiTBoundaryPlan,
    num_image_tokens: int,
) -> None:
    context_expected = (
        int(executor.net.in_context_len) > 0
        and plan.end_block > int(executor.net.in_context_start)
    )
    expected_tokens = num_image_tokens + (
        int(executor.net.in_context_len) if context_expected else 0
    )
    if hidden.ndim != 3 or int(hidden.shape[1]) != expected_tokens:
        raise ValueError(
            f"replacement for plan {plan.name} has invalid shape {tuple(hidden.shape)}; "
            f"expected [B,{expected_tokens},C]"
        )
