#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _modules(args: argparse.Namespace) -> list[str]:
    if args.modules:
        return _split_csv(args.modules)
    if args.jit_blocks is None:
        raise ValueError("Pass --modules or --jit-blocks")
    return [f"blocks.{idx}" for idx in range(args.jit_blocks)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a forced-true Safe-BFC smoke-test safe map.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-age", type=int, default=1)
    parser.add_argument("--modules")
    parser.add_argument("--jit-blocks", type=int)
    parser.add_argument("--boundary-name", default="jit_whole_backbone")
    parser.add_argument("--branches", default="global,cond,uncond")
    parser.add_argument("--solver-stages", default="euler")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.max_age <= 0:
        raise ValueError("--max-age must be positive")
    modules = _modules(args)
    branches = _split_csv(args.branches)
    stages = _split_csv(args.solver_stages)
    safe_by_stage: dict[str, dict[str, dict[str, dict[str, dict[str, bool]]]]] = {}
    u_by_stage: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for stage in stages:
        safe_by_stage[stage] = {}
        u_by_stage[stage] = {}
        for branch in branches:
            safe_by_stage[stage][branch] = {
                args.boundary_name: {
                    str(step): {str(age): True for age in range(1, args.max_age + 1)}
                    for step in range(args.steps)
                }
            }
            u_by_stage[stage][branch] = {
                args.boundary_name: {
                    str(step): {str(age): 0.0 for age in range(1, args.max_age + 1)}
                    for step in range(args.steps)
                }
            }
    payload = {
        "policy_name": "SafeMapCachePolicy",
        "model_name": "JiT",
        "model": "JiT-B/16",
        "forced_safe": True,
        "forced_safe_note": "Smoke-test safe map only; do not use for paper experiments.",
        "steps": args.steps,
        "solver_stages": stages,
        "branches": branches,
        "boundary_groups": {args.boundary_name: modules},
        "module_to_boundary": {module: args.boundary_name for module in modules},
        "max_age": args.max_age,
        "quantile": None,
        "lambda": None,
        "eps": 1e-12,
        "lte_floor": None,
        "safe": safe_by_stage,
        "u_ratio": u_by_stage,
        "cache_age_candidates": list(range(1, args.max_age + 1)),
        "calibration_num_images": 0,
        "seed": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
