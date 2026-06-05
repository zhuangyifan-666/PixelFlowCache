#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _set_default_env() -> None:
    defaults = {
        "PFC_STAGE3C_DECO50_SEEDS": "0,1,2",
        "PFC_STAGE3C_DECO50_NUM_SAMPLES": "16",
        "PFC_STAGE3C_DECO50_BATCH_SIZE": "4",
        "PFC_STAGE3C_DECO50_REFERENCE_STEPS": "50",
        "PFC_STAGE3C_DECO50_REDUCED_STEPS": "30",
        "PFC_STAGE3C_DECO50_TIMING_REPEATS": "2",
        "PFC_STAGE3C_DECO50_WARMUP_RUNS": "1",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def main() -> int:
    _set_default_env()

    from scripts.deco_stage3b_common import env_int, make_run_id, parse_int_list
    from scripts.deco_stage3b2_common import run_stage3b2_matrix, write_matrix_outputs

    env_prefix = "PFC_STAGE3C_DECO50"
    reference_steps = env_int(f"{env_prefix}_REFERENCE_STEPS", 50)
    seeds = parse_int_list(os.environ.get(f"{env_prefix}_SEEDS", "0,1,2"))
    reduced_steps = parse_int_list(os.environ.get(f"{env_prefix}_REDUCED_STEPS", "30"))
    run_id = os.environ.get(f"{env_prefix}_RUN_ID", make_run_id(seeds[0], reference_steps, "stage3c-deco50"))
    run_dir = Path(
        os.environ.get(f"{env_prefix}_DIR", ROOT / "logs/stage3c/deco_50step_seed_validation" / run_id)
    ).resolve()
    cache_methods = [
        ("all_candidates", "all_candidates", 2, 0.2, 1.0),
        ("backbone_plus_final", "backbone_plus_final", 2, 0.2, 1.0),
        ("final_only", "final_only", 2, 0.2, 1.0),
        ("backbone_only", "backbone_only", 2, 0.2, 1.0),
    ]
    rows = run_stage3b2_matrix(
        run_id=run_id,
        root_dir=run_dir,
        script_name="scripts/run_deco_stage3c_50step_seed_validation.py",
        seeds=seeds,
        reference_steps=reference_steps,
        cache_methods=cache_methods,
        reduced_steps=reduced_steps,
        env_prefix=env_prefix,
        default_num_samples=16,
        default_batch_size=4,
        default_save_diagnostics=False,
    )
    aggregate = write_matrix_outputs(
        run_dir,
        rows,
        "validation_results.csv",
        "validation_results.json",
        "validation_aggregate.csv",
        "summary.md",
        "DeCo Stage 3C 50-Step Seed Validation",
    )
    print(f"DeCo Stage 3C 50-step validation run dir: {run_dir}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
