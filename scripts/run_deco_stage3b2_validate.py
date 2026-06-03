#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.deco_stage3b_common import env_int, make_run_id, parse_int_list  # noqa: E402
from scripts.deco_stage3b2_common import (  # noqa: E402
    VALIDATION_CACHE_METHODS,
    env_bool,
    run_stage3b2_matrix,
    write_matrix_outputs,
)


def main() -> int:
    fast = env_bool("PFC_STAGE3B2_VALIDATE_FAST", False)
    reference_steps = env_int("PFC_STAGE3B2_VALIDATE_REFERENCE_STEPS", 30 if fast else 50)
    default_reduced = "18,22,26" if fast else "30,35,40"
    seeds = parse_int_list(os.environ.get("PFC_STAGE3B2_VALIDATE_SEEDS", "0"))
    reduced_steps = parse_int_list(os.environ.get("PFC_STAGE3B2_VALIDATE_REDUCED_STEPS", default_reduced))
    run_id = os.environ.get("PFC_STAGE3B2_VALIDATE_RUN_ID", make_run_id(seeds[0], reference_steps, "validate"))
    run_dir = Path(os.environ.get("PFC_STAGE3B2_VALIDATE_DIR", ROOT / "logs/stage3b2/deco_validate" / run_id)).resolve()
    rows = run_stage3b2_matrix(
        run_id=run_id,
        root_dir=run_dir,
        script_name="scripts/run_deco_stage3b2_validate.py",
        seeds=seeds,
        reference_steps=reference_steps,
        cache_methods=VALIDATION_CACHE_METHODS,
        reduced_steps=reduced_steps,
        env_prefix="PFC_STAGE3B2_VALIDATE",
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
        "DeCo Stage 3B2 50-Step Validation",
    )
    print(f"DeCo Stage 3B2 validation run dir: {run_dir}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
