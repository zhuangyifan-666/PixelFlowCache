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
    DECOMPOSITION_CACHE_METHODS,
    run_stage3b2_matrix,
    write_matrix_outputs,
)


def main() -> int:
    reference_steps = env_int("PFC_STAGE3B2_REFERENCE_STEPS", 20)
    seed = env_int("PFC_STAGE3B2_SEED", 0)
    seeds = parse_int_list(os.environ.get("PFC_STAGE3B2_SEEDS", str(seed)))
    reduced_steps = parse_int_list(os.environ.get("PFC_STAGE3B2_REDUCED_STEPS", "12,15,18"))
    run_id = os.environ.get("PFC_STAGE3B2_DECOMPOSITION_RUN_ID", make_run_id(seeds[0], reference_steps, "decomposition"))
    run_dir = Path(
        os.environ.get("PFC_STAGE3B2_DECOMPOSITION_DIR", ROOT / "logs/stage3b2/deco_decomposition" / run_id)
    ).resolve()
    rows = run_stage3b2_matrix(
        run_id=run_id,
        root_dir=run_dir,
        script_name="scripts/run_deco_stage3b2_decomposition.py",
        seeds=seeds,
        reference_steps=reference_steps,
        cache_methods=DECOMPOSITION_CACHE_METHODS,
        reduced_steps=reduced_steps,
        env_prefix="PFC_STAGE3B2",
        default_num_samples=8,
        default_batch_size=4,
        default_save_diagnostics=True,
    )
    aggregate = write_matrix_outputs(
        run_dir,
        rows,
        "decomposition_results.csv",
        "decomposition_results.json",
        "decomposition_aggregate.csv",
        "summary.md",
        "DeCo Stage 3B2 Cache Decomposition",
    )
    print(f"DeCo Stage 3B2 decomposition run dir: {run_dir}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
