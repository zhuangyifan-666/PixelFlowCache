#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_jit_stage3a_backbone_benchmark import main

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    os.environ.setdefault("PFC_STAGE3A_SEEDS", "0")
    os.environ.setdefault("PFC_STAGE3A_NUM_SAMPLES", "32")
    os.environ.setdefault("PFC_STAGE3A_PRESETS", "no_cache,quality_t02_08,speed_t02_10")
    os.environ.setdefault("PFC_STAGE3A_REDUCED_STEPS", "35,40")
    if "PFC_STAGE3A_BENCHMARK_DIR" not in os.environ:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.environ["PFC_STAGE3A_BENCHMARK_DIR"] = str(ROOT / "logs/stage3a/jit_backbone_benchmark_32samples" / f"{stamp}_seed0_ref50")
    raise SystemExit(main())
