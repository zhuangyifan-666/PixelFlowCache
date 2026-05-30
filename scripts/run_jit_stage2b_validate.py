#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_jit_stage2b_cache import build_config_from_args, run_experiment  # noqa: E402
from scripts.run_jit_stage2b_sweep import _row  # noqa: E402


def _make_run_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _validation_configs() -> list[tuple[str, int, float | None, float | None]]:
    return [
        ("none", 1, None, None),
        ("all", 2, 0.1, 0.8),
        ("all", 2, 0.1, 0.9),
        ("prefix:6", 2, 0.1, 0.8),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# JiT Stage 2B Validation",
        "",
        "| layers | interval | t min | t max | speedup | hit rate | rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cache_layers} | {cache_interval} | {active_t_min} | {active_t_max} | "
            "{speedup_median:.4f} | {cache_hit_rate:.4f} | {same_seed_rel_l2:.6f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base = build_config_from_args([])
    validate_id = os.environ.get("PFC_STAGE2B_VALIDATE_RUN_ID", _make_run_id(base.seed, base.steps))
    validate_dir = Path(
        os.environ.get("PFC_STAGE2B_VALIDATE_DIR", ROOT / "logs/stage2b/jit_validate" / validate_id)
    ).resolve()
    validate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layers, interval, t_min, t_max in _validation_configs():
        run_id = f"{validate_id}_{layers.replace(':', '-')}_i{interval}"
        config = replace(
            base,
            run_id=run_id,
            run_dir=validate_dir / "runs" / run_id,
            preview_dir=ROOT / "outputs/stage2b/previews/jit_validate" / validate_id / run_id,
            cache_layers=layers,
            cache_interval=interval,
            active_t_min=t_min,
            active_t_max=t_max,
            save_previews=False,
            diag_full_probe=False,
        )
        rows.append(_row(run_experiment(config), config))
    _write_csv(validate_dir / "validation_results.csv", rows)
    (validate_dir / "validation_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(validate_dir / "summary.md", rows)
    print(f"JiT Stage 2B validation dir: {validate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
