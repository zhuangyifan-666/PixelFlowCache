#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.deco_stage3b_common import (  # noqa: E402
    DeCoStage3BConfig,
    build_deco_denoiser_structure,
    candidate_modules,
    default_deco_config,
    default_deco_dir,
    detect_deco_ckpt,
    make_run_id,
    setup_deco_pythonpath,
    write_common_meta,
)


FIELDNAMES = [
    "module_name",
    "module_kind",
    "module_category",
    "cache_candidate",
    "parameter_count",
    "mean_rel_l2_delta",
]


def _load_stage1_deltas() -> dict[str, float]:
    candidates: list[Path] = sorted((ROOT / "logs/stage1/deco").glob("*/feature_delta_by_module.csv"))
    if not candidates:
        return {}
    latest = candidates[-1]
    deltas: dict[str, float] = {}
    with latest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("module_name")
            value = row.get("mean_rel_l2_delta")
            if name and value:
                try:
                    deltas[name] = float(value)
                except ValueError:
                    pass
    return deltas


def inspect_deco_cache_units(config: DeCoStage3BConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    setup_deco_pythonpath(config.deco_dir)
    device = torch.device("cpu")
    denoiser = build_deco_denoiser_structure(config, device)
    stage1_deltas = _load_stage1_deltas()
    rows = []
    category_counts: dict[str, int] = defaultdict(int)
    cacheable_count = 0
    for name, module, category, cacheable in candidate_modules(denoiser):
        category_counts[category] += 1
        cacheable_count += int(cacheable)
        rows.append(
            {
                "module_name": name,
                "module_kind": module.__class__.__name__,
                "module_category": category,
                "cache_candidate": cacheable,
                "parameter_count": sum(param.numel() for param in module.parameters()),
                "mean_rel_l2_delta": stage1_deltas.get(name),
            }
        )

    with (config.run_dir / "module_tree.txt").open("w", encoding="utf-8") as handle:
        for name, module in denoiser.named_modules():
            if name:
                handle.write(f"{name}\t{module.__class__.__name__}\n")
    with (config.run_dir / "module_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    (config.run_dir / "module_candidates.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "module_count": len(list(denoiser.named_modules())) - 1,
        "candidate_count": len(rows),
        "cacheable_count": cacheable_count,
        "category_counts": dict(sorted(category_counts.items())),
        "joined_stage1_deltas": bool(stage1_deltas),
    }
    lines = [
        "# DeCo Stage 3B Cache Unit Inspection",
        "",
        f"- run dir: `{config.run_dir}`",
        f"- module count: {summary['module_count']}",
        f"- listed candidates: {summary['candidate_count']}",
        f"- cacheable block-level units: {summary['cacheable_count']}",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count} |")
    (config.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_common_meta(config, "scripts/inspect_deco_cache_units.py", extra=summary)
    return summary


def main() -> int:
    seed = int(os.environ.get("PFC_STAGE3B_SEED", 0))
    steps = int(os.environ.get("PFC_STAGE3B_STEPS", 20))
    run_id = os.environ.get("PFC_STAGE3B_INSPECT_RUN_ID", make_run_id(seed, steps, "inspect"))
    config = DeCoStage3BConfig(
        deco_dir=default_deco_dir(),
        ckpt_path=detect_deco_ckpt(),
        config_path=default_deco_config(),
        run_id=run_id,
        run_dir=Path(os.environ.get("PFC_STAGE3B_INSPECT_DIR", ROOT / "logs/stage3b/deco_inspect" / run_id)).resolve(),
        steps=steps,
        seed=seed,
    )
    summary = inspect_deco_cache_units(config)
    print(f"DeCo Stage 3B inspect run dir: {summary['run_dir']}")
    print(f"Cacheable units: {summary['cacheable_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
