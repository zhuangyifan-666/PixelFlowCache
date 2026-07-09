from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from pfc.cache.cache_state import CacheEntry
from pfc.cache.safe_map_policy import SafeMapCachePolicy, compute_safe_map_density


ROOT = Path(__file__).resolve().parents[1]


def test_make_forced_safe_map_outputs_all_true_map(tmp_path: Path) -> None:
    out = tmp_path / "forced.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_forced_safe_map.py",
            "--out",
            str(out),
            "--jit-blocks",
            "2",
            "--steps",
            "3",
            "--max-age",
            "2",
            "--branches",
            "global,cond,uncond",
            "--solver-stages",
            "euler",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    density = compute_safe_map_density(payload)
    assert payload["forced_safe"] is True
    assert set(payload["branches"]) == {"global", "cond", "uncond"}
    assert payload["solver_stages"] == ["euler"]
    assert payload["boundary_groups"] == {"jit_whole_backbone": ["blocks.0", "blocks.1"]}
    assert payload["module_to_boundary"] == {
        "blocks.0": "jit_whole_backbone",
        "blocks.1": "jit_whole_backbone",
    }
    for branch in ("global", "cond", "uncond"):
        assert payload["safe"]["euler"][branch]["jit_whole_backbone"]["0"]["1"] is True
    assert density["safe_total"] > 0
    assert density["safe_density"] == 1.0

    policy = SafeMapCachePolicy(safe_map=payload)
    entry = CacheEntry(tensor=torch.ones(1), step_idx=0, t=0.0)
    for branch in ("global", "cond", "uncond"):
        assert policy.should_reuse_entry(
            step_idx=1,
            t=0.1,
            module_name="blocks.1",
            cfg_branch=branch,
            solver_stage="euler",
            entry=entry,
        )
