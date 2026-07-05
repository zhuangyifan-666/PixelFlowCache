from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pfc.cache.safe_map_policy import compute_safe_map_density


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
    assert density["safe_total"] > 0
    assert density["safe_density"] == 1.0
