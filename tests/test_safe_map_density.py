from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pfc.cache.safe_map_policy import compute_safe_map_density


ROOT = Path(__file__).resolve().parents[1]


def _map(value: bool) -> dict:
    return {
        "safe": {
            "euler": {
                "global": {
                    "jit_whole_backbone": {
                        "0": {"1": value, "2": False},
                        "1": {"1": True},
                    }
                }
            }
        }
    }


def test_safe_map_density_counts_true_and_total() -> None:
    density = compute_safe_map_density(_map(True))
    assert density["safe_total"] == 3
    assert density["safe_true"] == 2
    assert density["by_age"]["1"]["safe_true"] == 2


def test_safe_map_density_min_density_failure(tmp_path: Path) -> None:
    path = tmp_path / "safe.json"
    path.write_text(json.dumps(_map(False)), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/check_safe_map_density.py", "--safe-map", str(path), "--min-density", "0.99"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert '"safe_true": 1' in result.stdout
