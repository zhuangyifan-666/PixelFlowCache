from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_theta006_command_plan_contains_only_final_threshold() -> None:
    result = subprocess.run(
        ["bash", "scripts/print_stage4a_seacache_theta006_commands.sh"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "--method seacache_style" in output
    assert "--dynamic-cache-threshold 0.06" in output
    assert "stage4a_jit_seacache_theta0p06" in output
    assert "stage4a_deco_seacache_theta0p06" in output
    assert "0.05" not in output
    assert "theta0p05" not in output
    assert "delta0p05" not in output
