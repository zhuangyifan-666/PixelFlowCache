#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as exc:
    if os.environ.get("PFC_STAGE0_SMOKE_REEXEC") == "1":
        raise
    conda = subprocess.run(
        ["conda", "run", "-n", "jit", "python", "-c", "import torch"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if conda.returncode != 0:
        raise ModuleNotFoundError(
            "torch is required for Stage 0 smoke tests. Install torch in the active Python "
            "or run with an environment such as: conda run -n jit python scripts/run_stage0_smoke.py"
        ) from exc
    env = os.environ.copy()
    env["PFC_STAGE0_SMOKE_REEXEC"] = "1"
    raise SystemExit(
        subprocess.call(["conda", "run", "--no-capture-output", "-n", "jit", "python", str(__file__)], cwd=ROOT, env=env)
    )

from pfc.adapters.base import ModelAdapter  # noqa: E402
from pfc.cache.base_policy import NoCachePolicy  # noqa: E402
from pfc.samplers.unified_sampler import UnifiedPixelFlowSampler  # noqa: E402


class FakeXPredAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__("fake_xpred", "xpred")

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any, **kwargs: Any) -> torch.Tensor:
        return x + 2.0


class FakeVPredAdapter(ModelAdapter):
    def __init__(self, velocity: float = 1.0) -> None:
        super().__init__("fake_vpred", "vpred")
        self.velocity = velocity

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any, **kwargs: Any) -> torch.Tensor:
        return torch.full_like(x, self.velocity)


class CondAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__("cond_vpred", "vpred")

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any, **kwargs: Any) -> torch.Tensor:
        value = float(cond)
        return torch.full_like(x, value)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name} failed: {actual} != {expected}")


def append_repro_log(summary: dict[str, Any]) -> None:
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    log_path = docs_dir / "repro_log.md"
    if not log_path.exists():
        log_path.write_text("# PixelFlowCache Repro Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    f"## Stage 0 Smoke - {summary['timestamp_utc']}",
                    "",
                    f"- smoke test passed: {summary['passed']}",
                    f"- checks: {', '.join(summary['checks'])}",
                    "",
                ]
            )
        )


def main() -> int:
    checks: list[str] = []
    x = torch.ones(2, 3, 4, 4)
    raw = x + 2.0
    xpred = FakeXPredAdapter()
    assert_close(xpred.raw_to_velocity(raw, x, 0.5), torch.full_like(x, 4.0), "xpred scalar t")
    checks.append("xpred scalar t")

    t_vec = torch.tensor([0.5, 0.75])
    expected = torch.stack([torch.full((3, 4, 4), 4.0), torch.full((3, 4, 4), 8.0)], dim=0)
    assert_close(xpred.raw_to_velocity(raw, x, t_vec), expected, "xpred vector t")
    checks.append("xpred vector t")

    tokens = torch.ones(2, 5, 3)
    raw_tokens = tokens + 1.0
    expected_tokens = torch.stack([torch.full((5, 3), 2.0), torch.full((5, 3), 4.0)], dim=0)
    assert_close(xpred.raw_to_velocity(raw_tokens, tokens, t_vec), expected_tokens, "token t broadcast")
    checks.append("token t broadcast")

    vpred = FakeVPredAdapter(velocity=3.0)
    assert_close(vpred.raw_to_velocity(torch.full_like(x, 3.0), x, 0.9), torch.full_like(x, 3.0), "vpred raw")
    checks.append("vpred raw")

    sampler = UnifiedPixelFlowSampler(vpred, solver="euler", steps=4)
    sample, diagnostics = sampler.sample(torch.zeros(1, 1, 2, 2), cond=0, cache_policy=NoCachePolicy())
    assert_close(sample, torch.ones_like(sample) * 3.0, "euler sampler")
    assert sample.shape == (1, 1, 2, 2)
    assert diagnostics["num_steps"] == 4
    checks.append("euler sampler")

    cfg_sampler = UnifiedPixelFlowSampler(CondAdapter(), solver="euler", steps=1, cfg_scale=3.0)
    velocity, cfg_diag = cfg_sampler.predict_velocity(torch.zeros(1, 1, 1, 1), 0.5, cond=2.0, uncond=1.0)
    assert_close(velocity, torch.full_like(velocity, 4.0), "cfg formula")
    assert cfg_diag["cfg_enabled"] is True
    checks.append("cfg formula")

    logs_dir = ROOT / "logs/stage0"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "checks": checks,
    }
    out_path = logs_dir / "smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    append_repro_log(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
