from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from pfc.cache.spectral_dynamic_policy import RawAccumulatedDistancePolicy
from pfc.eval.pixelgen_runtime import PixelGenRuntimeConfig, sample_pixelgen_heun_jit


class FakePixelGenDenoiser(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return x


def test_pixelgen_dynamic_policy_updates_predictor_and_corrector() -> None:
    policy = RawAccumulatedDistancePolicy(threshold=1.0)
    decisions: list[dict[str, object]] = []
    config = PixelGenRuntimeConfig(
        pixelgen_dir=Path("third_party/PixelGen"),
        ckpt_path=Path("missing.ckpt"),
        img_size=8,
        steps=3,
        batch_size=2,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    noise = torch.zeros(2, 3, 8, 8)

    sample_pixelgen_heun_jit(
        FakePixelGenDenoiser(),
        labels,
        noise,
        config,
        dynamic_policy=policy,
        dynamic_proxy_downsample=4,
        dynamic_decision_writer=decisions.append,
    )

    solver_stages = [decision["solver_stage"] for decision in decisions]
    assert solver_stages == [
        "heun_predictor",
        "heun_corrector",
        "heun_predictor",
        "heun_corrector",
        "heun_predictor",
    ]
    assert {decision["branch"] for decision in decisions} == {"cfg_cat"}
    assert policy.stats().total_steps == 5
