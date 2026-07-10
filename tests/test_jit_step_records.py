from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import pfc.eval.jit_runtime as jit_runtime
from pfc.eval.jit_runtime import JiTRuntimeConfig, sample_jit


class FakeNet(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return x * 0.5


class FakeModel:
    net = FakeNet()
    t_eps = 0.05
    num_classes = 1000


def _config() -> JiTRuntimeConfig:
    return JiTRuntimeConfig(
        jit_dir=Path("."), ckpt_dir=Path("."), run_id="fake",
        run_dir=Path("."), preview_dir=Path("."), num_samples=2,
        batch_size=2, steps=3
    )


def test_step_record_norm_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jit_runtime, "l2_norm", lambda _value: pytest.fail("l2_norm called"))
    output, records = sample_jit(
        FakeModel(), torch.tensor([1, 2]), torch.ones(2, 3, 2, 2), _config(), mode="no_cache_50"
    )
    assert output.shape == (2, 3, 2, 2)
    assert records == []


def test_step_record_norm_runs_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def norm(_value: torch.Tensor) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    monkeypatch.setattr(jit_runtime, "l2_norm", norm)
    noise = torch.ones(2, 3, 2, 2)
    default_output, _ = sample_jit(FakeModel(), torch.tensor([1, 2]), noise, _config(), mode="x")
    recorded_output, records = sample_jit(
        FakeModel(), torch.tensor([1, 2]), noise, _config(), mode="x", collect_step_records=True
    )
    assert calls == 3
    assert len(records) == 3
    assert torch.equal(default_output, recorded_output)
