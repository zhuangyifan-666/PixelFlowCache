from __future__ import annotations

from typing import Any

import torch

from pfc.core.boundary_spec import BoundarySet, PredictionType
from pfc.core.model_adapter import PixelDiffusionModelAdapter


class DummyAdapter(PixelDiffusionModelAdapter):
    def __init__(self, prediction_type: PredictionType) -> None:
        self.prediction_type = prediction_type

    def list_boundary_candidates(self, model: Any) -> list[Any]:
        return []

    def default_boundary_set(self, model: Any, preset_name: str | None = None) -> BoundarySet:
        return BoundarySet(name="empty", boundaries=())

    def wrap_boundary_set(self, model: Any, boundary_set: BoundarySet, cache_state: Any, policy: Any) -> list[str]:
        return []

    def branch_mode(self) -> str:
        return "test"


def test_xpred_output_to_velocity_scalar_t() -> None:
    adapter = DummyAdapter(PredictionType.XPRED)
    x = torch.ones(2, 3, 4, 4)
    out = x + 0.5
    velocity = adapter.output_to_velocity(out, x, 0.5)
    assert torch.allclose(velocity, torch.ones_like(x))


def test_xpred_output_to_velocity_vector_t_broadcasts_bchw() -> None:
    adapter = DummyAdapter(PredictionType.XPRED)
    x = torch.zeros(2, 3, 4, 4)
    out = torch.ones_like(x)
    t = torch.tensor([0.0, 0.5])
    velocity = adapter.output_to_velocity(out, x, t)
    assert torch.allclose(velocity[0], torch.ones_like(velocity[0]))
    assert torch.allclose(velocity[1], 2.0 * torch.ones_like(velocity[1]))


def test_vpred_output_to_velocity_identity() -> None:
    adapter = DummyAdapter(PredictionType.VPRED)
    x = torch.zeros(2, 3, 4, 4)
    out = torch.randn_like(x)
    assert adapter.output_to_velocity(out, x, 0.5) is out
