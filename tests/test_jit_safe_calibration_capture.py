from __future__ import annotations

import torch
from torch import nn

from scripts import run_jit_safe_calibration as calibration


class FakeBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + self.offset


class FakeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(1.0), FakeBlock(2.0)])
        self.block_refs = list(self.blocks)

    def forward(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = z
        for block in self.blocks:
            x = block(x)
        label_term = labels.float().view(-1, 1, 1, 1) * 0.01
        time_term = t.float().view(-1, 1, 1, 1) * 0.001
        return x + label_term + time_term


class FakeModel:
    def __init__(self) -> None:
        self.net = FakeNet()
        self.num_classes = 100
        self.t_eps = 0.05


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.zeros(2, 1, 2, 2, dtype=torch.float32)
    t_scalar = torch.tensor(0.25, dtype=torch.float32)
    labels = torch.tensor([3, 4], dtype=torch.long)
    return z, t_scalar, labels


def _cfg(model: FakeModel, controller: object | None = None, **kwargs: object) -> torch.Tensor:
    z, t_scalar, labels = _inputs()
    return calibration._cfg_velocity(
        model=model,
        z=z,
        t_scalar=t_scalar,
        labels=labels,
        cfg_scale=float(kwargs.pop("cfg_scale", 1.5)),
        controller=controller,
        **kwargs,
    )


def _clone_store(store: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in store.items()}


def _call_counts(model: FakeModel) -> list[int]:
    return [block.calls for block in model.net.block_refs]


def test_cfg_velocity_capture_populates_caller_stores_and_disables_controller() -> None:
    model = FakeModel()
    capture_cond: dict[str, torch.Tensor] = {}
    capture_uncond: dict[str, torch.Tensor] = {}

    with calibration._BlockReplayController(model, ["blocks.0", "blocks.1"]) as controller:
        _cfg(model, controller, capture_cond=capture_cond, capture_uncond=capture_uncond)

        assert set(capture_cond) == {"blocks.0", "blocks.1"}
        assert set(capture_uncond) == {"blocks.0", "blocks.1"}
        assert controller.capture_store is None
        assert controller.replay_store is None


def test_cfg_velocity_keeps_empty_capture_dict_identity() -> None:
    model = FakeModel()
    capture_cond: dict[str, torch.Tensor] = {}
    original_id = id(capture_cond)

    with calibration._BlockReplayController(model, ["blocks.0", "blocks.1"]) as controller:
        _cfg(model, controller, capture_cond=capture_cond, capture_uncond={})

    assert id(capture_cond) == original_id
    assert len(capture_cond) > 0


def test_cfg_velocity_replay_uses_cached_activations_without_fresh_block_calls() -> None:
    model = FakeModel()
    capture_cond: dict[str, torch.Tensor] = {}
    capture_uncond: dict[str, torch.Tensor] = {}

    with calibration._BlockReplayController(model, ["blocks.0", "blocks.1"]) as controller:
        _cfg(model, controller, capture_cond=capture_cond, capture_uncond=capture_uncond)
        calls_after_capture = _call_counts(model)

        unaltered = _cfg(model, controller, replay_cond=capture_cond, replay_uncond=capture_uncond)
        replay_cond = _clone_store(capture_cond)
        replay_uncond = _clone_store(capture_uncond)
        replay_cond["blocks.1"] = torch.full_like(replay_cond["blocks.1"], 20.0)
        modified = _cfg(model, controller, replay_cond=replay_cond, replay_uncond=replay_uncond)

        assert _call_counts(model) == calls_after_capture
        assert not torch.allclose(unaltered, modified)
        assert controller.capture_store is None
        assert controller.replay_store is None


def test_cfg_velocity_replay_does_not_capture_into_same_branch() -> None:
    model = FakeModel()
    capture_cond: dict[str, torch.Tensor] = {}
    capture_uncond: dict[str, torch.Tensor] = {}

    with calibration._BlockReplayController(model, ["blocks.0", "blocks.1"]) as controller:
        _cfg(model, controller, capture_cond=capture_cond, capture_uncond=capture_uncond)
        unexpected_cond_capture: dict[str, torch.Tensor] = {}
        unexpected_uncond_capture: dict[str, torch.Tensor] = {}

        _cfg(
            model,
            controller,
            capture_cond=unexpected_cond_capture,
            capture_uncond=unexpected_uncond_capture,
            replay_cond=capture_cond,
            replay_uncond=capture_uncond,
        )

    assert unexpected_cond_capture == {}
    assert unexpected_uncond_capture == {}


def test_cfg_velocity_cond_and_uncond_replay_stores_do_not_leak_between_branches() -> None:
    model = FakeModel()
    capture_cond: dict[str, torch.Tensor] = {}
    capture_uncond: dict[str, torch.Tensor] = {}

    with calibration._BlockReplayController(model, ["blocks.0", "blocks.1"]) as controller:
        _cfg(model, controller, capture_cond=capture_cond, capture_uncond=capture_uncond)
        replay_cond = _clone_store(capture_cond)
        replay_uncond = _clone_store(capture_uncond)
        replay_cond["blocks.1"] = torch.full_like(replay_cond["blocks.1"], 12.0)
        replay_uncond["blocks.1"] = torch.full_like(replay_uncond["blocks.1"], -4.0)

        output = _cfg(model, controller, replay_cond=replay_cond, replay_uncond=replay_uncond, cfg_scale=2.0)
        swapped = _cfg(model, controller, replay_cond=replay_uncond, replay_uncond=replay_cond, cfg_scale=2.0)

        assert not torch.allclose(output, swapped)
        assert controller.capture_store is None
        assert controller.replay_store is None
