from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pfc.eval.pixelgen_runtime import (
    PixelGenRuntimeConfig,
    make_pixelgen_label_conditions,
    pixelgen_heun_timesteps,
    sample_pixelgen_heun_jit,
    simple_guidance_fn,
)


class FakeDenoiser(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scale = labels.to(x.dtype).view(-1, 1, 1, 1) * 1e-4
        return x * 0.8 + t.view(-1, 1, 1, 1) + scale


def _official_equivalent(
    denoiser: nn.Module, labels: torch.Tensor, noise: torch.Tensor, config: PixelGenRuntimeConfig
) -> torch.Tensor:
    x = noise.clone()
    condition, uncondition = make_pixelgen_label_conditions(labels, config.num_classes)
    cfg_condition = torch.cat([uncondition, condition])
    timesteps = pixelgen_heun_timesteps(
        config.steps, config.timeshift, device=x.device, dtype=x.dtype
    )
    for index, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t = t_cur.repeat(x.shape[0])
        cfg_x = torch.cat([x, x])
        out = denoiser(cfg_x, t.repeat(2), cfg_condition)
        velocity = (out - cfg_x) / (1.0 - t.repeat(2).view(-1, 1, 1, 1)).clamp_min(config.t_eps)
        cfg = config.cfg if float(t_cur) > config.guidance_interval_min and float(t_cur) <= config.guidance_interval_max else 1.0
        velocity = simple_guidance_fn(velocity, cfg)
        dt = t_next - t_cur
        x_hat = x + velocity * dt
        if index < config.steps - 1:
            cfg_x_hat = torch.cat([x_hat, x_hat])
            out_hat = denoiser(cfg_x_hat, t_next.repeat(x.shape[0] * 2), cfg_condition)
            v_hat = (out_hat - cfg_x_hat) / (1.0 - t_next.repeat(x.shape[0] * 2).view(-1, 1, 1, 1)).clamp_min(config.t_eps)
            v_hat = simple_guidance_fn(v_hat, cfg)
            x = x + (velocity + v_hat) * 0.5 * dt
        else:
            x = x_hat
    return x


def test_pixelgen_no_cache_matches_official_equivalent_fake_sampler() -> None:
    config = PixelGenRuntimeConfig(
        pixelgen_dir=Path("."), ckpt_path=Path("missing"), img_size=4,
        steps=4, batch_size=2, timeshift=2.0, exact_heun=True,
    )
    labels = torch.tensor([3, 9])
    noise = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(7))
    expected = _official_equivalent(FakeDenoiser(), labels, noise, config)
    actual, records = sample_pixelgen_heun_jit(FakeDenoiser(), labels, noise, config)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert records[-1]["corrector_ran"] is False


def test_pixelgen_hot_path_uses_cpu_metadata_not_scalar_item() -> None:
    source = (Path(__file__).resolve().parents[1] / "pfc/eval/pixelgen_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "_scalar_float" not in source
    assert ".item(" not in source
