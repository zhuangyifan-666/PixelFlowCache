# JiT TaylorSeer-Style Baseline

This baseline is an adapted TaylorSeer-style implementation for PixelFlowCache JiT experiments. It is not an official TaylorSeer reproduction.

The official TaylorSeer repository in the sibling folder `../baseline-taylorseer/TaylorSeer` was used only as a conceptual reference. PixelFlowCache implements its own clean-room policy and does not copy large GPL-3.0 code blocks from the official implementation.

## Defaults

- `taylorseer_style`: `interval=4`, `max_order=4`
- `taylorseer_quality_i3_o3`: `interval=3`, `max_order=3`

These settings mirror the public TaylorSeer-DiT recommendation levels, but the JiT integration here forecasts JiT block outputs through PixelFlowCache's `CachedModule` abstraction.

## Mechanism

TaylorSeer-style caching does not directly stale-reuse the last cached feature. At fresh steps, it stores freshly computed JiT block outputs. At forecast steps, it predicts the current block output from recent fresh features using polynomial extrapolation over step indices.

The current implementation uses Lagrange polynomial extrapolation, which supports non-consecutive fresh history steps. No calibration is required.

For formal timing comparisons, remove the old run directory before launching the 1000-image proxy run and do not pass `--resume`. Resume mode is available only as an explicit opt-in for interrupted runs, because skipped images make latency and speedup incomparable.

## Difference From Safe-BFC

- TaylorSeer-style: forecasts features from fresh history.
- Safe-BFC: uses calibrated safe reuse/refresh decisions from a safe map.

This document records the method wiring only. It does not claim any FID, IS, PSNR, SSIM, LPIPS, latency, or speedup result.
