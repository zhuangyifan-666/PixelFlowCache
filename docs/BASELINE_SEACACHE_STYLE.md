# Adapted SeaCache-Style Baseline

This repository implements an independent SeaCache-style baseline for JiT and DeCo BoundaryFlowCache experiments. It uses SeaCache as a reference idea, but the implementation is local to `pfc/` and is adapted to pixel-space flow generation.

## What It Is

The baseline is a training-free dynamic cache schedule with accumulated distance:

- TeaCache-style baseline: raw relative L1 distance between proxy features.
- SeaCache-style baseline: relative L1 distance after a timestep-dependent SEA spectral filter.
- Refresh rule: accumulate per-step distance and refresh when accumulated distance exceeds threshold `delta`.

## Difference From Official SeaCache

This is not an official SeaCache result or a drop-in copy of SeaCache source.

Differences:

- It targets JiT-B/16 and DeCo ImageNet-256 pixel-space flow models.
- It uses the current noisy image/state `x_t` as a low-overhead BCHW proxy by default.
- It does not compute the distance from model outputs, because that would require a full forward before deciding whether to cache.
- It uses BoundaryFlowCache cache units for fair comparison:
  - JiT: all backbone blocks.
  - DeCo: safe `all_candidates` modules.

## SEA Filter

The sampler convention is:

- `t = 0`: noise
- `t = 1`: image

For the adapted SEA filter:

```text
a(t) = clamp(t, eps, 1)
b(t) = clamp(1 - t, eps, 1)
S(f) = |f|^{-beta}
G_t(f) = a(t) * S(f) / (a(t)^2 * S(f) + b(t)^2)
```

The filter is normalized to a stable mean gain by default. Filtering is applied with FFT/iFFT over BCHW proxy tensors. Proxies are downsampled to at most `64x64` by default to keep overhead low.

## Commands

Print single-GPU command plans without running inference:

```bash
bash scripts/print_stage4a_seacache_baseline_commands.sh
```

Example JiT command:

```bash
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/run_jit_stage4a_generate.py \
  --method seacache_style \
  --dynamic-cache-threshold 0.10 \
  --num-images 1000 \
  --batch-size 8 \
  --seed 0 \
  --run-id stage4a_jit_seacache_n1000_delta0p10_seed0 \
  --save-png \
  --no-save-npz \
  --resume
```

Example DeCo command:

```bash
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n deco python scripts/run_deco_stage4a_generate.py \
  --method seacache_style \
  --dynamic-cache-threshold 0.10 \
  --num-images 1000 \
  --batch-size 4 \
  --seed 0 \
  --run-id stage4a_deco_seacache_n1000_delta0p10_seed0 \
  --save-png \
  --no-save-npz \
  --resume
```

## Suggested Protocol

1. Run the 1000-image threshold sweep for TeaCache-style and SeaCache-style methods.
2. Collect latency and optional proxy FID/IS.
3. Choose thresholds that match or bracket BoundaryFlowCache speedups.
4. Run 50k only for selected thresholds.
5. Collect and plot results with the existing Stage 4A collection utilities.

## Warnings

- Do not launch long inference from Codex.
- Do not run FID from Codex.
- This is an adapted baseline, not official SeaCache.
- The final repository should not contain `baseline/SeaCache` source or submodule content.
