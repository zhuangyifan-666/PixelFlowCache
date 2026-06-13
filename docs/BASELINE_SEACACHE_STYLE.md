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
bash scripts/print_stage4a_seacache_theta006_commands.sh
```

The final adapted SeaCache-style baseline threshold is:

```text
theta / delta = 0.06
```

Example JiT command:

```bash
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/run_jit_stage4a_generate.py \
  --method seacache_style \
  --dynamic-cache-threshold 0.06 \
  --num-images 50000 \
  --batch-size 8 \
  --seed 0 \
  --run-id stage4a_jit_seacache_theta0p06_n50000_seed0 \
  --save-png \
  --no-save-npz \
  --resume
```

Example DeCo command:

```bash
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n deco python scripts/run_deco_stage4a_generate.py \
  --method seacache_style \
  --dynamic-cache-threshold 0.06 \
  --num-images 50000 \
  --batch-size 4 \
  --seed 0 \
  --run-id stage4a_deco_seacache_theta0p06_n50000_seed0 \
  --save-png \
  --no-save-npz \
  --resume
```

## Suggested Protocol

1. Print the theta=0.06 command plan with `bash scripts/print_stage4a_seacache_theta006_commands.sh`.
2. Optionally run the 1000-image theta=0.06 command first to confirm runtime behavior.
3. Run the 50k JiT and DeCo theta=0.06 commands manually.
4. Compute FID/IS after generation completes.
5. Collect and plot results into `logs/stage4a/summary/seacache_theta0p06_50k`.

## Warnings

- Do not launch long inference from Codex.
- Do not run FID from Codex.
- This is an adapted baseline, not official SeaCache.
- Exploratory theta values from earlier development are excluded from final results.
- The final repository should not contain `baseline/SeaCache` source or submodule content.
