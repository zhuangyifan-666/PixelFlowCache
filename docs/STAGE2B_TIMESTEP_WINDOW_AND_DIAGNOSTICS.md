# Stage 2B Timestep Window And Diagnostics

Stage 2B improves the JiT fixed-interval whole-block cache baseline. It is still a JiT-only baseline stage, not the final PixelFlowCache method.

## Why Stage 2B

Stage 2 proved that whole-block caching can produce real compute skipping and measurable speedup. It also showed that naive fixed-interval caching can introduce large same-seed drift. In the initial debug grid, `all/i2` had better rel-L2 than `middle/i2`, which suggests that consistent layer-group behavior can matter more than caching only visually smooth middle blocks.

JiT predicts `x0` and converts to velocity:

```text
v = (x0_pred - z) / clamp(1 - t)
```

Late-step `x0` errors can therefore be amplified by `1 / (1 - t)`. Stage 2B adds timestep windows to test disabling cache near late sampling steps.

## New Features

- Timestep and step-index cache windows in `FixedIntervalCachePolicy`.
- Layer specs: `prefix:n`, `suffix:n`, `range:start:end`, `every:stride`, `complement:spec`, and `stage1top:csv:k`.
- Per-batch cache entry clearing without resetting accumulated stats.
- Repeated timing with median and mean latency.
- Step-level velocity diagnostics in `step_error_stats.jsonl`.
- Optional full-on-cache-state probe via `PFC_STAGE2B_DIAG_FULL_PROBE=1`.

## How To Run

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
conda activate jit
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2b_cache.sh
```

Fast sweep:

```bash
export PFC_STAGE2B_SWEEP_FAST=1
bash scripts/run_jit_stage2b_sweep.sh
```

Plot:

```bash
python scripts/plot_stage2b_jit.py --sweep-dir logs/stage2b/jit_sweep/<run_id>
```

Optional validation:

```bash
bash scripts/run_jit_stage2b_validate.sh
```

Useful overrides:

```bash
export PFC_STAGE2B_NUM_SAMPLES=8
export PFC_STAGE2B_BATCH_SIZE=4
export PFC_STAGE2B_STEPS=20
export PFC_STAGE2B_CACHE_INTERVAL=2
export PFC_STAGE2B_CACHE_LAYERS=all
export PFC_STAGE2B_ACTIVE_T_MIN=0.1
export PFC_STAGE2B_ACTIVE_T_MAX=0.8
export PFC_STAGE2B_TIMING_REPEATS=3
export PFC_STAGE2B_DIAG_FULL_PROBE=0
```

## Outputs

Single runs write:

- `logs/stage2b/jit/<run_id>/meta.json`
- `logs/stage2b/jit/<run_id>/config.json`
- `logs/stage2b/jit/<run_id>/no_cache_summary.json`
- `logs/stage2b/jit/<run_id>/cache_summary.json`
- `logs/stage2b/jit/<run_id>/comparison.json`
- `logs/stage2b/jit/<run_id>/cache_stats.json`
- `logs/stage2b/jit/<run_id>/step_error_stats.jsonl`

Sweeps write:

- `logs/stage2b/jit_sweep/<run_id>/sweep_results.csv`
- `logs/stage2b/jit_sweep/<run_id>/sweep_results.json`
- `logs/stage2b/jit_sweep/<run_id>/summary.md`

Figures go under ignored `outputs/stage2b/figures/`.

## How To Interpret

- `speedup_median`: no-cache median latency divided by cached median latency.
- `cache_hit_rate`: fraction of wrapped block calls served from cache.
- `same_seed_rel_l2`: final image drift relative to same-seed no-cache sampling.
- `low/mid/high_freq_delta_ratio`: frequency composition of final image difference.
- `trajectory_error`: cached trajectory velocity compared to no-cache trajectory velocity at the same step.
- `probe_error`: optional local cache approximation error on the cached state when full probe is enabled.
- `amplification`: `1 / max(1 - t, eps)`, useful for spotting late-step risk.

Trajectory error includes accumulated sampling divergence. Probe error, when enabled, isolates local approximation error but costs extra model forwards and is not included in primary timing.

## Not Implemented

- No token cache.
- No DeCo cache.
- No adaptive online policy.
- No calibration.
- No final PixelFlowCache method.

## Next Step

Use Stage 2B sweep results to choose safer JiT layer/window baselines, then run a separate DeCo branch-cache feasibility study.
