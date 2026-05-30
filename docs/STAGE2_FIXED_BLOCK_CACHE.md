# Stage 2 Fixed-Interval JiT Block Cache

Stage 2 implements the first real compute-skipping baseline in PixelFlowCache: a fixed-interval whole-block cache for JiT Transformer blocks.

This is a baseline to prove the runtime path works. It is not the final PixelFlowCache method.

## Goal

- Wrap JiT `denoiser.net.blocks`.
- Reuse cached block outputs on fixed intervals.
- Actually skip selected block forward calls on cache hits.
- Preserve a clean no-cache same-seed baseline.
- Measure wall-clock speedup, cache hit rate, and same-seed output degradation.

## Implemented

- `RuntimeCacheState` for runtime tensor entries and JSON-serializable hit/miss stats.
- `FixedIntervalCachePolicy` for interval-based refresh/reuse decisions.
- `CachedModule` wrapper that skips the original module on cache hits.
- JiT block wrapping helpers for `all`, `none`, `early`, `middle`, `late`, comma lists, and `topk:<csv>:<k>`.
- Same-seed JiT comparison between fresh no-cache and fresh cached models.
- Stage 2 single-run, grid, and plotting scripts.

## Not Implemented

- No token cache.
- No DeCo branch cache.
- No frequency-aware policy.
- No solver-aware Heun or multistep cache.
- No calibration.
- No FID-scale evaluation.

## Why JiT First

JiT exposes a clean `denoiser.net.blocks` `ModuleList`, and JiT-B/16 has 12 blocks. Its sampler predicts `x0`, then converts to velocity:

```text
v = (x0_pred - z) / clamp(1 - t)
```

That makes JiT the lowest-risk model for validating real block-level compute skipping before touching DeCo.

## How To Run

Use one GPU by default:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
conda activate jit
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2_cache.sh
```

Run the fast grid:

```bash
export PFC_STAGE2_GRID_FAST=1
bash scripts/run_jit_stage2_grid.sh
```

Plot a grid:

```bash
python scripts/plot_stage2_jit_cache.py --grid-dir logs/stage2/jit_grid/<run_id>
```

Useful overrides:

```bash
export PFC_STAGE2_NUM_SAMPLES=8
export PFC_STAGE2_BATCH_SIZE=4
export PFC_STAGE2_STEPS=20
export PFC_STAGE2_CACHE_INTERVAL=2
export PFC_STAGE2_CACHE_LAYERS=middle
export PFC_STAGE2_CACHE_BRANCHES=cond,uncond
export PFC_STAGE2_WARMUP_RUNS=1
```

## Outputs

Single runs write:

- `logs/stage2/jit/<run_id>/meta.json`
- `logs/stage2/jit/<run_id>/config.json`
- `logs/stage2/jit/<run_id>/no_cache_summary.json`
- `logs/stage2/jit/<run_id>/cache_summary.json`
- `logs/stage2/jit/<run_id>/comparison.json`
- `logs/stage2/jit/<run_id>/cache_stats.json`
- `logs/stage2/jit/<run_id>/step_stats.jsonl`

Grid runs write:

- `logs/stage2/jit_grid/<run_id>/grid_results.csv`
- `logs/stage2/jit_grid/<run_id>/grid_results.json`
- `logs/stage2/jit_grid/<run_id>/summary.md`

Ignored visual outputs:

- `outputs/stage2/previews/jit/<run_id>/`
- `outputs/stage2/figures/`

## How To Interpret

- `speedup`: no-cache latency divided by cached latency. Values above 1 mean the cached run was faster.
- `cache_hit_rate`: fraction of wrapped block calls served from cache.
- `same_seed_mse` and `same_seed_rel_l2`: degradation relative to the same-seed no-cache output.
- `frequency_delta.high_ratio`: high-frequency share of the output difference.

The default script runs one unmeasured warmup for both no-cache and cached modes before timing. Small debug runs are useful for engineering validation, not final model-quality claims.

## DeCo Status

DeCo cache is deferred. Existing Stage 1 DeCo candidates can be exported with:

```bash
python scripts/export_stage2_cache_candidates.py --run-dir logs/stage1/deco/<run_id>
```

The current local Stage 1 DeCo candidate export lists modules such as `dec_net.final_layer`, `dec_net.res_blocks.0`, and late `blocks.*` modules as smooth candidates, but no DeCo modules are wrapped in Stage 2.

## Limitations

- Default runs use only 8 samples and 20 Euler steps.
- No FID is computed.
- Cache quality is measured against same-seed no-cache output only.
- Torch compile and CUDA startup effects can affect small-run latency.
- Profiling hooks from Stage 1 are not part of Stage 2 timing.

## Next Step

Stage 2B should use Stage 1 candidate ranking to choose better JiT layers and run a broader speed-quality sweep. A separate DeCo branch-cache feasibility study should come after the JiT baseline is stable.
