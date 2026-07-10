# BoundaryFlowCache Method

BoundaryFlowCache is a boundary-aware cache for pixel-space flow diffusion models. It reuses selected model outputs inside a timestep window where consecutive flow evaluations are empirically stable, while forcing fresh computation outside the window.

## Core Idea

Flow samplers move from noise toward image. Early and late portions of the trajectory are more sensitive to stale features. BoundaryFlowCache therefore activates only inside a configured timestep window and refreshes on the first eligible hit before reuse begins.

The canonical method registry is `pfc/eval/method_presets.py`. Fixed BFC remains a legacy/diagnostic comparison. Experimental Safe-BFC uses calibrated safe maps. Adapted SeaCache, TaylorSeer, SpeCa, and DiCache policies provide the main baseline interfaces. PixARC Stage-1 instrumentation is implemented; the Stage-2 oracle and final scheduler are not implemented.

## PixBFC Interface

The implementation now exposes the method as PixBFC: a generic interface for pixel-space diffusion and flow models. The core code separates three concerns:

- `PixelDiffusionModelAdapter`: model-specific prediction type, boundary candidates, and wrapper installation.
- `BoundarySpec` / `BoundarySet`: named cacheable boundaries such as whole backbone, decoder, or final output.
- `CacheScheduler`: model-independent refresh/reuse scheduling.

JiT, DeCo, and PixelGen have adapters. PixelDiT is pinned as third-party source only and still needs a runtime adapter. A future pixel-space model should add an adapter rather than changing the cache state or `CachedModule` mechanics. See [PIXBFC_GENERALIZATION.md](PIXBFC_GENERALIZATION.md).

## Adapted Dynamic Baselines

For comparison, the final code also exposes:

- `teacache_style`: raw relative L1 accumulated-distance scheduling on the current image/state proxy `x_t`.
- `seacache_style`: the same accumulated-distance rule after applying a timestep-dependent SEA spectral filter to the `x_t` proxy.

These baselines reuse the same cache units as BoundaryFlowCache but choose refresh/reuse dynamically per step. They are adapted baselines, not official SeaCache implementations.

TaylorSeer uses branch-isolated history. SpeCa uses forecast plus explicit verification accounting. DiCache uses a probe/deep-block schedule with CFG-prefix sharing disabled by default for fairness. Safe-BFC is experimental and requires a nonempty calibrated map. Generation defaults disable per-step host diagnostics so timing does not acquire hidden CUDA synchronization.

## JiT

JiT is treated as an x-pred pixel-space flow model. The final presets wrap JiT transformer blocks with whole-backbone BoundaryFlowCache:

- `no_cache_50`: 50-step reference
- `bfc_quality_t02_08`: cache all backbone blocks, interval 2, active t `[0.2, 0.8)`
- `bfc_speed_t02_10`: cache all backbone blocks, interval 2, active t `[0.2, 1.0)`
- `reduced_steps_35`: 35-step no-cache baseline
- `reduced_steps_30`: 30-step no-cache baseline

## DeCo

DeCo is treated as a v-pred pixel-space flow model. The final presets cache safe module outputs around block, decoder, and final velocity-producing modules:

- `no_cache_50`: 50-step reference
- `bfc_all_candidates_t02_10`: cache safe backbone, decoder, and final modules, interval 2, active t `[0.2, 1.0)`
- `bfc_backbone_plus_final_t02_10`: cache backbone plus final module, interval 2, active t `[0.2, 1.0)`
- `reduced_steps_35`: 35-step no-cache baseline
- `reduced_steps_30`: 30-step no-cache baseline

The DeCo wrapper excludes normalization, modulation, embedding, dropout, and tiny modules from cache candidates.

## Current Scope

The runtime cache validates a low-cost input signature (shape, dtype, device, batch, and session) before reuse and records output metadata. It never hashes tensor contents. Formal speed comparison uses synchronized single-GPU sampling latency; model load, input preparation, postprocess, PNG/NPZ, manifest I/O, resume skips, and parallel orchestration are separately recorded.

## Policy reset contract

Cache policies expose two distinct reset operations. `clear_batch()` and `reset_runtime_state()` clear batch/session history and pending decisions; `reset_stats()` clears cumulative counters, running moments, bounded diagnostic samples, and host-dispatch timings without changing immutable policy configuration. JiT warmup completion calls runtime reset and statistics reset for Safe-BFC, TaylorSeer, SpeCa, and DiCache, and also clears and resets `RuntimeCacheState`. Warmup activity is therefore excluded from formal `cache_stats.json` output.
