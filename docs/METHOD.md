# BoundaryFlowCache Method

BoundaryFlowCache is a boundary-aware cache for pixel-space flow diffusion models. It reuses selected model outputs inside a timestep window where consecutive flow evaluations are empirically stable, while forcing fresh computation outside the window.

## Core Idea

Flow samplers move from noise toward image. Early and late portions of the trajectory are more sensitive to stale features. BoundaryFlowCache therefore activates only inside a configured timestep window and refreshes on the first eligible hit before reuse begins.

The retained implementation uses a fixed cache interval and explicit method presets. It does not include adaptive calibration, token cache, branch cache, or solver-aware cache.

## PixBFC Interface

The implementation now exposes the method as PixBFC: a generic interface for pixel-space diffusion and flow models. The core code separates three concerns:

- `PixelDiffusionModelAdapter`: model-specific prediction type, boundary candidates, and wrapper installation.
- `BoundarySpec` / `BoundarySet`: named cacheable boundaries such as whole backbone, decoder, or final output.
- `CacheScheduler`: model-independent refresh/reuse scheduling.

JiT and DeCo are the first two adapters. A future pixel-space model should add an adapter rather than changing the cache state or `CachedModule` mechanics. See [PIXBFC_GENERALIZATION.md](PIXBFC_GENERALIZATION.md).

## Adapted Dynamic Baselines

For comparison, the final code also exposes:

- `teacache_style`: raw relative L1 accumulated-distance scheduling on the current image/state proxy `x_t`.
- `seacache_style`: the same accumulated-distance rule after applying a timestep-dependent SEA spectral filter to the `x_t` proxy.

These baselines reuse the same cache units as BoundaryFlowCache but choose refresh/reuse dynamically per step. They are adapted baselines, not official SeaCache implementations.

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

The final codebase is built for reproducible full-generation comparison of no-cache, BoundaryFlowCache, and reduced-step baselines. Historical profiling and exploratory analysis scripts were removed from the current tree.
