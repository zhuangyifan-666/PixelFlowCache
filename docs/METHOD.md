# BoundaryFlowCache Method

BoundaryFlowCache is a boundary-aware cache for pixel-space flow diffusion models. It reuses selected model outputs inside a timestep window where consecutive flow evaluations are empirically stable, while forcing fresh computation outside the window.

## Core Idea

Flow samplers move from noise toward image. Early and late portions of the trajectory are more sensitive to stale features. BoundaryFlowCache therefore activates only inside a configured timestep window and refreshes on the first eligible hit before reuse begins.

The retained implementation uses a fixed cache interval and explicit method presets. It does not include adaptive calibration, token cache, branch cache, or solver-aware cache.

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
