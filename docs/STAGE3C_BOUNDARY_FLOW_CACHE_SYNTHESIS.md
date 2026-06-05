# Stage 3C BoundaryFlowCache Synthesis

Stage 3C consolidates the existing JiT Stage 3A and DeCo Stage 3B2 results into a method-level BoundaryFlowCache benchmark and paper-ready analysis. It is a reporting and synthesis stage, not a new cache implementation stage.

## Why Stage 3C Exists

Stage 3A established JiT BackboneCache presets against reduced-step no-cache baselines. Stage 3B2 decomposed DeCo direct-v-pred cache units into final/output, decoder, and backbone boundaries.

The common result is that cache preserves the full-step same-seed trajectory better than reduced-step sampling at similar speed. Stage 3C makes that result explicit across both prediction parameterizations:

- JiT: x-pred model, whole-backbone boundary cache.
- DeCo: v-pred model, output/final velocity boundary cache plus optional upstream cache for speed.

## What Stage 3C Is Not

Stage 3C does not implement token cache, adaptive online policy, solver-aware cache, calibration, frequency-aware cache, PixelGen transfer, PixelDiT transfer, or full ImageNet-scale FID. It also does not modify `third_party/JiT` or `third_party/DeCo`.

## Method-Level Observations

### Observation 1: Pixel-Space Flow Cache Is Boundary-Sensitive

Whole-block output cache is not just independent module skipping. A cached block output overwrites the fresh trajectory at that boundary. Arbitrary block subset cache can waste upstream computation or create feature trajectory mismatch.

### Observation 2: JiT X-Pred Benefits From Whole-Backbone Boundary Cache

The current JiT presets cache all Transformer blocks as one backbone boundary. The strongest settings suppress early high-noise reuse:

- `quality_t02_08`: all blocks, interval 2, active t `[0.2,0.8)`.
- `speed_t02_10`: all blocks, interval 2, active t `[0.2,1.0)`.

This is more reliable than a naive late-only intuition: early high-noise steps are dangerous, but useful reuse can extend through much of the trajectory once those early steps are excluded.

### Observation 3: DeCo V-Pred Quality Is Controlled By Output Boundary

The current DeCo decomposition shows that final/output velocity boundary cache controls same-seed quality. Backbone and decoder cache mainly add speed when paired with an output/final boundary.

Current important DeCo rows:

- `all_candidates`: final/output, backbone, and decoder candidates.
- `backbone_plus_final`: backbone plus final/output boundary.
- `final_only`: final/output boundary only.
- `backbone_only`: backbone boundary without final/output cache.

In the existing 50-step validation, `all_candidates`, `backbone_plus_final`, and `final_only` have the same rel-L2, while `backbone_only` is worse.

### Observation 4: Cache Beats Reduced-Step Sampling At Similar Speed

Both JiT and DeCo show lower same-seed rel-L2 than reduced-step no-cache baselines at comparable speed. This supports treating cache as full-trajectory reuse rather than replacing it with fewer solver steps.

## Proposed Framework Name

The current working method name is `BoundaryFlowCache`.

Alternatives kept for writing:

- `PixelFlowCache-Boundary`
- `Boundary-aware PixelFlowCache`

## Current Method Presets

JiT:

- `BackboneCache-quality`: all blocks, interval 2, active t `[0.2,0.8)`.
- `BackboneCache-speed`: all blocks, interval 2, active t `[0.2,1.0)`.

DeCo:

- `OutputBoundaryCache-quality/speed`: `all_candidates` or `backbone_plus_final`, interval 2, active t `[0.2,1.0)`.
- Diagnostic boundary rows: `final_only`, `backbone_only`, decoder variants.

## Commands

Collect unified results:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
python scripts/collect_stage3c_unified_results.py
```

Generate paper tables:

```bash
UNIFIED_DIR="$(ls -td logs/stage3c/unified/* | head -n 1)"
python scripts/make_stage3c_paper_tables.py --unified-dir "$UNIFIED_DIR"
```

Generate plots:

```bash
python scripts/plot_stage3c_unified.py --unified-dir "$UNIFIED_DIR"
```

Optional DeCo 50-step multi-seed validation:

```bash
export PFC_CUDA_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
conda run -n deco python scripts/run_deco_stage3c_50step_seed_validation.py
```

## Expected Outputs

Unified logs:

- `logs/stage3c/unified/<run_id>/unified_results.csv`
- `logs/stage3c/unified/<run_id>/unified_results.json`
- `logs/stage3c/unified/<run_id>/unified_cache_vs_reduced.csv`
- `logs/stage3c/unified/<run_id>/summary.md`
- `logs/stage3c/unified/<run_id>/unified_boundary_observations.md`
- `logs/stage3c/unified/<run_id>/paper_table_main_cache_vs_reduced.md`
- `logs/stage3c/unified/<run_id>/paper_table_boundary_ablation.md`
- `logs/stage3c/unified/<run_id>/paper_table_seed_stability.md`

Figures:

- `outputs/stage3c/figures/stage3c_speed_quality_jit_deco.png`
- `outputs/stage3c/figures/stage3c_rel_l2_cache_vs_reduced.png`
- `outputs/stage3c/figures/stage3c_speedup_cache_vs_reduced.png`
- `outputs/stage3c/figures/stage3c_boundary_ablation_deco.png`
- `outputs/stage3c/figures/stage3c_jit_vs_deco_best_methods.png`

Optional validation:

- `logs/stage3c/deco_50step_seed_validation/<run_id>/validation_results.csv`
- `logs/stage3c/deco_50step_seed_validation/<run_id>/validation_results.json`
- `logs/stage3c/deco_50step_seed_validation/<run_id>/validation_aggregate.csv`
- `logs/stage3c/deco_50step_seed_validation/<run_id>/summary.md`

## Current Limitations

- Same-seed rel-L2 is not final perceptual quality.
- No FID-scale evaluation yet.
- No token cache.
- No adaptive policy.
- No calibration.
- No PixelGen or PixelDiT transfer yet.

## Next Step Options

- Stage 3D: optional DeCo 50-step multi-seed validation if GPU time is available.
- Stage 4A: PixelGen x-pred transfer.
- Stage 4B: PixelDiT direct-v-pred or dual-level transfer.
- Stage 4C: perceptual metrics and small FID.
