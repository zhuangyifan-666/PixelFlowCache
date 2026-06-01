# Stage 3A JiT BackboneCache Benchmark

Stage 3A consolidates the JiT whole-backbone cache from Stage 2D into named BackboneCache presets and compares those presets against reduced-step no-cache baselines.

This is still a JiT-only benchmark stage. It does not implement token cache, DeCo cache, adaptive online policy, solver-aware cache, calibration, or full ImageNet-scale FID.

## Motivation

Stage 2D identified stable whole-backbone cache policies:

- `quality_t02_08`: all blocks, interval 2, active t `[0.2,0.8)`.
- `speed_t02_10`: all blocks, interval 2, active t `[0.2,1.0)`.
- First-hit-delay variants for `[0.1,0.8)`.

These policies must be compared against a simple alternative: reducing no-cache sampling steps. A cache speedup is only useful if it preserves the 50-step trajectory better than a reduced-step no-cache baseline at similar speed.

## BackboneCache Presets

- `quality_t02_08`: quality-first Stage 2D preset.
- `speed_t02_10`: speed-quality Stage 2D preset.
- `quality_t01_08_w1`: `[0.1,0.8)` with one active-window warmup refresh.
- `quality_t01_08_w2`: `[0.1,0.8)` with two active-window warmup refreshes.
- `aggressive_i3_t02_08`: interval-3 comparison at `[0.2,0.8)`.
- `no_cache`: 50-step no-cache reference.

Whole-block output caching in JiT behaves as boundary/backbone-level trajectory reuse, not independent arbitrary layer caching. See [STAGE2C_BOUNDARY_CACHE_OBSERVATION.md](STAGE2C_BOUNDARY_CACHE_OBSERVATION.md).

## Commands

Use one GPU by default:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage3a_backbone_benchmark.sh
```

Plot and generate report tables:

```bash
BENCHMARK_DIR="$(ls -td logs/stage3a/jit_backbone_benchmark/* | head -n 1)"
python scripts/plot_stage3a_jit.py --benchmark-dir "$BENCHMARK_DIR"
python scripts/make_stage3a_report_tables.py --benchmark-dir "$BENCHMARK_DIR"
```

Optional standalone reduced-step run:

```bash
bash scripts/run_jit_stage3a_reduced_steps.sh
```

Optional 32-sample subset:

```bash
bash scripts/run_jit_stage3a_backbone_benchmark_32samples.sh
```

## Expected Outputs

Benchmark:

- `logs/stage3a/jit_backbone_benchmark/<run_id>/benchmark_results.csv`
- `logs/stage3a/jit_backbone_benchmark/<run_id>/benchmark_results.json`
- `logs/stage3a/jit_backbone_benchmark/<run_id>/benchmark_aggregate.csv`
- `logs/stage3a/jit_backbone_benchmark/<run_id>/summary.md`
- `logs/stage3a/jit_backbone_benchmark/<run_id>/paper_table.md`
- `logs/stage3a/jit_backbone_benchmark/<run_id>/paper_table.csv`

Figures:

- `outputs/stage3a/figures/jit_stage3a_speed_quality_cache_vs_reduced.png`
- `outputs/stage3a/figures/jit_stage3a_rel_l2_by_method.png`
- `outputs/stage3a/figures/jit_stage3a_speedup_by_method.png`
- `outputs/stage3a/figures/jit_stage3a_cache_hit_rate_by_method.png`
- `outputs/stage3a/figures/jit_stage3a_frequency_delta_by_method.png`

## Interpretation

- `speedup_vs_reference` compares a method to the same-seed 50-step no-cache reference.
- Cache rows use the same 50 Euler steps and apply the preset cache policy.
- Reduced-step rows disable cache and use fewer Euler steps.
- `same_seed_rel_l2`, MSE, PSNR, and frequency delta compare each method output against the 50-step no-cache output from the same seed and initial noise.
- A strong BackboneCache preset should beat reduced-step no-cache baselines at similar speed.

## Limitations

- Same-seed image difference only; no full FID yet.
- JiT-only.
- No DeCo, PixelDiT, or PixelGen cache yet.
- No LPIPS/DINO perceptual metrics unless added in a later stage.

## Next Step

Use Stage 3A to select the JiT BackboneCache baseline. Stage 3B should either test DeCo branch/backbone cache feasibility or add perceptual metrics such as LPIPS/DINO if the packages are already available.
