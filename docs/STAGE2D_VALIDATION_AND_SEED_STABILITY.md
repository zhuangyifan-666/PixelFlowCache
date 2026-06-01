# Stage 2D Validation And Seed Stability

Stage 2D validates JiT fixed whole-backbone cache windows. It is still a JiT-only fixed-cache stage, not the final PixelFlowCache method.

This stage does not implement token cache, DeCo cache, adaptive online policy, solver-aware cache, calibration, or new model weights.

## Motivation

Stage 2C found the best 20-step debug quality at all blocks, interval 2, active t `[0.2,0.8)`. The 50-step validation did not include that window, so Stage 2D adds the missing 50-step comparisons.

Stage 2C also showed that accumulated trajectory drift dominates local probe error, and that the first early cache hit around `t=0.15` can be damaging. Stage 2D therefore adds a first-hit delay option: after a module/CFG branch enters the active window, force a small number of refreshes before reuse.

Finally, Stage 2D checks whether the candidate windows are stable across a small seed sweep.

## Commands

Use one GPU by default:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2d_validate_best_windows.sh
bash scripts/run_jit_stage2d_first_hit_delay.sh
```

If GPU time allows:

```bash
bash scripts/run_jit_stage2d_seed_sweep.sh
```

Plot completed runs:

```bash
python scripts/plot_stage2d_jit.py \
  --validate-dir logs/stage2d/jit_validate_best/<run_id> \
  --first-hit-dir logs/stage2d/jit_first_hit_delay/<run_id> \
  --seed-sweep-dir logs/stage2d/jit_seed_sweep/<run_id>
```

Useful overrides:

```bash
export PFC_STAGE2D_NUM_SAMPLES=32
export PFC_STAGE2D_BATCH_SIZE=4
export PFC_STAGE2D_STEPS=50
export PFC_STAGE2D_TIMING_REPEATS=3
export PFC_STAGE2D_FIRST_HIT_WARMUPS=0,1,2
export PFC_STAGE2D_SEEDS=0,1,2
export PFC_STAGE2D_SEED_NUM_SAMPLES=16
```

## Expected Outputs

Best-window validation:

- `logs/stage2d/jit_validate_best/<run_id>/validation_results.csv`
- `logs/stage2d/jit_validate_best/<run_id>/validation_results.json`
- `logs/stage2d/jit_validate_best/<run_id>/summary.md`

First-hit delay:

- `logs/stage2d/jit_first_hit_delay/<run_id>/first_hit_delay_results.csv`
- `logs/stage2d/jit_first_hit_delay/<run_id>/first_hit_delay_results.json`
- `logs/stage2d/jit_first_hit_delay/<run_id>/summary.md`

Seed sweep:

- `logs/stage2d/jit_seed_sweep/<run_id>/seed_sweep_results.csv`
- `logs/stage2d/jit_seed_sweep/<run_id>/seed_sweep_results.json`
- `logs/stage2d/jit_seed_sweep/<run_id>/summary.md`

Figures:

- `outputs/stage2d/figures/jit_stage2d_validate_speed_quality.png`
- `outputs/stage2d/figures/jit_stage2d_seed_rel_l2_mean_std.png`
- `outputs/stage2d/figures/jit_stage2d_seed_speedup_mean_std.png`
- `outputs/stage2d/figures/jit_stage2d_first_hit_delay_rel_l2.png`
- `outputs/stage2d/figures/jit_stage2d_first_hit_delay_speedup.png`

## Interpretation

- Compare `[0.2,0.8)` against `[0.1,0.8)` to test whether skipping the first active hit remains useful at 50 steps.
- Compare `[0.2,1.0)` against `[0.1,1.0)` to test whether a later start keeps the high-speed wide window acceptable.
- First-hit delay isolates whether a forced refresh after entering the active window can recover quality without giving up the whole early active region.
- Seed sweep reports mean and population standard deviation across seeds for speedup, hit rate, rel-L2, PSNR, and MSE.

## Limitations

The experiments remain small same-seed diagnostics. They do not replace full ImageNet-scale FID or human inspection. The cache remains fixed whole-block output reuse for JiT only.

## Next Step

Use Stage 2D to choose the final JiT fixed-cache baseline. After that, decide whether the next research step should generalize to a DeCo branch-level cache or design a trajectory-aware adaptive schedule.
