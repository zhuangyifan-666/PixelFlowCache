# JiT Safe-BFC 1000-Image Proxy Plan

This document lists entry points for a 1000-image proxy experiment only. These proxy results should not be interpreted as final 50k FID/IS.

## Flow

1. Run Safe-BFC calibration to produce `safe_map_quality.json` and `safe_map_speed.json`.
2. Generate 1000 images for `no_cache_50`, `safe_bfc_quality`, `safe_bfc_speed`, `seacache_style`, `reduced_steps_35`, and `reduced_steps_30`.
3. Compute FID/IS for each method's own image folder.
4. Compute paired PSNR/SSIM/LPIPS/rel_l2 for each non-reference method against the same-run `no_cache_50` images.
5. Collect `generation_meta.json`, `latency.json`, `cache_stats.json`, FID/IS, and paired metrics into one summary.

## Print Commands

The planner prints commands only; it does not run calibration, generation, FID/IS, or paired metrics.

```bash
bash scripts/print_jit_safe_1000_commands.sh --dry-run
```

Equivalent Python entry point:

```bash
conda run -n jit python scripts/run_jit_safe_1000_eval_plan.py --dry-run
```

## Calibration Entry Point

```bash
conda run -n jit python scripts/run_jit_safe_calibration.py \
  --num-calibration-images 128 \
  --batch-size 8 \
  --seed 123 \
  --run-id stage5a_jit_safe_calib128_seed123 \
  --jit-ckpt-dir ckpts/JiT/JiT-B-16-256 \
  --out-dir calibrations/jit_safe/stage5a_jit_safe_calib128_seed123 \
  --boundary-groups whole_backbone \
  --max-age 3 \
  --quantile 0.95 \
  --quality-lambda 0.5 \
  --speed-lambda 1.0 \
  --lte-floor 1e-3
```

Use `--dry-run` to inspect paths without loading a checkpoint or instantiating JiT.
