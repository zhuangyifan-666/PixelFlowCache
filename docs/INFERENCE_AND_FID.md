# Inference And FID

This page describes the retained final utilities. The command-plan scripts print commands; they do not run generation or FID by themselves.

## JiT Generation

```bash
conda run -n jit python scripts/run_jit_stage4a_generate.py \
  --method bfc_speed_t02_10 \
  --num-images 1000 \
  --batch-size 8 \
  --seed 0 \
  --run-id demo_n1000_seed0 \
  --save-png \
  --no-save-npz
```

Use `--dry-run` to validate paths without loading the model.

## DeCo Generation

```bash
conda run -n deco python scripts/run_deco_stage4a_generate.py \
  --method bfc_all_candidates_t02_10 \
  --num-images 1000 \
  --batch-size 4 \
  --seed 0 \
  --run-id demo_n1000_seed0 \
  --save-png \
  --no-save-npz
```

Use `--dry-run` to validate checkpoint and config paths without loading the model.

## PixelGen Generation

PixelGen support is experimental/in progress for future ImageNet-256 Stage 4A runs. The commands below are entry points only; no PixelGen 50k FID/IS results are claimed in this repository.

```bash
conda run -n pixelgen python scripts/run_pixelgen_stage4a_generate.py \
  --method bfc_quality_t02_08 \
  --num-images 8 \
  --batch-size 2 \
  --seed 0 \
  --run-id dryrun_pixelgen \
  --pixelgen-ckpt ckpts/PixelGen/PixelGen_XL_160ep.ckpt \
  --amp-dtype bf16 \
  --dry-run
```

Print the PixelGen 50k command plan without running generation:

```bash
bash scripts/print_stage4a_pixelgen_50k_commands.sh
```

Print the adapted PixelGen SeaCache-style baseline command without running generation:

```bash
bash scripts/print_stage4a_pixelgen_50k_commands.sh --methods seacache_style
```

Dry-run the adapted PixelGen SeaCache-style entry point without loading a checkpoint:

```bash
conda run -n pixelgen python scripts/run_pixelgen_stage4a_generate.py \
  --method seacache_style \
  --num-images 8 \
  --batch-size 2 \
  --run-id dryrun_pixelgen_seacache \
  --pixelgen-ckpt ckpts/PixelGen/PixelGen_XL_160ep.ckpt \
  --amp-dtype bf16 \
  --dynamic-cache-threshold 0.06 \
  --sea-beta 2.0 \
  --sea-proxy-downsample 64 \
  --dry-run
```

This is an adapted SeaCache-style dynamic-cache baseline using PixelGen `x_t` proxy decisions; it is not the official SeaCache implementation and should not be reported until its generation and FID/IS evaluation have been run.

## Command Plans

```bash
bash scripts/print_stage4a_smoke_commands.sh
bash scripts/print_stage4a_proxy_fid_commands.sh
bash scripts/print_stage4a_full_50k_commands.sh
bash scripts/print_stage4a_pixelgen_50k_commands.sh
bash scripts/print_jit_safe_1000_commands.sh --dry-run
```

For a custom plan:

```bash
conda run -n jit python scripts/run_stage4a_full_eval_plan.py \
  --models jit,deco \
  --num-images 50000 \
  --out-script /tmp/pfc_stage4a_50k.sh
```

Review generated command plans before launching long jobs.

## JiT Safe-BFC Proxy Plan

`docs/JIT_SAFE_BFC_1000.md` describes the JiT-B/16 ImageNet-256 Safe-BFC 1000-image proxy workflow. The flow is calibration, 1000-image generation, FID/IS per method folder, paired PSNR/SSIM/LPIPS/rel_l2 against the same-run `no_cache_50` images, and summary collection.

The 1000-image workflow is proxy-only and should not be interpreted as final 50k FID/IS.

## Reference Preparation

```bash
conda run -n jit python scripts/prepare_stage4a_imagenet_reference.py --dry-run
```

Pass `--source-root`, `--out-dir`, and `--limit` as needed. The script is designed to prepare a real-image folder for FID/IS tools; it does not belong in git.

## FID/IS Evaluation

```bash
conda run -n jit python scripts/evaluate_stage4a_fid.py \
  --fake-dir outputs/stage4a/full_generation/jit/demo_n1000_seed0/bfc_speed_t02_10/images \
  --real-dir /path/to/imagenet/val \
  --backend auto \
  --metrics fid,is \
  --batch-size 64 \
  --out logs/stage4a/fid/demo_n1000_seed0/jit/bfc_speed_t02_10/fid_results.json
```

Use the real `torch_fidelity`, `cleanfid`, or `torchmetrics` package. Do not use `scripts/jit_stubs` as an evaluation backend.

## Collection And Plotting

```bash
conda run -n jit python scripts/collect_stage4a_fid_results.py \
  --root outputs/stage4a/full_generation \
  --fid-root logs/stage4a/fid \
  --run-id stage4a_n50000_seed0 \
  --num-images 50000 \
  --out-dir logs/stage4a/summary/stage4a_n50000_seed0_clean

conda run -n jit python scripts/plot_stage4a_full_eval.py \
  --summary-dir logs/stage4a/summary/stage4a_n50000_seed0_clean \
  --num-images 50000
```

The collector computes speedup within `(model, run_id, num_images)` groups so smoke and full-generation rows are not mixed.
