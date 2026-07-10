# Setup

This repository tracks source code only. Checkpoints, datasets, logs, outputs, generated images, plots, and review bundles are intentionally ignored.

## Submodules

```bash
bash scripts/setup_third_party.sh
bash scripts/setup_third_party.sh --print-only
bash scripts/setup_third_party.sh --check-only
```

Expected submodules:

- `third_party/JiT`
- `third_party/DeCo`
- `third_party/PixelGen` for experimental PixelGen Stage 4A runs
- `third_party/PixelDiT` (source only; runtime adapter pending)

Do not edit files inside `third_party` for BoundaryFlowCache changes.

## Environments

The scripts assume separate conda environments:

- `jit`: JiT generation, FID/IS evaluation, plotting, collection, and tests
- `deco`: DeCo generation
- `pixelgen`: experimental PixelGen generation

Use `conda run -n jit ...`, `conda run -n deco ...`, and `conda run -n pixelgen ...` in command plans.

## Checkpoints

Expected local paths:

- `ckpts/JiT/JiT-B-16-256/checkpoint-last.pth`
- `ckpts/DeCo/DeCo_XL.ckpt`
- `ckpts/PixelGen/PixelGen_XL_160ep.ckpt`

The files are not committed. If local layouts differ, pass `--jit-ckpt-dir`, `--deco-ckpt`, or `--pixelgen-ckpt` to the generation scripts.

## ImageNet

FID/IS evaluation can use either:

- a real ImageNet validation folder via `--real-dir`
- a compatible FID stats file via `--fid-stats`

Use this dry-run before preparing references:

```bash
conda run -n jit python scripts/prepare_stage4a_imagenet_reference.py \
  --imagenet-root /path/to/ILSVRC --dry-run
```

## GPU Policy

Generation scripts set `CUDA_VISIBLE_DEVICES` from `PFC_CUDA_DEVICES` if it is present. Set the variable explicitly before manual long runs:

```bash
export PFC_CUDA_DEVICES=0
```

The cleanup and validation commands do not launch long GPU jobs.

Run `scripts/preflight_experiments.py --strict` before any smoke. It checks submodules, checkpoint/stat structure without loading them, safe-map density, packages, GPU inventory, disk space, shell syntax/line endings, registry/config consistency, and stale absolute paths. It never downloads assets or runs generation. PixelGen has a runtime adapter; PixelDiT remains source-only.
