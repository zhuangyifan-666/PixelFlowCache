# BoundaryFlowCache / PixelFlowCache

BoundaryFlowCache is a compact implementation of boundary-aware caching for pixel-space flow diffusion models. The retained code focuses on final 50-step generation, BoundaryFlowCache acceleration, reduced-step baselines, and FID/IS evaluation.

The codebase is being refactored into PixBFC, a general adapter-based framework for pixel-space diffusion and flow models. JiT and DeCo remain the supported models; new models should add a boundary adapter instead of changing the core cache state.

## Supported Models

- JiT-B/16 ImageNet-256
- DeCo ImageNet-256

## Main 50k Results

| model | method | steps | speedup vs no-cache | FID | IS |
|---|---|---:|---:|---:|---:|
| JiT | no_cache_50 | 50 | 1.000 | 4.173 | 279.086 |
| JiT | bfc_quality_t02_08 | 50 | 1.462 | 4.247 | 276.805 |
| JiT | bfc_speed_t02_10 | 50 | 1.757 | 4.287 | 277.536 |
| JiT | reduced_steps_35 | 35 | 1.557 | 4.677 | 283.538 |
| JiT | reduced_steps_30 | 30 | 1.757 | 5.179 | 291.259 |
| DeCo | no_cache_50 | 50 | 1.000 | 2.057 | 316.068 |
| DeCo | bfc_all_candidates_t02_10 | 50 | 1.652 | 2.359 | 307.574 |
| DeCo | bfc_backbone_plus_final_t02_10 | 50 | 1.559 | 2.359 | 307.574 |
| DeCo | reduced_steps_30 | 30 | 1.602 | 2.671 | 304.857 |

The DeCo `reduced_steps_35` run had a timing anomaly and is documented in [docs/RESULTS_50K.md](docs/RESULTS_50K.md), but it is not used as the main speed baseline.

## Installation

```bash
git submodule update --init --recursive third_party/JiT third_party/DeCo
```

Expected local assets:

- JiT checkpoint: `ckpts/JiT/JiT-B-16-256/checkpoint-last.pth`
- DeCo checkpoint: `ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt`
- ImageNet root: a local ImageFolder-compatible ILSVRC directory
- Conda envs: `jit` for JiT/evaluation, `deco` for DeCo

Checkpoints, datasets, logs, outputs, and result bundles are ignored and are not part of the repository.

## Inference

JiT example:

```bash
conda run -n jit python scripts/run_jit_stage4a_generate.py \
  --method bfc_speed_t02_10 \
  --num-images 1000 \
  --batch-size 8 \
  --run-id demo_n1000_seed0 \
  --save-png \
  --no-save-npz
```

DeCo example:

```bash
conda run -n deco python scripts/run_deco_stage4a_generate.py \
  --method bfc_all_candidates_t02_10 \
  --num-images 1000 \
  --batch-size 4 \
  --run-id demo_n1000_seed0 \
  --save-png \
  --no-save-npz
```

Print command plans without running generation:

```bash
bash scripts/print_stage4a_full_50k_commands.sh
```

## FID And IS

Prepare an ImageNet reference folder if needed:

```bash
conda run -n jit python scripts/prepare_stage4a_imagenet_reference.py --dry-run
```

Evaluate a generated folder:

```bash
conda run -n jit python scripts/evaluate_stage4a_fid.py \
  --fake-dir outputs/stage4a/full_generation/jit/demo_n1000_seed0/bfc_speed_t02_10/images \
  --real-dir /path/to/imagenet/val \
  --backend auto \
  --metrics fid,is \
  --out logs/stage4a/fid/demo_n1000_seed0/jit/bfc_speed_t02_10/fid_results.json
```

Collect and plot existing results:

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

## Repository Layout

- `pfc/core`: generic PixBFC boundary specs, model adapter interface, scheduler interface, runtime container, and registry
- `pfc/adapters`: JiT and DeCo boundary adapters
- `pfc/cache`: cache state, fixed-interval policy, cached module wrappers, JiT/DeCo BoundaryFlowCache wrappers
- `pfc/eval`: method presets, label scheduling, generation IO, JiT/DeCo runtime helpers
- `pfc/diagnostics`: lightweight tensor and frequency diagnostics used by retained runtime code
- `scripts`: final generation, FID/IS, ImageNet reference, command planning, collection, plotting, and submodule setup utilities
- `docs`: final method, setup, inference/FID, 50k results, and cleanup manifest

## Generalization To New Pixel Diffusion Models

PixBFC represents model-specific details through a `PixelDiffusionModelAdapter`. An adapter declares the prediction parameterization, lists cacheable boundaries, selects a default boundary set, and installs wrappers on an already constructed model. The cache state and fixed-window scheduler stay model-independent.

See [docs/PIXBFC_GENERALIZATION.md](docs/PIXBFC_GENERALIZATION.md) for the abstraction and the checklist for adding a future PixelGen or PixelDiT adapter.

## Not Included

- Historical smoke, profiling, and exploratory ablation scripts
- Checkpoints, datasets, generated samples, logs, plots, or uploaded result bundles
- Token cache, adaptive online policy, solver-aware cache, calibration, or frequency-aware cache

See [docs/METHOD.md](docs/METHOD.md), [docs/PIXBFC_GENERALIZATION.md](docs/PIXBFC_GENERALIZATION.md), [docs/SETUP.md](docs/SETUP.md), and [docs/INFERENCE_AND_FID.md](docs/INFERENCE_AND_FID.md) for details.

## Adapted SeaCache-Style Baseline

The codebase also includes independent adapted `teacache_style` and `seacache_style` dynamic-cache baselines for JiT and DeCo. These use the current image/state `x_t` as a low-overhead proxy and reuse BoundaryFlowCache cache units for fair comparison. They are not official SeaCache results.

Print single-GPU threshold sweep commands:

```bash
bash scripts/print_stage4a_seacache_theta006_commands.sh
```

The final adapted SeaCache-style baseline target is `theta/delta = 0.06`. Numerical results should be added only after the theta=0.06 50k runs and FID/IS collection complete. See [docs/BASELINE_SEACACHE_STYLE.md](docs/BASELINE_SEACACHE_STYLE.md) for the method details and suggested protocol.
