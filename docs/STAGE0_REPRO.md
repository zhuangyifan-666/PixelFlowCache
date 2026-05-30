# Stage 0 Reproducibility

Stage 0 establishes official no-cache reproducibility for JiT and DeCo on this server. It also adds a minimal unified `pfc` interface, CPU smoke tests, repo/checkpoint inspection, and reproducible log locations.

Stage 0 is not cache acceleration. It does not implement block cache, token cache, branch cache, solver-aware cache, PixelDiT, PixelGen, or FID-scale evaluation.

## Server Paths

- Project: `/mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache`
- Shared ImageNet base: `/mnt/iset/nfs-main/public/datasets/ILSVRC`
- Usable ImageFolder root detected on this server: `/mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC`
- Expected JiT checkpoint dir: `ckpts/JiT/JiT-B-16-256`
- Expected JiT checkpoint file: `ckpts/JiT/JiT-B-16-256/checkpoint-last.pth`
- Expected DeCo checkpoint: `ckpts/DeCo/imagenet256_epoch800.ckpt`

The inspection and run scripts auto-detect nested ImageFolder roots by looking for `train/` and `val/`.

## Environments

- JiT: `conda activate jit`
- DeCo: `conda activate deco`

The shell wrappers prefer `conda run -n jit` and `conda run -n deco`. If that is unavailable, they activate the environment inside the script.

## GPU Policy

- Use at most two GPUs by default.
- Always run `nvidia-smi` before GPU jobs.
- Restrict jobs with `CUDA_VISIBLE_DEVICES`.
- For JiT multi-GPU launch, `PFC_NPROC` must match the number of visible GPUs.

Useful overrides:

```bash
export PFC_CUDA_DEVICES=0,1
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PFC_NPROC=2
export PFC_NUM_IMAGES=16
export PFC_GEN_BSZ=8
```

## Commands

Inspect third-party repos, ImageNet root, and checkpoints:

```bash
python scripts/inspect_repos.py
```

Run CPU-only smoke checks:

```bash
python scripts/run_stage0_smoke.py
pytest -q
```

Run the official JiT debug baseline:

```bash
bash scripts/run_official_jit_baseline.sh
```

The JiT script uses the official JiT parser/main and `--evaluate_gen`, but the default entrypoint disables TensorBoard writer creation to avoid the official FID path and keep the 16 debug samples.

Run the official DeCo debug baseline:

```bash
bash scripts/run_official_deco_baseline.sh
```

The DeCo script uses `configs/deco_stage0_debug.yaml`, a project-owned copy of the official ImageNet-256 config with `max_num_instances: 16` and `pred_batch_size: 8`.

## Expected Logs

- `logs/stage0/repo_status.json`
- `logs/stage0/smoke_summary.json`
- `logs/stage0/jit_official_baseline.log`
- `logs/stage0/deco_official_baseline.log`

## Expected Outputs

- `outputs/stage0/jit_official_debug`
- `outputs/stage0/deco_official_debug`

## Troubleshooting

Missing conda: load the server conda installation, then rerun the same script.

Missing checkpoint: run `python scripts/inspect_repos.py` and check the expected JiT and DeCo paths printed there.

ImageNet root mismatch: set `PFC_IMAGENET_PATH` to the directory whose direct child is `train/`. On this server that is usually `/mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC`.

DINOv2 missing for DeCo: if predict instantiates the training encoder, it may need `/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`. The scripts warn but do not download it.

CUDA OOM: reduce `PFC_GEN_BSZ` for JiT or `data.pred_batch_size` in `configs/deco_stage0_debug.yaml`.

Busy GPUs: set `PFC_CUDA_DEVICES` explicitly after inspecting `nvidia-smi`.

