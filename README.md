# PixelFlowCache

PixelFlowCache is a research codebase for studying no-cache baselines, profiling, and later cache acceleration for pixel-space flow diffusion models. The current implementation status is **Stage 1 profiling infrastructure** for JiT and DeCo.

The project still does not implement block cache, token cache, branch cache, solver-aware cache, or calibration. Stage 1 only measures feature smoothness, velocity behavior, CFG behavior, and frequency behavior.

## Quickstart on this server

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
bash scripts/setup_third_party.sh
python scripts/inspect_repos.py
python scripts/run_stage0_smoke.py
pytest -q
bash scripts/run_official_jit_baseline.sh
bash scripts/run_official_deco_baseline.sh
```

## Stage 1 Profiling

Use one GPU by default:

```bash
export PFC_CUDA_DEVICES=0
bash scripts/run_profile_jit_stage1.sh
bash scripts/run_profile_deco_stage1.sh
```

Summarize and plot a run:

```bash
python scripts/summarize_stage1_profiles.py --run-dir logs/stage1/jit/<run_id>
python scripts/plot_stage1_profiles.py --run-dir logs/stage1/jit/<run_id>
```

See [docs/STAGE1_PROFILING.md](docs/STAGE1_PROFILING.md) for details.

The default server paths are:

- Project: `/mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache`
- ImageNet: `/mnt/iset/nfs-main/public/datasets/ILSVRC`
- JiT checkpoint dir: `ckpts/JiT/JiT-B-16-256`
- DeCo checkpoint: `ckpts/DeCo/imagenet256_epoch800.ckpt`

Use at most two GPUs for Stage 0 baselines and one GPU for Stage 1 profiling by default. The scripts run `nvidia-smi`, select visible devices via `CUDA_VISIBLE_DEVICES`, and honor `PFC_CUDA_DEVICES`.

See [docs/STAGE0_REPRO.md](docs/STAGE0_REPRO.md) for the full reproduction protocol.

Datasets, checkpoints, generated samples, logs, and other large binaries are intentionally ignored and should not be committed.
