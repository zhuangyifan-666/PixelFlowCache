# PixelFlowCache

PixelFlowCache is a research codebase for studying no-cache baselines, profiling, and later cache acceleration for pixel-space flow diffusion models. The current implementation status is **Stage 0 / Week 1**: official no-cache reproducibility for JiT and DeCo plus a small unified `pfc` interface for CPU smoke tests.

Stage 0 does not implement block cache, token cache, branch cache, or solver-aware cache logic. Those belong after the official baselines and profiling infrastructure are stable.

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

The default server paths are:

- Project: `/mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache`
- ImageNet: `/mnt/iset/nfs-main/public/datasets/ILSVRC`
- JiT checkpoint dir: `ckpts/JiT/JiT-B-16-256`
- DeCo checkpoint: `ckpts/DeCo/imagenet256_epoch800.ckpt`

Use at most two GPUs by default. The baseline scripts run `nvidia-smi`, select visible devices via `CUDA_VISIBLE_DEVICES`, and honor `PFC_CUDA_DEVICES`.

See [docs/STAGE0_REPRO.md](docs/STAGE0_REPRO.md) for the full reproduction protocol.

Datasets, checkpoints, generated samples, logs, and other large binaries are intentionally ignored and should not be committed.

