# PixelFlowCache

PixelFlowCache is a research codebase for studying no-cache baselines, profiling, and cache acceleration for pixel-space flow diffusion models. The current implementation status is **Stage 2C JiT window ablation and full-probe diagnostics**.

Stage 2 implements the first actual compute-skipping baseline: fixed-interval whole-block cache for JiT Transformer blocks. Stage 2B extends that baseline with timestep windows, layer-group sweeps, repeated timing, and velocity-error diagnostics. Stage 2C focuses on controlled JiT window ablations and local-error probes. The project still does not implement token cache, DeCo cache, adaptive online policy, solver-aware cache, frequency-aware cache, or calibration.

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

## Stage 2 Fixed-Interval JiT Block Cache

Use one GPU by default:

```bash
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2_cache.sh
```

Run the fast grid:

```bash
export PFC_STAGE2_GRID_FAST=1
bash scripts/run_jit_stage2_grid.sh
```

Plot grid results:

```bash
python scripts/plot_stage2_jit_cache.py --grid-dir logs/stage2/jit_grid/<run_id>
```

See [docs/STAGE2_FIXED_BLOCK_CACHE.md](docs/STAGE2_FIXED_BLOCK_CACHE.md) for details.

## Stage 2B Timestep Windows And Diagnostics

Use one GPU by default:

```bash
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2b_cache.sh
```

Run the fast Stage 2B sweep:

```bash
export PFC_STAGE2B_SWEEP_FAST=1
bash scripts/run_jit_stage2b_sweep.sh
```

Plot Stage 2B sweep results:

```bash
python scripts/plot_stage2b_jit.py --sweep-dir logs/stage2b/jit_sweep/<run_id>
```

See [docs/STAGE2B_TIMESTEP_WINDOW_AND_DIAGNOSTICS.md](docs/STAGE2B_TIMESTEP_WINDOW_AND_DIAGNOSTICS.md) for details.

## Stage 2C JiT Window Ablation And Probe

Use one GPU by default:

```bash
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2c_window_ablation.sh
bash scripts/run_jit_stage2c_probe.sh
```

Plot Stage 2C results:

```bash
python scripts/plot_stage2c_jit.py \
  --window-dir logs/stage2c/jit_window_ablation/<run_id> \
  --probe-dir logs/stage2c/jit_probe/<run_id>
```

Optional validation:

```bash
bash scripts/run_jit_stage2c_validate.sh
```

See [docs/STAGE2C_WINDOW_ABLATION_AND_PROBE.md](docs/STAGE2C_WINDOW_ABLATION_AND_PROBE.md) and [docs/STAGE2C_BOUNDARY_CACHE_OBSERVATION.md](docs/STAGE2C_BOUNDARY_CACHE_OBSERVATION.md) for details.

The default server paths are:

- Project: `/mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache`
- ImageNet: `/mnt/iset/nfs-main/public/datasets/ILSVRC`
- JiT checkpoint dir: `ckpts/JiT/JiT-B-16-256`
- DeCo checkpoint: `ckpts/DeCo/imagenet256_epoch800.ckpt`

Use at most two GPUs for Stage 0 baselines and one GPU for Stage 1/Stage 2 runs by default. The scripts run `nvidia-smi`, select visible devices via `CUDA_VISIBLE_DEVICES`, and honor `PFC_CUDA_DEVICES`.

See [docs/STAGE0_REPRO.md](docs/STAGE0_REPRO.md) for the full reproduction protocol.

Datasets, checkpoints, generated samples, logs, and other large binaries are intentionally ignored and should not be committed.
