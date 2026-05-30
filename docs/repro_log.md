# PixelFlowCache Repro Log

This file is updated by Stage 0 inspection, smoke, and baseline scripts.

## Stage 0 Inspect - 2026-05-30T08:34:42.797928+00:00

- root commit: 8903f5c598cc8f26242c99b57c23e7548c4a7cbf
- JiT commit: cbc743a2ada5e9762697da2c83f8c4f8379e8c17
- DeCo commit: 0792af05c9d8dce6c61e5636d136488f264065c7
- detected ImageNet root: /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC
- detected JiT checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth
- detected DeCo checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt
- selected GPUs for each run: not selected by inspect
- smoke test passed: not run by inspect
- JiT official debug baseline ran: not run by inspect
- DeCo official debug baseline ran: not run by inspect
- known blockers: see missing fields above

## Stage 0 Smoke - 2026-05-30T08:38:18.115424+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 0 Final Run - 2026-05-30T08:50:39Z

- root commit: 8903f5c598cc8f26242c99b57c23e7548c4a7cbf
- JiT commit: cbc743a2ada5e9762697da2c83f8c4f8379e8c17
- DeCo commit: 0792af05c9d8dce6c61e5636d136488f264065c7
- detected ImageNet root: /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC
- detected JiT checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth
- detected DeCo checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt
- selected GPUs for JiT: 0,1
- selected GPUs for DeCo: 0,1
- smoke test passed: yes, `python scripts/run_stage0_smoke.py`
- pytest status: passed, `pytest -q` in activated `jit` environment, 8 tests
- JiT official debug baseline ran: yes, 16 PNGs under `outputs/stage0/jit_official_debug`
- DeCo official debug baseline ran: yes, 16 PNGs plus `output.npz` under `outputs/stage0/deco_official_debug/exp_DeCo_256_XL_stage0_debug16`
- known blockers: none blocking Stage 0
- notes: JiT official code imports `torch_fidelity` at module import time; Stage 0 uses a project-local no-FID stub because FID is intentionally disabled for the 16-image debug run.
- notes: The first DeCo debug config used `max_num_instances: 16` with 1000 conditions, which DeCo rounded to 1000 samples. That run was stopped after 32 images and the debug config now restricts conditions to labels 0-15.

## Stage 0 Inspect - 2026-05-30T08:52:33.149783+00:00

- root commit: 8903f5c598cc8f26242c99b57c23e7548c4a7cbf
- JiT commit: cbc743a2ada5e9762697da2c83f8c4f8379e8c17
- DeCo commit: 0792af05c9d8dce6c61e5636d136488f264065c7
- detected ImageNet root: /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC
- detected JiT checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth
- detected DeCo checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt
- selected GPUs for each run: not selected by inspect
- smoke test passed: not run by inspect
- JiT official debug baseline ran: not run by inspect
- DeCo official debug baseline ran: not run by inspect
- known blockers: see missing fields above

## Stage 0 Completed - 2026-05-30T08:52:33Z

- root commit: 8903f5c598cc8f26242c99b57c23e7548c4a7cbf
- JiT commit: cbc743a2ada5e9762697da2c83f8c4f8379e8c17
- DeCo commit: 0792af05c9d8dce6c61e5636d136488f264065c7
- detected ImageNet root: /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC
- detected JiT checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth
- detected DeCo checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt
- selected GPUs for JiT: 0,1
- selected GPUs for DeCo: 0,1
- smoke/test status: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed in activated `jit` environment with 8 tests.
- JiT baseline status: completed with 16 debug PNGs under `outputs/stage0/jit_official_debug`.
- DeCo baseline status: completed with 16 debug PNGs and `output.npz` under `outputs/stage0/deco_official_debug/exp_DeCo_256_XL_stage0_debug16`.
- known blockers: none blocking Stage 0.

## Stage 0 Smoke - 2026-05-30T11:20:25.527483+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 0 Inspect - 2026-05-30T11:20:37.341072+00:00

- root commit: 43d287f57015b9545467724069df6955ce7b1338
- JiT commit: cbc743a2ada5e9762697da2c83f8c4f8379e8c17
- DeCo commit: 0792af05c9d8dce6c61e5636d136488f264065c7
- detected ImageNet root: /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC
- detected JiT checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth
- detected DeCo checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt
- selected GPUs for each run: not selected by inspect
- smoke test passed: not run by inspect
- JiT official debug baseline ran: not run by inspect
- DeCo official debug baseline ran: not run by inspect
- known blockers: see missing fields above

## Stage 0 Cleanup Patch - 2026-05-30T11:20:37Z

- current HEAD before cleanup patch: 43d287f57015b9545467724069df6955ce7b1338
- cleanup commit status: no cleanup commit was made in this turn; changes remain in the worktree/index for review.
- Stage 0 generation artifacts: produced before this cleanup patch at the previously recorded Stage 0 run commit 8903f5c598cc8f26242c99b57c23e7548c4a7cbf.
- third_party cleanup: PixelDiT and PixelGen were present as gitlinks in `HEAD` but not declared in `.gitmodules`; they are removed from Stage 0 tracking, leaving JiT and DeCo as the only configured submodules.
- validation after cleanup: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed in the activated `jit` environment with 8 tests; `python scripts/inspect_repos.py` passed.
## Stage 0 Smoke - 2026-05-30T11:41:43.830740+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Inspect - 2026-05-30T11:41:59.387804+00:00

- root commit: b1a6233c2eb329ac06e188c7acd10f7ed59c7aee
- JiT commit: cbc743a2ada5e9762697da2c83f8c4f8379e8c17
- DeCo commit: 0792af05c9d8dce6c61e5636d136488f264065c7
- detected ImageNet root: /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC
- detected JiT checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth
- detected DeCo checkpoint path: /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache/ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt
- selected GPUs for each run: not selected by inspect
- smoke test passed: not run by inspect
- JiT official debug baseline ran: not run by inspect
- DeCo official debug baseline ran: not run by inspect
- known blockers: see missing fields above

## Stage 1 Profiling - 2026-05-30T11:44:30Z

- current HEAD before Stage 1 patch commit: b1a6233c2eb329ac06e188c7acd10f7ed59c7aee
- Stage 1 implementation commit: this line originally described the pre-commit worktree state for the 2026-05-30T11:44:30Z run; Stage 1 was later committed and the verified committed HEAD is recorded below.
- scope: profiling infrastructure only; no cache acceleration, no block skipping, no token/block/branch/solver cache.
- GPU used for JiT profile: 0
- GPU used for DeCo profile: 0
- tests status: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed in activated `jit` environment with 15 tests.
- JiT profile status: completed.
- JiT run dir: `logs/stage1/jit/20260530T114242Z_seed0_steps10`
- JiT records: 240 feature records, 30 velocity records, 10 frequency records, 10 step records.
- JiT summary: 12 profiled blocks; smoothest mean rel-L2 modules include `blocks.0`, `blocks.1`, and `blocks.11`.
- DeCo profile status: completed.
- DeCo run dir: `logs/stage1/deco/20260530T114327Z_seed0_steps10`
- DeCo records: 1720 feature records, 30 velocity records, 10 frequency records, 10 step records.
- DeCo summary: 170 candidate modules; smoothest mean rel-L2 modules include `blocks.7.adaLN_modulation`, `dec_net.final_layer`, and `blocks.26.norm1`.
- generated figures:
  - `outputs/stage1/figures/jit_block_temporal_delta_heatmap.png`
  - `outputs/stage1/figures/jit_velocity_norm_by_step.png`
  - `outputs/stage1/figures/jit_xpred_amplification_by_step.png`
  - `outputs/stage1/figures/jit_frequency_ratio_by_step.png`
  - `outputs/stage1/figures/deco_module_temporal_delta_heatmap.png`
  - `outputs/stage1/figures/deco_velocity_norm_by_step.png`
  - `outputs/stage1/figures/deco_frequency_ratio_by_step.png`
- notes: `matplotlib` was installed into the `jit` conda environment for plotting; numpy was set to `1.24.4` to remain compatible with the existing scipy constraint.
- known blockers: none blocking Stage 1; DeCo emitted torch-dynamo graph-break warnings from hooks, expected for diagnostic profiling.
## Stage 0 Smoke - 2026-05-30T12:11:07.953030+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 1 Verified Commit - 2026-05-30T12:12:34Z

- current `git rev-parse HEAD`: ccadbd012db17c6a1838b64f6225a00bdf59d3f4
- cleanup patch status: not committed in this turn; the JiT rerun below was produced from the current HEAD plus this Stage 1 cleanup worktree patch.
- original JiT run dir: `logs/stage1/jit/20260530T114242Z_seed0_steps10`
- original DeCo run dir: `logs/stage1/deco/20260530T114327Z_seed0_steps10`
- cleanup JiT rerun dir: `logs/stage1/jit/20260530T121145Z_seed0_steps10`
- cleanup DeCo rerun status: pending; existing DeCo run remains available, but its feature logs predate the new split-batch and module-category recording.
- pytest status: `pytest -q` passed in the activated `jit` environment with 17 tests; `python scripts/run_stage0_smoke.py` passed.
- Stage 2 candidate export: generated `stage2_cache_candidates.csv` for the original JiT run, original DeCo run, and cleanup JiT rerun.
- logs/outputs status: `logs/` and `outputs/` are ignored and are not committed.
- scope: profiling cleanup only; no cache acceleration, block cache, token cache, branch cache, solver-aware cache, or calibration was implemented.

## Stage 1 Cleanup Verified - 2026-05-30T12:22:53Z

- current `git rev-parse HEAD`: 3c4f0a6d798c87dd5e31dc9812f110da4fa33272
- pytest status: `pytest -q` passed in the activated `jit` environment with 18 tests.
- smoke status: `python scripts/run_stage0_smoke.py` passed.
- FeatureRecorder cond/uncond keying fixed: yes; default `previous_key_fields=("module_name", "cfg_branch")` separates branch-local previous tensors.
- FeatureRecorder legacy keying: available via `previous_key_fields=("module_name",)`.
- FeatureRecorder `split_batch_dim0` support added: yes; it records whole-tensor stats plus uncond, cond, and cond-minus-uncond summaries without saving full tensors.
- `categorize_deco_module` added: yes; import check returned `block` for `blocks.7` and `norm_or_modulation` for `blocks.7.adaLN_modulation`.
- `export_stage2_cache_candidates.py` import/run status: passed on existing local JiT and DeCo Stage 1 run dirs.
- candidate export outputs checked:
  - `logs/stage1/jit/20260530T121145Z_seed0_steps10/stage2_cache_candidates.csv`
  - `logs/stage1/deco/20260530T114327Z_seed0_steps10/stage2_cache_candidates.csv`
- GPU profiling status in this cleanup turn: not rerun.
- artifact status: Stage 1 profiling artifacts remain ignored under `logs/` and `outputs/`; checkpoints, datasets, generated images, and large binaries are not committed.
- scope: bugfix cleanup only; no cache acceleration, Stage 2 cache implementation, block cache, token cache, branch cache, solver-aware cache, or calibration was implemented.
## Stage 0 Smoke - 2026-05-30T12:22:22.334913+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 1 Cleanup Patch-1 Verified - 2026-05-30
- current `git rev-parse HEAD`: 8a7d74ce5c12520f2ea18032e6c2f4fb75a2b7e6
- validation environment: `conda activate jit`
- `python -m py_compile pfc/profiling/feature_recorder.py`: passed
- `python -m py_compile pfc/profiling/module_selectors.py`: passed
- `python -m py_compile scripts/export_stage2_cache_candidates.py`: passed
- import check for `categorize_deco_module` and `FeatureRecorder`: passed
- `pytest -q`: passed with 18 tests
- scope: consistency cleanup only; no cache acceleration was implemented.
## Stage 0 Smoke - 2026-05-30T12:43:43.791494+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-05-30T12:53:52.719128+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-05-30T12:58:52.910176+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 2 Fixed-Interval JiT Block Cache - 2026-05-30T12:59:15Z

- current `git rev-parse HEAD`: c526139d82281f9b44d335d5a3b2fcd7105d410f
- implementation status: worktree patch pending; not committed in this turn.
- scope: fixed-interval whole-block cache for JiT only.
- not implemented: token cache, DeCo cache, branch-aware DeCo cache, frequency-aware cache, solver-aware cache, calibration.
- validation environment: `conda activate jit`
- CPU tests: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed with 33 tests.
- GPU policy: used one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- Stage 2 single JiT run: completed.
- Stage 2 single JiT run dir: `logs/stage2/jit/20260530T125647Z_seed0_steps20_i2_middle`
- Stage 2 single setting: middle blocks `[3, 4, 5, 6, 7, 8]`, interval 2, 8 samples, 20 Euler steps, one warmup run.
- Stage 2 single speedup: 1.321190418311986
- Stage 2 single no-cache latency: 1.2272571776993573 sec
- Stage 2 single cached latency: 0.9289025720208883 sec
- Stage 2 single cache hit rate: 0.5
- Stage 2 single cache stats: 480 total wrapped-block calls, 240 hits, 240 misses, 240 refreshes.
- Stage 2 single quality: same-seed MSE 0.02990465611219406; MAE 0.1380673199892044; rel-L2 0.42048725485801697.
- Stage 2 fast grid: completed.
- Stage 2 grid dir: `logs/stage2/jit_grid/20260530T125718Z_seed0_steps20`
- Stage 2 fast grid summary: `none/i1` speedup 0.9989671702807001 and rel-L2 0.0; `middle/i2` speedup 1.3247696088284628 and rel-L2 0.42048725485801697; `middle/i3` speedup 1.4637215804991033 and rel-L2 0.5512039065361023; `all/i2` speedup 1.8886902414919833 and rel-L2 0.25922855734825134.
- Stage 2 plots: generated under ignored `outputs/stage2/figures/`.
- DeCo Stage 2 status: candidate export only; no DeCo cache was implemented.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.

## Stage 2 Fixed-Interval JiT Block Cache Verified Commit - 2026-05-30
- current `git rev-parse HEAD`: <填当前HEAD>
- validation environment: `conda activate jit`
- `python scripts/run_stage0_smoke.py`: passed
- `pytest -q`: passed with 33 tests
- scope: fixed-interval whole-block JiT cache only.
- not implemented: token cache, DeCo cache, frequency-aware policy, solver-aware policy, calibration.
- Stage 2 single run dir: `logs/stage2/jit/20260530T125647Z_seed0_steps20_i2_middle`
- Stage 2 grid dir: `logs/stage2/jit_grid/20260530T125718Z_seed0_steps20`
- note: logs/outputs/previews/figures are ignored and not committed.
