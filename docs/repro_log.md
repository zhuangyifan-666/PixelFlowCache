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
## Stage 0 Smoke - 2026-05-30T13:24:23.403281+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-05-30T13:33:41.848587+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 2B Timestep Window And Diagnostics - 2026-05-30T13:37:21Z

- current `git rev-parse HEAD`: 9a6c07d370dadb68da94500bbad2632c7bdae8a5
- implementation status: worktree patch pending; not committed in this turn.
- scope: JiT fixed-interval whole-block cache analysis only.
- not implemented: token cache, DeCo cache, PixelDiT/PixelGen, adaptive online policy, calibration, final PixelFlowCache method.
- new features: active timestep/step windows in `FixedIntervalCachePolicy`; Stage 2B layer specs; per-batch cache entry clearing; repeated timing; velocity/frequency error diagnostics; optional full-on-cache-state probe.
- validation environment: `conda activate jit`
- CPU tests: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed with 45 tests.
- GPU policy: used one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- Stage 2B single run: completed.
- Stage 2B single run dir: `logs/stage2b/jit/20260530T133416Z_seed0_steps20_i2_all`
- Stage 2B single setting: all 12 JiT blocks, interval 2, active t window `[0.1, 0.8)`, 8 samples, 20 Euler steps, 3 timing repeats, 1 warmup.
- Stage 2B single median speedup: 1.4877936898875264
- Stage 2B single no-cache median latency: 1.2261524298228323 sec
- Stage 2B single cached median latency: 0.8241414371877909 sec
- Stage 2B single cache hit rate: 0.35
- Stage 2B single cache stats: 960 total wrapped-block calls, 336 hits, 624 misses, 624 refreshes.
- Stage 2B single quality: same-seed MSE 0.0011018130462616682; MAE 0.021136298775672913; rel-L2 0.08071035146713257; PSNR 35.5995208090011.
- Stage 2B single diagnostics: `step_error_stats.jsonl` written with 40 records.
- Stage 2B fast sweep: completed.
- Stage 2B sweep dir: `logs/stage2b/jit_sweep/20260530T133455Z_seed0_steps20`
- Stage 2B best speed-quality row by lowest rel-L2: `all`, interval 2, active t `[0.1, 0.8)`, speedup 1.4890113989899492, rel-L2 0.08071035146713257.
- Stage 2B fastest row: `all`, interval 2, active t `[0.1, 1.0)`, speedup 1.7517469597350046, rel-L2 0.09124936908483505.
- Active t max finding: reducing `active_t_max` from 1.0 to 0.8 improved rel-L2 from 0.09124936908483505 to 0.08071035146713257, with expected speedup reduction from 1.7517469597350046 to 1.4890113989899492.
- Layer group finding in fast sweep: `all` and `suffix:6` had the lowest rel-L2 at 0.08071035146713257; `all` was faster than `suffix:6`, `prefix:6`, and `middle` under the `[0.1, 0.8)` window.
- Stage 2B plots: generated under ignored `outputs/stage2b/figures/`.
- Stage 2B validation run: not run in this turn; fast single and fast sweep completed successfully.
- weights/data status: no new weights downloaded; no datasets scanned for Stage 2B.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.

## Stage 0 Smoke - 2026-05-31T05:40:33.419217+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 0 Smoke - 2026-05-31T05:49:35.669045+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 2C JiT Window Ablation And Probe - 2026-05-31T05:50:36Z

- current `git rev-parse HEAD`: cd45f83d71f1ebd5e9c886c9154a3b3122bf40da
- implementation status: worktree patch pending; not committed in this turn.
- scope: JiT fixed-interval whole-block cache window ablation and local full-probe diagnostics only.
- not implemented: token cache, DeCo cache, adaptive online policy, solver-aware cache, calibration, final PixelFlowCache method.
- validation environment: `conda activate jit`
- CPU tests: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed with 51 tests.
- GPU policy attempted: one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- `nvidia-smi` status: failed with driver communication error, so Stage 2C GPU runs were not started.
- Stage 2C window ablation status: blocked before model load; see ignored log `logs/stage2c/jit_stage2c_window_ablation_stdout.log`.
- Stage 2C probe status: blocked before model load; see ignored log `logs/stage2c/jit_stage2c_probe_stdout.log`.
- Stage 2C validation status: not run because the required GPU check failed.
- Plot status: empty-input check passed; full plots were not generated because no Stage 2C window/probe run directories were produced.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.
- next command after GPU driver is available: `export PFC_CUDA_DEVICES=0; bash scripts/run_jit_stage2c_window_ablation.sh && bash scripts/run_jit_stage2c_probe.sh`

## Stage 2C JiT Window Ablation And Probe Results - 2026-05-31T06:15:31Z

- current `git rev-parse HEAD`: cd45f83d71f1ebd5e9c886c9154a3b3122bf40da
- GPU policy used: one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- Stage 2C window ablation run dir: `logs/stage2c/jit_window_ablation/20260531T055834Z_seed0_steps20`.
- Stage 2C window ablation rows: 10 data rows plus header in `window_ablation_results.csv`.
- Best Stage 2C debug same-seed rel-L2 row: all blocks, interval 2, active t `[0.2, 0.8)`, speedup 1.3894, hit rate 0.3000, rel-L2 0.032933.
- Fastest Stage 2C debug row: all blocks, interval 2, active t `[0.1, 1.0)`, speedup 1.7304, hit rate 0.4500, rel-L2 0.091249.
- Boundary finding: increasing `active_t_min` from 0.1 to 0.2 improved rel-L2 from 0.080710 to 0.032933 but reduced speedup from 1.4880 to 1.3894.
- Boundary finding: increasing `active_t_max` from 0.7 to 1.0 improved speedup from 1.3934 to 1.7304 but worsened rel-L2 from 0.079128 to 0.091249.
- Interval finding on the selected debug best window `[0.2, 0.8)`: interval 2 had rel-L2 0.032933 and speedup 1.3894; interval 3 had rel-L2 0.080137 and speedup 1.6055.
- Stage 2C probe run dir: `logs/stage2c/jit_probe/20260531T060052Z_seed0_steps20_i2_all`.
- Stage 2C probe setting: all blocks, interval 2, active t `[0.1, 0.8)`, 4 samples, 20 Euler steps.
- Stage 2C probe result: speedup 1.4876, hit rate 0.3500, final rel-L2 0.083956.
- Probe diagnostics: mean trajectory rel-L2 0.044883; mean local probe rel-L2 0.010539; max local probe rel-L2 0.065953; amplification/probe correlation -0.233014.
- Probe dominance: accumulated trajectory drift dominated local probe error.
- Stage 2C validation run dir: `logs/stage2c/jit_validate/20260531T060229Z_seed0_steps50`.
- Stage 2C validation rows: no-cache reference, all/i2 `[0.1,0.8)`, all/i2 `[0.1,1.0)`, all/i3 `[0.1,0.8)`.
- Stage 2C validation result: all/i2 `[0.1,0.8)` had speedup 1.4687 and rel-L2 0.040435; all/i2 `[0.1,1.0)` had speedup 1.7079 and rel-L2 0.043077; all/i3 `[0.1,0.8)` had speedup 1.7060 and rel-L2 0.070958.
- Validation caveat: the debug-best `[0.2,0.8)` window was not included in this validation run because `PFC_STAGE2C_BEST_CONFIG_JSON` was not provided.
- Stage 2C figures generated under ignored `outputs/stage2c/figures/`.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.
## Stage 0 Smoke - 2026-05-31T05:58:19.433924+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-06-01T05:29:07.866263+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 2D Implementation Start - 2026-06-01T05:36:28Z

- current `git rev-parse HEAD`: 9f83614b9bbb0359374a578da030ebb2485a6e73
- implementation status: worktree patch pending; not committed in this turn.
- scope: JiT fixed whole-backbone best-window validation, seed stability, and first-hit delay ablation.
- not implemented: token cache, DeCo cache, adaptive online policy, solver-aware cache, calibration, final PixelFlowCache method.
- validation environment: `conda run -n jit`
- initial CPU checks: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed with 51 tests before edits.
- planned validation configs include all/i2 `[0.2,0.8)` and all/i2 `[0.2,1.0)`.
- artifact status: no new weights downloaded; logs, outputs, checkpoints, datasets, generated images, and large binaries remain ignored.

## Stage 2D Implementation Check - 2026-06-01T05:39:45Z

- current `git rev-parse HEAD`: 9f83614b9bbb0359374a578da030ebb2485a6e73
- implementation status: worktree patch pending; not committed in this turn.
- new policy option: `active_window_warmup_refreshes`, default 0, per module and CFG branch.
- validation environment: `conda run -n jit`
- `python scripts/run_stage0_smoke.py`: passed.
- `pytest -q`: passed with 58 tests.
- Stage 2D validation status: attempted with one GPU via `PFC_CUDA_DEVICES=0`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage2d/jit_stage2d_validate_best_windows_stdout.log`.
- Stage 2D first-hit delay status: attempted with one GPU via `PFC_CUDA_DEVICES=0`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage2d/jit_stage2d_first_hit_delay_stdout.log`.
- Stage 2D seed sweep status: attempted, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage2d/jit_stage2d_seed_sweep_stdout.log`.
- Plot status: empty-input check passed; full plots require completed Stage 2D run directories.
- next GPU-visible commands: `export PFC_CUDA_DEVICES=0; bash scripts/run_jit_stage2d_validate_best_windows.sh; bash scripts/run_jit_stage2d_first_hit_delay.sh; bash scripts/run_jit_stage2d_seed_sweep.sh`.
- artifact status: no new weights downloaded; logs, outputs, checkpoints, datasets, generated images, and large binaries remain ignored.

## Stage 2D Implementation Check - 2026-06-01T05:43:28Z

- current `git rev-parse HEAD`: 9f83614b9bbb0359374a578da030ebb2485a6e73
- implementation status: worktree patch pending; not committed in this turn.
- final CPU checks after warmup semantics adjustment: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed with 58 tests.
- first-hit delay semantics: warmup refreshes are consumed on candidate reuse steps inside the active window, so warmup=1 delays the first cache hit for the default interval-2 `[0.1,0.8)` setting.
- GPU status in this session: `nvidia-smi` still fails with driver communication error, so Stage 2D validation, first-hit delay, and seed sweep remain pending for a GPU-visible shell.
- artifact status: no new weights downloaded; logs, outputs, checkpoints, datasets, generated images, and large binaries remain ignored.

## Stage 2D JiT Validation And Seed Stability Results - 2026-06-01T06:16:12Z

- current `git rev-parse HEAD`: 9f83614b9bbb0359374a578da030ebb2485a6e73
- GPU policy used: one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- Stage 2D validation run dir: `logs/stage2d/jit_validate_best/20260601T054448Z_seed0_steps50`.
- Stage 2D validation rows: no-cache reference plus all/i2 `[0.1,0.8)`, all/i2 `[0.1,1.0)`, all/i2 `[0.2,0.8)`, all/i2 `[0.2,1.0)`, and all/i3 `[0.2,0.8)`.
- Best 50-step validation quality: all/i2 `[0.2,0.8)`, speedup 1.3913, hit rate 0.3000, rel-L2 0.019059, MSE 0.00007648, PSNR 47.1850.
- Fastest 50-step validation row: all/i2 `[0.1,1.0)`, speedup 1.7097, hit rate 0.4400, rel-L2 0.043077.
- Speed-quality point: all/i2 `[0.2,1.0)` reached speedup 1.6079, hit rate 0.4000, rel-L2 0.024648, MSE 0.00012792, PSNR 44.9512.
- 50-step comparison: `[0.2,0.8)` beat `[0.1,0.8)` on rel-L2, 0.019059 vs 0.040435, at lower speedup, 1.3913 vs 1.4677.
- First-hit delay run dir: `logs/stage2d/jit_first_hit_delay/20260601T055631Z_seed0_steps20`.
- First-hit delay result for all/i2 `[0.1,0.8)`: warmup 0 rel-L2 0.080710 speedup 1.4904; warmup 1 rel-L2 0.032933 speedup 1.3975; warmup 2 rel-L2 0.022872 speedup 1.3063.
- First-hit delay finding: delaying the first active-window reuse improves quality substantially, with expected speedup loss.
- Seed sweep run dir: `logs/stage2d/jit_seed_sweep/20260601T055730Z_seed0_steps50`.
- Seed sweep seeds: 0, 1, 2; 16 samples, 50 steps, 2 timing repeats.
- Seed sweep all/i2 `[0.1,0.8)`: speedup mean/std 1.4699/0.0011; rel-L2 mean/std 0.045155/0.006983; MSE mean/std 0.00041215/0.00010498.
- Seed sweep all/i2 `[0.1,1.0)`: speedup mean/std 1.7068/0.0047; rel-L2 mean/std 0.047799/0.006838; MSE mean/std 0.00046073/0.00010783.
- Seed sweep all/i2 `[0.2,0.8)`: speedup mean/std 1.3919/0.0008; rel-L2 mean/std 0.022393/0.006337; MSE mean/std 0.00010614/0.00005609.
- Seed sweep all/i2 `[0.2,1.0)`: speedup mean/std 1.6047/0.0009; rel-L2 mean/std 0.027789/0.005509; MSE mean/std 0.00015824/0.00005770.
- Stage 2D figures generated under ignored `outputs/stage2d/figures/`.
- artifact status: no new weights downloaded; logs, outputs, checkpoints, datasets, generated images, and large binaries remain ignored.
## Stage 0 Smoke - 2026-06-01T05:38:00.196263+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-06-01T05:43:11.886500+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-06-01T06:40:02.233554+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 3A Implementation Start - 2026-06-01T06:51:10Z

- current `git rev-parse HEAD`: 619c4a0a3fced8346dcec4effec157590671afbd
- implementation status: worktree patch pending; not committed in this turn.
- scope: JiT BackboneCache preset benchmark versus reduced-step no-cache baselines.
- not implemented: token cache, DeCo cache, adaptive online policy, solver-aware cache, calibration, final PixelFlowCache method.
- validation environment: `conda run -n jit`
- initial CPU checks: `python scripts/run_stage0_smoke.py` passed; `pytest -q` passed with 58 tests before edits.
- planned benchmark: cache presets `quality_t02_08`, `speed_t02_10`, first-hit-delay variants, `aggressive_i3_t02_08`, plus reduced-step no-cache 30/35/40.
- artifact status: no new weights downloaded; logs, outputs, checkpoints, datasets, generated images, and large binaries remain ignored.

## Stage 3A Implementation Check - 2026-06-01T06:54:22Z

- current `git rev-parse HEAD`: 619c4a0a3fced8346dcec4effec157590671afbd
- implementation status: worktree patch pending; not committed in this turn.
- new preset abstraction: `pfc/cache/backbone_cache_presets.py` with required JiT BackboneCache presets.
- benchmark scripts added: reduced-step baseline, unified BackboneCache benchmark, optional 32-sample subset, plotting, and paper-table generation.
- validation environment: `conda run -n jit`
- `python scripts/run_stage0_smoke.py`: passed.
- `pytest -q`: passed with 67 tests.
- Stage 3A benchmark status: attempted with one GPU via `PFC_CUDA_DEVICES=0`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage3a/jit_stage3a_backbone_benchmark_stdout.log`.
- Plot status: empty-input check passed; full plots require a completed Stage 3A benchmark run directory.
- benchmark result status: pending GPU-visible shell.
- next GPU-visible commands: `export PFC_CUDA_DEVICES=0; bash scripts/run_jit_stage3a_backbone_benchmark.sh; BENCHMARK_DIR="$(ls -td logs/stage3a/jit_backbone_benchmark/* | head -n 1)"; python scripts/plot_stage3a_jit.py --benchmark-dir "$BENCHMARK_DIR"; python scripts/make_stage3a_report_tables.py --benchmark-dir "$BENCHMARK_DIR"`.
- artifact status: no new weights downloaded; logs, outputs, checkpoints, datasets, generated images, and large binaries remain ignored.
## Stage 0 Smoke - 2026-06-01T06:53:32.025768+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 3A JiT BackboneCache Benchmark Results - 2026-06-01T07:08:13Z

- current `git rev-parse HEAD`: 619c4a0a3fced8346dcec4effec157590671afbd
- implementation status: worktree patch pending; not committed in this turn.
- GPU policy used by benchmark wrapper: one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- Stage 3A benchmark run dir: `logs/stage3a/jit_backbone_benchmark/20260601T065811Z_seed0_ref50`.
- benchmark scope: JiT 50-step no-cache reference, BackboneCache presets, and reduced-step no-cache baselines using matched seeds/noise/labels.
- seeds: 0, 1, 2; samples per seed: 16; reference steps: 50.
- report files: `benchmark_results.csv`, `benchmark_aggregate.csv`, `summary.md`, `paper_table.md`, and `paper_table.csv`.
- figures generated under ignored `outputs/stage3a/figures/`: speed-quality scatter, speedup bar, rel-L2 bar, cache hit-rate bar, and frequency-delta bar.
- Best quality cache preset: `quality_t02_08`, speedup mean 1.3965, rel-L2 mean/std 0.022393/0.006337, PSNR mean 46.3436, hit rate 0.3000.
- Warmup-equivalent cache preset: `quality_t01_08_w2`, speedup mean 1.3909, rel-L2 mean/std 0.022393/0.006337, PSNR mean 46.3436, hit rate 0.3000.
- Best speed-quality cache preset: `speed_t02_10`, speedup mean 1.6072, rel-L2 mean/std 0.027789/0.005509, PSNR mean 44.2968, hit rate 0.4000.
- Aggressive cache preset: `aggressive_i3_t02_08`, speedup mean 1.5533, rel-L2 mean/std 0.033563/0.006292, PSNR mean 42.6470, hit rate 0.3800.
- Reduced-step baselines: 30 steps speedup 1.6607 rel-L2 0.181010; 35 steps speedup 1.4237 rel-L2 0.133187; 40 steps speedup 1.2445 rel-L2 0.111390.
- Main Stage 3A finding: at comparable speed, BackboneCache preserves the 50-step trajectory much better than reducing no-cache solver steps; for example `quality_t02_08` has rel-L2 0.022393 at speedup 1.3965, compared with reduced-35 rel-L2 0.133187 at speedup 1.4237.
- post-check `conda run -n jit python scripts/run_stage0_smoke.py`: passed.
- post-check `conda run -n jit pytest -q`: passed with 67 tests.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.
## Stage 0 Smoke - 2026-06-01T07:08:52.332341+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-06-01T07:22:18.294159+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 3B DeCo Direct-Velocity Cache Implementation Check - 2026-06-01T07:52:50Z

- current `git rev-parse HEAD`: 9787ff60e3a6e6dde8dc73e29135e451bc07a3d1
- implementation status: worktree patch pending; not committed in this turn.
- scope: DeCo direct-v-pred whole-unit cache feasibility, module inspection, fixed-interval cache run, reduced-step baseline, benchmark sweep, and plotting scripts.
- not implemented: token cache, adaptive online cache, solver-aware cache, calibration, or final PixelFlowCache policy.
- DeCo checkpoint detected: `ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt`.
- `conda run -n deco python scripts/run_stage0_smoke.py`: passed.
- `conda run -n deco pytest -q`: failed because the `deco` environment does not have `pytest` installed.
- post-check `conda run -n jit python scripts/run_stage0_smoke.py`: passed.
- post-check `conda run -n jit pytest -q`: passed with 72 tests.
- static checks: `python -m py_compile` on Stage 3B Python files passed; `bash -n` on Stage 3B wrappers passed.
- DeCo inspect run dir: `logs/stage3b/deco_inspect/20260601T074842Z_seed0_steps20_inspect`.
- DeCo inspect result: 496 named modules, 427 listed inspection rows, 32 safe cacheable units.
- Safe cacheable units by category: 28 `backbone_block`, 3 `decoder_block`, 1 `final_head`.
- Excluded inspection categories: 179 `norm_or_modulation`, 216 `tiny_module`.
- DeCo cache run status: attempted through `bash scripts/run_deco_stage3b_cache.sh`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage3b/deco_stage3b_cache_stdout.log`.
- DeCo benchmark status: attempted through `bash scripts/run_deco_stage3b_benchmark.sh`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage3b/deco_stage3b_benchmark_stdout.log`.
- GPU blocker: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- Next GPU-visible commands: `export PFC_CUDA_DEVICES=0; bash scripts/run_deco_stage3b_cache.sh; bash scripts/run_deco_stage3b_benchmark.sh; BENCHMARK_DIR="$(ls -td logs/stage3b/deco_benchmark/* | head -n 1)"; python scripts/plot_stage3b_deco.py --benchmark-dir "$BENCHMARK_DIR"`.
- new weights downloaded: none.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.

## Stage 3B DeCo Direct-Velocity Cache Results - 2026-06-01T08:05:23Z

- current `git rev-parse HEAD`: 9787ff60e3a6e6dde8dc73e29135e451bc07a3d1
- implementation status: worktree patch pending; not committed in this turn.
- GPU policy used by user-run wrappers: one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- DeCo inspect run dir: `logs/stage3b/deco_inspect/20260601T074842Z_seed0_steps20_inspect`.
- DeCo cache run dir: `logs/stage3b/deco/20260601T075825Z_seed0_steps20_i2-backbone-blocks`.
- DeCo benchmark run dir: `logs/stage3b/deco_benchmark/20260601T075857Z_seed0_steps20_benchmark`.
- Benchmark setting: seed 0, 8 samples, 20-step no-cache reference, reduced-step baselines 12/15/18.
- Standalone cache run: `backbone_blocks`, interval 2, active t `[0.2,1.0)`, 28 wrapped modules, speedup 1.0692, hit rate 0.4000, rel-L2 0.154051.
- Best benchmark speed-quality cache: `all_candidates_i2_t02_10`, speedup 1.5928, rel-L2 0.072756, MSE 0.00080664, PSNR 36.9538, hit rate 0.4000.
- Best quality cache rows: `all_candidates_i2_t02_10`, `decoder_i2_t02_10`, and `final_i2_t02_10` tied at rel-L2 0.072756 in this debug run; their speedups were 1.5928, 1.0511, and 1.0089.
- Backbone-only cache rows: `[0.2,0.8)` speedup 1.3172 rel-L2 0.102509; `[0.2,1.0)` speedup 1.4727 rel-L2 0.154051.
- Reduced-step baselines: 12 steps speedup 1.6530 rel-L2 0.259579; 15 steps speedup 1.3285 rel-L2 0.210121; 18 steps speedup 1.1069 rel-L2 0.139081.
- Main Stage 3B finding: in this small debug benchmark, DeCo cache beats reduced-step no-cache at comparable speed; `all_candidates_i2_t02_10` is close to reduced-12 speed but has much lower rel-L2, 0.072756 vs 0.259579.
- Figures generated under ignored `outputs/stage3b/figures/`: speed-quality scatter, rel-L2 bar, speedup bar, cache hit-rate bar, and frequency-delta bar.
- Plot note: matplotlib used a temporary cache under `/tmp` because `/root/.config/matplotlib` was not writable; plot generation still completed.
- new weights downloaded: none.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.
## Stage 0 Smoke - 2026-06-01T07:48:43.149927+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-06-03T02:25:24.944715+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 3B2 DeCo Cache Decomposition Implementation Check - 2026-06-03T02:33:22Z

- current `git rev-parse HEAD`: 01e173a1f46376cc74d40489919948b9806a872e
- implementation status: worktree patch pending; not committed in this turn.
- scope: DeCo-only cache decomposition and validation scaffolding.
- not implemented: token cache, adaptive online policy, solver-aware cache, calibration, or final PixelFlowCache policy.
- explicit DeCo cache specs added: `final_only`, `decoder_only_no_final`, `decoder_plus_final`, `backbone_only`, `backbone_plus_final`, `backbone_plus_decoder_no_final`, `late_backbone_only:<n>`, and `late_backbone_plus_final:<n>`.
- per-config nested artifact support added under `logs/stage3b2/.../runs/<method_name>_seed<seed>/`.
- scripts added: decomposition, validation, seed sweep, plotting, and report-table generation.
- `conda run -n deco python scripts/run_stage0_smoke.py`: passed.
- `conda run -n deco python -m pytest -q`: failed because the `deco` environment does not have `pytest` installed.
- post-check `conda run -n jit python scripts/run_stage0_smoke.py`: passed.
- post-check `conda run -n jit pytest -q`: passed with 80 tests.
- static checks: `python -m py_compile` on Stage 3B2 Python files passed; `bash -n` on Stage 3B2 wrappers passed.
- DeCo decomposition status: attempted through `bash scripts/run_deco_stage3b2_decomposition.sh`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage3b2/deco_stage3b2_decomposition_stdout.log`.
- DeCo seed sweep status: attempted through `bash scripts/run_deco_stage3b2_seed_sweep.sh`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage3b2/deco_stage3b2_seed_sweep_stdout.log`.
- DeCo validation status: attempted through `bash scripts/run_deco_stage3b2_validate.sh`, but this session's `nvidia-smi` failed before model load; see ignored log `logs/stage3b2/deco_stage3b2_validate_stdout.log`.
- GPU blocker: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- Next GPU-visible commands: `export PFC_CUDA_DEVICES=0; bash scripts/run_deco_stage3b2_decomposition.sh; DECOMP_DIR="$(ls -td logs/stage3b2/deco_decomposition/* | head -n 1)"; python scripts/plot_stage3b2_deco.py --decomposition-dir "$DECOMP_DIR"; python scripts/make_stage3b2_report_tables.py --decomposition-dir "$DECOMP_DIR"`.
- Optional GPU-visible commands: `bash scripts/run_deco_stage3b2_seed_sweep.sh; bash scripts/run_deco_stage3b2_validate.sh`.
- new weights downloaded: none.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.

## Stage 3B2 DeCo Cache Decomposition Results - 2026-06-03T02:51:56Z

- current `git rev-parse HEAD`: 01e173a1f46376cc74d40489919948b9806a872e
- implementation status: worktree patch pending; not committed in this turn.
- GPU policy used by user-run wrapper: one GPU via `PFC_CUDA_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`.
- DeCo decomposition run dir: `logs/stage3b2/deco_decomposition/20260603T023555Z_seed0_steps20_decomposition`.
- Decomposition setting: seed 0, 8 samples, 20-step no-cache reference, cache interval 2, active t `[0.2,1.0)`, reduced-step baselines 12/15/18.
- Generated report files: `decomposition_results.csv`, `decomposition_results.json`, `decomposition_aggregate.csv`, `summary.md`, `paper_table_deco_decomposition.md`, and `paper_table_deco_decomposition.csv`.
- Generated figures under ignored `outputs/stage3b2/figures/`: decomposition speed-quality, rel-L2 by cache unit, speedup by cache unit, final cache effect, cache vs reduced steps, and validation placeholder.
- Best speed among same-quality cache rows: `backbone_plus_decoder_no_final`, speedup 1.1320, rel-L2 0.072756, hit rate 0.4000.
- `all_candidates`: speedup 1.1313, rel-L2 0.072756, PSNR 36.9538, hit rate 0.4000.
- `backbone_plus_final`: speedup 1.0987, rel-L2 0.072756, PSNR 36.9538, hit rate 0.4000.
- `final_only`: speedup 0.8499, rel-L2 0.072756, PSNR 36.9538, hit rate 0.4000.
- `decoder_only_no_final` and `decoder_plus_final` also matched rel-L2 0.072756 in this debug run, but were slower than no-cache.
- Backbone-only rows were worse: `backbone_only` speedup 1.0028 rel-L2 0.154051; `late_backbone_only_6` speedup 0.8430 rel-L2 0.154051.
- Reduced-step baselines: 12 steps speedup 1.6698 rel-L2 0.259579; 15 steps speedup 1.3489 rel-L2 0.210121; 18 steps speedup 1.1202 rel-L2 0.139081.
- Main Stage 3B2 finding: in this 20-step debug decomposition, any cached decoder/final output-side boundary matched the best quality, while backbone-only cache was substantially worse; cache still beat the nearest reduced-step no-cache baseline at comparable speed.
- Validation status: not run yet in this turn.
- Seed sweep status: not run yet in this turn.
- Plot note: matplotlib used a temporary cache under `/tmp` because `/root/.config/matplotlib` was not writable; plot generation completed.
- new weights downloaded: none.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.
## Stage 0 Smoke - 2026-06-03T02:32:46.585300+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula
## Stage 0 Smoke - 2026-06-05T01:38:32.138442+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 0 Smoke - 2026-06-05T01:48:37.844427+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 3C BoundaryFlowCache Synthesis - 2026-06-05T01:49:33Z

- current `git rev-parse HEAD`: 5a16e1463ec275d707b473c5222e872b41c54e5b
- implementation status: worktree patch pending; no cleanup or Stage 3C commit has been created in this turn.
- Stage 3C role: unified JiT Stage 3A and DeCo Stage 3B2 results into method-level BoundaryFlowCache analysis.
- unified result dir: `logs/stage3c/unified/20260605T014850Z`
- unified inputs:
  - JiT Stage 3A: `logs/stage3a/jit_backbone_benchmark/20260601T065811Z_seed0_ref50`
  - DeCo 50-step validation: `logs/stage3b2/deco_validate/20260603T024359Z_seed0_steps50_validate`
  - DeCo seed sweep: `logs/stage3b2/deco_seed_sweep/20260603T024020Z_seed0_steps20_seed-sweep`
  - DeCo decomposition: `logs/stage3b2/deco_decomposition/20260603T023555Z_seed0_steps20_decomposition`
- CPU smoke: `conda run -n jit python scripts/run_stage0_smoke.py` passed.
- pytest: `conda run -n jit pytest -q` passed with 83 tests and 2 CUDA-unavailable warnings.
- Stage 3C collector: `conda run -n jit python scripts/collect_stage3c_unified_results.py` completed.
- Stage 3C table generation: `conda run -n jit python scripts/make_stage3c_paper_tables.py --unified-dir logs/stage3c/unified/20260605T014850Z` completed.
- Stage 3C plot generation: `conda run -n jit python scripts/plot_stage3c_unified.py --unified-dir logs/stage3c/unified/20260605T014850Z` completed.
- generated tables: `paper_table_main_cache_vs_reduced`, `paper_table_boundary_ablation`, and `paper_table_seed_stability` under the unified dir.
- generated figures under ignored `outputs/stage3c/figures/`: speed-quality, rel-L2 cache-vs-reduced, speedup cache-vs-reduced, DeCo boundary ablation, and JiT-vs-DeCo best methods.
- optional DeCo Stage 3C 50-step multi-seed validation: not run in this turn; existing Stage 3B2 50-step seed0 validation and 20-step seed sweep were used.
- current best JiT method: `speed_t02_10` for speed-quality, with `quality_t02_08` as the quality preset.
- current best DeCo method: `all_candidates` for speedup at rel-L2 0.0460 in the 50-step seed0 validation; `backbone_plus_final` matches the same rel-L2 with lower speedup.
- new weights downloaded: none.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, and large binaries remain ignored and should not be committed.

## Stage 0 Smoke - 2026-06-05T03:39:49.546302+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 0 Smoke - 2026-06-05T03:52:58.429229+00:00

- smoke test passed: True
- checks: xpred scalar t, xpred vector t, token t broadcast, vpred raw, euler sampler, cfg formula

## Stage 4A Full Inference And FID-Ready Pipeline - 2026-06-05T03:53:13Z

- current `git rev-parse HEAD`: 697e1f1402495ebf3de4889b0eb106d21612c0e3
- implementation status: worktree patch pending; no Stage 4A commit has been created in this turn.
- Stage 4A role: add full-generation, FID-ready evaluation, ImageNet reference preparation, command-plan generation, result collection, and plotting scripts for JiT and DeCo.
- CPU smoke: `conda run -n jit python scripts/run_stage0_smoke.py` passed.
- pytest: `conda run -n jit pytest -q` passed with 91 tests and 2 CUDA-unavailable warnings.
- syntax check: `conda run -n jit python -m py_compile ...` passed for new Stage 4A Python modules and scripts.
- help checks completed:
  - `conda run -n jit python scripts/run_jit_stage4a_generate.py --help`
  - `conda run -n jit python scripts/run_deco_stage4a_generate.py --help`
  - `conda run -n jit python scripts/evaluate_stage4a_fid.py --help`
- command-plan generation completed:
  - `conda run -n jit python scripts/run_stage4a_full_eval_plan.py --models jit,deco --num-images 100 --out-script scripts/launch_stage4a_smoke_100.sh`
  - `bash scripts/print_stage4a_smoke_commands.sh`
- generated manual launch script: `scripts/launch_stage4a_smoke_100.sh`
- real inference launched by Codex: no.
- FID/IS/KID computed by Codex: no.
- background jobs submitted by Codex: no.
- new weights downloaded: none.
- artifact status: `logs/`, `outputs/`, `ckpts/`, datasets, generated images, FID outputs, and large binaries remain ignored and should not be committed.

## Stage 4A JiT Launch Fix - 2026-06-05T03:59:13Z

- current `git rev-parse HEAD`: 697e1f1402495ebf3de4889b0eb106d21612c0e3
- user-run smoke blocker: `scripts/run_jit_stage4a_generate.py` imported `_sample_jit` from `scripts.run_jit_stage2b_cache`, but the function lives in `scripts.run_jit_stage2_cache`.
- fix: Stage 4A JiT runtime helper now imports `_sample_jit` from `scripts.run_jit_stage2_cache`.
- additional fix: JiT `--dry-run` now recursively serializes nested `Path` values in the resolved config.
- regression test added: `tests/test_stage4a_plan.py` checks the runtime helper source module and dry-run JSON path serialization.
- validation:
  - `conda run -n jit python -m py_compile scripts/run_jit_stage4a_generate.py` passed.
  - `conda run -n jit pytest -q tests/test_stage4a_plan.py` passed with 3 tests.
  - `conda run -n jit python scripts/run_jit_stage4a_generate.py --method no_cache_50 --num-images 100 --batch-size 8 --seed 0 --run-id stage4a_n100_seed0 --dry-run` passed.
  - `conda run -n jit pytest -q` passed with 93 tests and 2 CUDA-unavailable warnings.
- real inference launched by Codex after this fix: no.
- FID/IS/KID computed by Codex: no.
- new weights downloaded: none.

## Stage 4A 100-Image Smoke Generation - 2026-06-05

- user-run command: `export PFC_CUDA_DEVICES=0; bash scripts/launch_stage4a_smoke_100.sh`
- generation status: completed for all 10 configured methods.
- generated image counts: 100 PNGs and 100 manifest rows for each JiT and DeCo method.
- JiT methods completed:
  - `no_cache_50`
  - `bfc_quality_t02_08`
  - `bfc_speed_t02_10`
  - `reduced_steps_35`
  - `reduced_steps_30`
- DeCo methods completed:
  - `no_cache_50`
  - `bfc_all_candidates_t02_10`
  - `bfc_backbone_plus_final_t02_10`
  - `reduced_steps_35`
  - `reduced_steps_30`
- generation output root: `outputs/stage4a/full_generation`
- run id: `stage4a_n100_seed0`
- generation-only summary: `logs/stage4a/summary/stage4a_n100_seed0_generation_only`
- FID status: not computed. The launch script stopped at the first FID command because no supported backend was installed in the evaluation environment.
- FID blocker: `No supported FID backend found. Install one of: pip install torch-fidelity, pip install clean-fid, or pip install torchmetrics[image].`
- third-party cleanup: generated `__pycache__` files in JiT/DeCo were removed or restored after inspection.
- new weights downloaded: none.
