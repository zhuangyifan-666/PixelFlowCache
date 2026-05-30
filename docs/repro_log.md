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
