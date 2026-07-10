# Experiment Readiness Status

Last updated: 2026-07-10 20:25:37 Asia/Shanghai

Recovery snapshot: `C:\Users\user\AppData\Local\Temp\PixelFlowCache_recovery_20260710_180506`

## Preserved implementation

- [x] Timing schema v2, provenance, and cache input signatures
- [x] DeCo/PixelGen/JiT sharding foundations
- [x] FID backend validation and strict paired-metric JSON
- [x] Safe-BFC, TaylorSeer, SpeCa, and DiCache core policies
- [x] Configuration consistency and readiness-planning foundations
- [x] Shell line-ending policy and static shell validation

## Hardening completed

- [x] Unified policy `reset_stats()` and `reset_runtime_state()` contracts
- [x] Warmup reset for every policy and `RuntimeCacheState`
- [x] Atomic label schedules and PNG/manifest resume reconciliation
- [x] Duplicate, stale, wrong-label, and canonical-path manifest handling
- [x] Generic JiT/DeCo/PixelGen multi-environment launcher
- [x] Method registry validation, Safe maps, and method-specific debug routing
- [x] Generic shard merge entrypoint with model metadata
- [x] Server readiness Gates 0-7 with model environments and batch sizes
- [x] Stage 4A Safe-BFC map validation and command routing
- [x] CPU-safe `required_gpus=0` preflight and requested-model checks
- [x] Independent conda environment probes and exact Safe-tree density checks
- [x] Canonical config-consistency integration and torch-fidelity API capability check
- [x] Strict single-GPU timing comparison signatures and detailed diffs
- [x] Policy reset and FID capability documentation

## Windows verification

- [x] 180 Python files passed `py_compile`
- [x] 10 YAML configurations parsed successfully
- [x] 10 tracked shell scripts are LF
- [x] 10 tracked shell scripts passed `bash -n` with `D:\Git\bin\bash.exe`
- [x] Shell policy tests passed
- [x] 365 tests collected
- [x] 364 tests passed, 1 skipped, 0 failed in four file groups
- [x] CPU/non-strict preflight returned zero with GPU validation disabled
- [x] Readiness planner print-only returned zero
- [x] Stage 4A full planner returned zero with both Safe maps
- [x] JiT launcher print-only returned zero for five cache methods and debug routes
- [x] DeCo and PixelGen launcher print-only returned zero with batch size 4
- [x] Final Git/workspace hygiene audit passed

## Server-required verification

- [ ] Blocked on server GPU: checkpoint loading
- [ ] Blocked on server GPU: CUDA no-cache smoke
- [ ] Blocked on server GPU: DiCache force-full equivalence
- [ ] Blocked on server GPU: synchronized single-GPU timing
- [ ] Blocked on server GPU: four-GPU execution
- [ ] Blocked on server GPU: FID/IS
- [ ] Blocked on server GPU: PSNR/SSIM/LPIPS/relative-L2 paired metrics

No known Windows code-level blocker remains. This status does not claim checkpoint, CUDA, multi-GPU, FID, or paired-metric execution.

## PixARC Stage 1

- [x] JiT fresh split-forward capture and joint boundary plans implemented
- [x] Fresh-state replay, age-1/age-2 reuse, and first-order Taylor counterfactuals implemented
- [x] Full Euler transition risk, radial frequency risk, and cheap signals implemented
- [x] Strict per-sample atomic output, resume, shard merge, and validator implemented
- [x] Four-worker independent-shard launcher supports Windows print-only planning
- [x] Windows fake-model tests, strict synthetic merge/validator tests, dry-run, and print-only passed
- [ ] Blocked on server GPU: 2-image strict correctness smoke
- [ ] Blocked on server GPU: 32-sample four-GPU instrumentation collection

PixARC Stage 1 is instrumentation only. The Stage-2 sequential oracle, risk predictor, calibration, online controller, and final scheduler remain unimplemented.
