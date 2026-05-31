# Stage 2C Window Ablation And Probe Diagnostics

Stage 2C is a JiT-only analysis stage for the fixed-interval whole-block cache baseline. It tightens the Stage 2B window choice and separates local cache approximation error from accumulated trajectory drift.

This stage does not implement token cache, DeCo cache, adaptive online policy, solver-aware cache, calibration, or the final PixelFlowCache method.

## Motivation

Stage 2B showed that disabling cache near late timesteps reduces same-seed drift for JiT. Because JiT predicts `x0` and converts it to velocity with `1 / clamp(1 - t)`, late errors can be amplified. Stage 2C tests whether the lower or upper window boundary matters more, and uses a full probe to compare cached velocity against a fresh model on the cached state.

## Commands

Use one GPU by default:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
export PFC_CUDA_DEVICES=0
bash scripts/run_jit_stage2c_window_ablation.sh
bash scripts/run_jit_stage2c_probe.sh
```

Plot completed runs:

```bash
python scripts/plot_stage2c_jit.py \
  --window-dir logs/stage2c/jit_window_ablation/<run_id> \
  --probe-dir logs/stage2c/jit_probe/<run_id>
```

Optional larger validation:

```bash
bash scripts/run_jit_stage2c_validate.sh
python scripts/plot_stage2c_jit.py --validation-dir logs/stage2c/jit_validate/<run_id>
```

Useful overrides:

```bash
export PFC_STAGE2C_NUM_SAMPLES=8
export PFC_STAGE2C_BATCH_SIZE=4
export PFC_STAGE2C_STEPS=20
export PFC_STAGE2C_TIMING_REPEATS=3
export PFC_STAGE2C_WARMUP_RUNS=1
export PFC_STAGE2C_INCLUDE_INTERVAL4=0
```

For the probe script:

```bash
export PFC_STAGE2C_NUM_SAMPLES=4
export PFC_STAGE2C_DIAG_PROBE_STEPS=all
```

## Outputs

Window ablation:

- `logs/stage2c/jit_window_ablation/<run_id>/window_ablation_results.csv`
- `logs/stage2c/jit_window_ablation/<run_id>/window_ablation_results.json`
- `logs/stage2c/jit_window_ablation/<run_id>/summary.md`

Full probe:

- `logs/stage2c/jit_probe/<run_id>/step_error_stats.jsonl`
- `logs/stage2c/jit_probe/<run_id>/probe_summary.json`
- `logs/stage2c/jit_probe/<run_id>/summary.md`

Validation:

- `logs/stage2c/jit_validate/<run_id>/validation_results.csv`
- `logs/stage2c/jit_validate/<run_id>/validation_results.json`
- `logs/stage2c/jit_validate/<run_id>/summary.md`

Figures are written under ignored `outputs/stage2c/figures/`.

## Interpretation

- `active_t_min` tests whether early-step cache reuse hurts quality.
- `active_t_max` tests the late-step amplification boundary.
- `speedup_median` is still based on the main no-cache vs cached timing; full-probe diagnostics are extra and should not be reported as speedup.
- `trajectory_error` compares cached trajectory velocity against the no-cache trajectory at the same step, so it includes accumulated drift.
- `probe_error` compares cached velocity against a fresh model evaluated on the cached state, so it isolates local approximation error at that state.

## Limitations

The current cache is still whole-block output reuse for JiT only. Whole-block subset behavior is boundary-dominated for sequential Transformer blocks; see [STAGE2C_BOUNDARY_CACHE_OBSERVATION.md](STAGE2C_BOUNDARY_CACHE_OBSERVATION.md).

## Next Step

Use Stage 2C to select a scientifically defensible JiT fixed-cache baseline before Stage 3 design work. The next implementation step should be profiling-informed method design, not adding token or DeCo cache inside this cleanup stage.
