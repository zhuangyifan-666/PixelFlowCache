# PixARC Stage-1 JiT Instrumentation

## Scope

This implementation collects JiT fresh-trajectory counterfactual labels. It is instrumentation, not the final PixARC method: it does not implement the Stage-2 sequential oracle, a risk predictor, calibration, an online scheduler, or a cache controller. The 32-sample dataset is a diagnostic dataset and is not suitable for final FID/IS claims. No safe or certified claim follows from Stage 1.

## Fresh-State Semantics

For every Euler step, the runner captures a complete fresh cond/uncond forward and advances only with

`x_next_fresh = x + dt * v_cfg_fresh`.

Every replay, reuse, and Taylor candidate starts from the same fresh `x`, uses the current branch condition, current CFG interval/scale, current x-prediction-to-velocity conversion, and the same Euler `dt`. Candidate states never update the trajectory or the history. Only detached, cloned fresh boundary values are appended after every candidate for the current step has finished.

## Boundary Plans

Plans use half-open block ranges `[start, end)`. For a 12-block JiT-B/16 model, the defaults are `early=[0,4)`, `middle=[4,8)`, `late=[8,12)`, `early_middle=[0,8)`, and `whole=[0,12)`. Arbitrary supported depths are split into thirds; `middle_late` is available as a non-default diagnostic plan.

A candidate executes the current prefix `[0,start)`, substitutes a historical or forecast fresh hidden value at `end`, executes the current suffix `[end,total_blocks)`, then applies the current final layer. Existing context tokens are retained exactly once.

## Actions

- `fresh`: one zero-risk record per sample and step, independent of plan.
- `replay_age_0`: current fresh boundary replay used only as a correctness gate.
- `reuse_age_1` / `reuse_age_2`: fresh boundary output from step `i-1` / `i-2`.
- `taylor_order_1`: linear extrapolation from the two latest past fresh boundary outputs.

Unavailable history-dependent actions emit `action_ready=false`, `skip_reason="insufficient_history"`, and JSON `null` metrics. They never emit NaN or Infinity.

## Risk Labels

Let `delta = x_next_candidate - x_next_fresh` and `scale = atol + rtol * max(abs(x), abs(x_next_fresh))`.

- Solver-scaled RMS: `sqrt(mean((delta / scale)^2))`.
- Relative transition error: `||candidate_update - fresh_update|| / (||fresh_update|| + eps)`.
- Velocity error: `||v_candidate - v_fresh|| / (||v_fresh|| + eps)`.
- Branch errors compare candidate and fresh cond/uncond x-predictions.
- Low/high-frequency risks apply cached radial FFT masks to `delta` and the fresh update.

All reductions and FFTs use float32. Diagnostic action latency covers the candidate cond/uncond forward, CFG combination, and Euler transition only. It excludes risk, FFT, and I/O and is not comparable to end-to-end generation speedup.

## Output Contract

Each sample is committed through a temporary directory, atomic rename, and a final `DONE.json` marker:

```text
outputs/pixarc/stage1/jit/<run_id>/
  run_config.json
  labels.json
  samples/sample_000000/
    risk_records.jsonl
    correctness_records.jsonl
    sample_summary.json
    DONE.json
  shard_0/shard_meta.json
  merged/
    risk_records.jsonl
    correctness_records.jsonl
    action_latency.csv
    risk_summary.csv
    correctness_summary.json
    stage1_summary.md
    validation_report.json
```

Resume skips only matching `DONE.json` samples. Incomplete sample directories are discarded and recomputed; completed samples from another configuration are rejected.

## Correctness Gates

The validator requires fresh split-forward equivalence, complete replay-age-0 checks, default replay thresholds `max_abs <= 1e-5` and `relative_l2 <= 1e-6`, finite strict JSON, zero future leakage, complete action readiness, matching shape/dtype, cleared per-sample history, and at least one positive ready reuse risk. `--strict` returns nonzero on `BLOCK`.

Four-GPU execution uses four independent sample shards and separate worker logs. It does not use distributed model execution, and parallel wall time is not reported as algorithm speedup. Windows supports `--dry-run` and `--print-only`; real checkpoint/CUDA validation remains server-only.
