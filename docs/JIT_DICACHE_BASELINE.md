# Adapted DiCache-style baseline for JiT

This is a clean-room, adapted DiCache-style baseline for JiT-B/16 at ImageNet 256 resolution, not an official DiCache reproduction. The official repository is a read-only design reference and is neither imported nor required at runtime. There are currently no experimental results. The default `probe_depth=1`, `delta_y` threshold `0.4`, retention ratio `0.2`, and gamma range `[1.0, 1.5]` come from the released FLUX example and remain unvalidated for JiT.

## Fair CFG execution

The default `share_cfg_prefix=false` mode separately executes `x_embedder`, `pos_embed`, and `t_embedder` for cond and uncond. This matches the two `model.net` calls in the no-cache sampler, so a reported DiCache speedup cannot silently include CFG-prefix sharing. Both branches retain their own prefix tensors, conditions, probes, and residual histories while sharing one full/reuse decision.

`share_cfg_prefix=true` computes the read-only common prefix once and is labeled `dicache_style_shared_cfg_prefix`. It is an independent ablation, not part of the default comparison table. Prefix calls and savings are reported separately from Transformer block-call savings.

`dicache_force_full` with the default non-shared prefix is expected to match no-cache output, dtype, CFG interval, timestep schedule, block/final-layer calls, and prefix-call counts.

## Online Probe Profiling

Every step runs the first `probe_depth` blocks for both branches. Relative-L1 metrics are computed as float32 zero-dimensional tensors on the source device. When previous input and probe histories exist, `delta_x`, cond `delta_y`, and uncond `delta_y` are stacked and transferred to the host together. Thus online scheduling performs at most one necessary device-to-host decision sync per step; first or insufficient-history steps that cannot form metrics do not sync. There are no separate per-metric transfers.

The only schedule variant is `released_flux_compat`. It centralizes the first-step, retention-boundary, last-step, and strict-`<` adaptive rules. Observed statistics include all steps where probe changes can be measured. Decision statistics include only the adaptive-threshold path; forced, first, retention, last, and insufficient-history full steps never contaminate decision-error distributions.

This is batch-level routing, not per-sample routing. Cond and uncond always make the same full/reuse decision.

## JiT residual and split forward

The adaptation defines:

```text
h0 = image tokens after x_embedder + pos_embed, before block 0
hm = image tokens after the first probe_depth blocks
hM = image tokens after all blocks, before final_layer
P  = hm - h0
R  = hM - h0
```

Refresh resumes at `blocks[probe_depth:]`; it does not repeat shallow blocks. Reuse skips only deep blocks. JiT context tokens are inserted at most once, and `final_layer` remains fresh every step. Fixed position/context buffers are aligned to the active hidden tensor's device and dtype so bf16 features are not promoted to float32.

## Dynamic Cache Trajectory Alignment

Each branch retains the last two true refresh pairs `(R_old, P_old)` and `(R_new, P_new)`. On reuse:

```text
gamma_raw = mean(abs(P_current - P_old))
            / (mean(abs(P_new - P_old)) + eps)
gamma = clamp(gamma_raw, gamma_min, gamma_max)
R_hat = R_old + gamma * (R_new - R_old)
```

Gamma remains a device tensor while forming `R_hat`; successful gamma scalars are buffered and transferred once when batch statistics are finalized. With fewer than two true histories, reuse falls back to the latest true residual. A non-finite or near-zero probe denominator is a degenerate trajectory: it also falls back to the latest residual and is not counted as successful DCTA or in gamma statistics. Approximate residuals never enter true history.

History views are compacted before retention when they could keep a larger context-bearing storage alive. Memory reporting deduplicates actual backing storages by device and storage pointer, and exposes current/peak bytes plus tensor and unique-storage counts.

## Statistics and timing

Running statistics expose exact count, sum, sum of squares, mean, standard deviation, extrema, and bounded samples used for approximate quantiles. Four-card merging combines those sufficient statistics by their own counts and re-estimates quantiles from merged bounded samples.

`effective_block_compute_saving_rate` includes the fixed per-step probe-block cost and is a block-call proxy, not exact FLOPs. CFG-prefix saving is separate. Local `time.perf_counter` fields are named `*_host_dispatch_time_sec` and have `timing_semantics=host_dispatch_only`; they are debug signals, not CUDA latency or kernel wall time. Final speedup must come from synchronized, end-to-end, single-card sampler wall-clock. Four-card wall-clock only describes generation throughput.

The mandatory correctness gate compares force-full DiCache with JiT no-cache before any quality or speed claim. CFG-prefix sharing remains disabled by default for fairness.
## Warmup reset semantics

`clear_batch()` flushes pending gamma statistics before clearing DiCache batch history; `reset_runtime_state()` provides the common runtime-reset interface. `reset_stats()` clears DCTA and decision counters, error/gamma running statistics and bounded samples, synchronization counts, history-memory peaks, and host-dispatch timings without changing policy configuration. JiT runs both reset phases after warmup, so formal statistics contain no warmup DCTA, error, gamma, synchronization, or timing data.
