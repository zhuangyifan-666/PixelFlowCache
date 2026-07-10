# JiT adapted SpeCa-style baseline

## Scope

This is an **adapted SpeCa-style baseline**, not an official SpeCa reproduction. It is intended for a controlled JiT-B/16 ImageNet-256 comparison with the existing 50-step reference, reduced-step, SeaCache-style, TaylorSeer-style, and Safe-BFC methods.

The implementation was informed by the released Cache4Diffusion SpeCa DiT code, but that repository is a read-only reference. It is not copied into PixelFlowCache, imported at runtime, or required for generation. PixelFlowCache does not reproduce the official DiT model implementation.

## Reference behavior

The released DiT implementation uses TaylorSeer as a fast draft feature predictor. A full step evaluates the complete transformer and updates Taylor derivative history. A Taylor step forecasts attention and MLP residual features for each transformer block. After the configured minimum speculative length, it also evaluates the final transformer block freshly, compares that block output with the forecast, and uses the resulting error to schedule following computation. Initial steps are full, a speculative run has minimum and maximum lengths, and its acceptance threshold tightens from the noisy stage toward the image/detail stage.

The paper describes relative-L2 verification, while the released DiT command default is `error_metric="relative_l1"`. The released implementation's “final-layer” check is concretely the final Transformer block (`layer == 27` for the 28-block DiT-XL/2), rather than the post-transformer output layer.

Files reviewed for this behavior:

- `Cache4Diffusion/README.md`
- `dit/speca-dit/sample.py`
- `dit/speca-dit/models.py`
- `dit/speca-dit/cache_functions/cache_init.py`
- `dit/speca-dit/cache_functions/cal_type.py`
- `dit/speca-dit/cache_functions/update_cache.py`
- `dit/speca-dit/cache_functions/score_evaluate.py`
- `dit/speca-dit/cache_functions/scores.py`
- `dit/speca-dit/taylor_utils/__init__.py`

## Adaptation

| Aspect | Official SpeCa DiT | PixelFlowCache adapted SpeCa |
| --- | --- | --- |
| Draft unit | attention/MLP residual | JiT block output |
| Model | latent DiT-XL/2 | pixel-space JiT-B/16 |
| Blocks | 28 | 12 |
| CFG | concatenated | separate cond/uncond, shared schedule |
| Verifier | last transformer block | last selected JiT block |
| Metric default | released code `relative_l1` | `relative_l1` |
| Decision effect | following computation | next sampling step |
| Sample adaptivity | official framework claim | batch-level in this adaptation |

The draft reuses `TaylorSeerCachePolicy` history storage and Lagrange polynomial extrapolation at JiT block-output boundaries. Feature history remains isolated by batch session, module, CFG branch, and solver stage. The full/speculative schedule is shared across selected modules and the cond/uncond forwards. This is batch-level adaptation; it does not perform strict per-sample routing.

## Schedule

JiT steps are indexed `0, 1, ..., total_steps - 1` in the noise-to-image direction. One decision is made per step and solver stage:

1. Finalize the previous step's cond/uncond verification errors.
2. Run full computation while `step_idx < first_full_steps` (default 3).
3. Run full computation if any selected module/branch lacks the required Taylor history (default 2 observations). This all-module preflight prevents a partly forecast, partly full step.
4. After a full step, begin a speculative run with counter 1.
5. Force full computation after `max_forecast_steps` speculative steps (default 5).
6. Verification eligibility is based on the number of speculative steps completed *before* entering the current step. It begins on speculative step `min_forecast_steps + 1`; with the default minimum of 2, speculative steps 1 and 2 are not verified and step 3 is the first verified step.
7. Once verification has produced an error, reject the run on the next step when that error exceeds the next step's threshold. Otherwise continue speculatively. Five speculative steps are allowed with the default maximum of 5, and the following step is full.

A verification-enabled step still returns and commits its forecast as the current block result. The fresh verifier output is used only for error measurement: it is not written into Taylor history or `RuntimeCacheState`, and it does not roll back the current step. Its error can affect only the next sampling step. If no verifier error is produced, the next step is forced full. If only one expected CFG branch produces an error, that error is used and the missing branch is recorded.

## Metric and threshold

All verifier arithmetic is accumulated in float32. Supported metrics are `l1`, `l2`, `relative_l1`, `relative_l2`, and `cosine_error`. The default follows the released code and uses element-wise relative L1:

```text
mean(abs(predicted - fresh) / (abs(fresh) + eps))
```

For step `i`, the threshold is

```text
progress_i = (i + 1) / total_steps
threshold_i = max(base_threshold * decay_rate ** progress_i, min_threshold)
```

Defaults are `base_threshold=0.1`, `decay_rate=0.01`, and `min_threshold=0.01`. Thus the threshold is non-increasing from noise toward image details and never falls below the floor.

The cond and uncond verifier errors are finalized only after the step. Default aggregation is their mean, which matches a concatenated equal-sized CFG batch for element-wise mean metrics. `max` is available as a conservative alternative. A cond error cannot change the uncond mode in the same step.

Verification error statistics keep exact global count, mean, standard deviation, minimum, and maximum, plus a bounded deque of at most 4096 values for quantile estimation. `p50`, `p90`, and `p95` are therefore marked approximate after the bound is exceeded; full raw values are not copied into `cache_stats.json`.

## Compute accounting

`runtime_cache.hit_rate` is the raw rate at which a forecast tensor was returned. It is not SpeCa's block compute-saving rate because the verifier block also performs a fresh forward. SpeCa therefore reports `logical_managed_calls`, `full_compute_calls`, `forecast_committed`, and `verifier_fresh_calls` separately. `effective_compute_saving_rate` subtracts verifier fresh calls from forecasts before dividing by logical managed calls. Result tables should prefer this effective rate over the raw hit rate, while overall synchronized sampler latency remains the primary performance measurement.

## Verifier resolution and overhead

`auto` canonicalizes selected module names, parses trailing numeric `blocks.N` indices, and selects the largest index. For JiT-B/16 with all blocks selected this resolves to `blocks.11`. An explicit verifier must belong to the selected set. If numeric parsing is impossible, the resolver warns and uses the last selected module; generation metadata stores both requested and resolved values.

JiT-B/16 has only 12 blocks, so one fresh verifier block represents an estimated structural fraction of `1/12` of the selected transformer blocks, larger than `1/28` for DiT-XL/2. This fraction is not a measured speed cost. Per-operation `perf_counter` fields are explicitly named `*_host_dispatch_time_sec`; they measure Python/host dispatch only, not asynchronous GPU wall-clock. CUDA event profiling is disabled and CUDA timing fields remain null. Final algorithm latency must use synchronized wall-clock timing around the sampler, and verifier GPU overhead requires a separate profiler run. The official SpeCa 3.5% overhead must not be reused as a JiT result.

Four-card launchers use shard-specific debug JSONL paths to avoid concurrent writes. Multi-GPU sharding only increases generation throughput for dataset production; it is not evidence of algorithmic single-card speedup.

## Status

No generation, FID/IS, paired image metrics, or quality/speed sweep is reported here. The baseline currently provides code, configuration, dry-run/planner integration, debug JSONL events, and CPU-only unit tests. It must not be described as outperforming any comparison method until controlled experiments are run.

Server execution must pass the unified strict preflight and 16-image baseline smoke before the single-GPU timing gate.
## Warmup reset semantics

`clear_batch()` flushes the current speculative run and removes stream, verification, and inherited forecast history; `reset_runtime_state()` also restarts the batch-session identifier. `reset_stats()` clears verifier counts, verification-error moments and bounded samples, run summaries, forecast statistics, and host-dispatch timings while preserving configuration. Both phases run after warmup, so warmup verification and speculative activity is excluded from formal statistics.
