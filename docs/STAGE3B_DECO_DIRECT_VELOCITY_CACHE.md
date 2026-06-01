# Stage 3B DeCo Direct-Velocity Cache

Stage 3B tests whether the whole-backbone boundary-cache idea from JiT transfers to DeCo, where the denoiser predicts velocity directly.

This stage does not implement token cache, adaptive online cache, calibration, solver-aware cache, or full ImageNet-scale FID.

## Motivation

Stage 3A showed that JiT BackboneCache preserves the 50-step trajectory much better than reduced-step no-cache baselines at similar speed. JiT is x-pred, so cached block errors are amplified through the x-pred to velocity conversion near the image end of the trajectory.

DeCo is direct v-pred. Its cache error is measured directly in velocity/output space, so Stage 3B is a separate feasibility test rather than an automatic transfer of JiT settings.

## Cache Units

The Stage 3B wrappers only cache coarse DeCo units:

- `backbone_blocks`: `blocks.N`
- `decoder_blocks`: `dec_net.res_blocks.N`
- `final`: `dec_net.final_layer`
- `all_candidates`: the safe union of the above

Norm, adaLN/modulation, embedding, linear-only, tiny modules, and arbitrary nested submodules are excluded by default.

## Commands

Use one GPU by default:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
export PFC_CUDA_DEVICES=0
bash scripts/run_deco_stage3b_inspect.sh
bash scripts/run_deco_stage3b_cache.sh
bash scripts/run_deco_stage3b_benchmark.sh
```

Plot a benchmark:

```bash
BENCHMARK_DIR="$(ls -td logs/stage3b/deco_benchmark/* | head -n 1)"
python scripts/plot_stage3b_deco.py --benchmark-dir "$BENCHMARK_DIR"
```

Optional reduced-step only baseline:

```bash
bash scripts/run_deco_stage3b_reduced_steps.sh
```

Useful defaults:

- `PFC_STAGE3B_NUM_SAMPLES=8`
- `PFC_STAGE3B_BATCH_SIZE=4`
- `PFC_STAGE3B_STEPS=20`
- `PFC_STAGE3B_CACHE_INTERVAL=2`
- `PFC_STAGE3B_ACTIVE_T_MIN=0.2`
- `PFC_STAGE3B_ACTIVE_T_MAX=1.0`
- `PFC_STAGE3B_CACHE_UNITS=backbone_blocks`

## Outputs

Inspection:

- `logs/stage3b/deco_inspect/<run_id>/module_tree.txt`
- `logs/stage3b/deco_inspect/<run_id>/module_candidates.csv`
- `logs/stage3b/deco_inspect/<run_id>/module_candidates.json`
- `logs/stage3b/deco_inspect/<run_id>/summary.md`

Cache run:

- `logs/stage3b/deco/<run_id>/no_cache_summary.json`
- `logs/stage3b/deco/<run_id>/cache_summary.json`
- `logs/stage3b/deco/<run_id>/comparison.json`
- `logs/stage3b/deco/<run_id>/cache_stats.json`
- `logs/stage3b/deco/<run_id>/velocity_stats.jsonl`
- `logs/stage3b/deco/<run_id>/frequency_stats.jsonl`
- `logs/stage3b/deco/<run_id>/step_stats.jsonl`
- `logs/stage3b/deco/<run_id>/summary.md`

Benchmark:

- `logs/stage3b/deco_benchmark/<run_id>/benchmark_results.csv`
- `logs/stage3b/deco_benchmark/<run_id>/benchmark_results.json`
- `logs/stage3b/deco_benchmark/<run_id>/benchmark_aggregate.csv`
- `logs/stage3b/deco_benchmark/<run_id>/summary.md`

Figures:

- `outputs/stage3b/figures/deco_stage3b_speed_quality.png`
- `outputs/stage3b/figures/deco_stage3b_rel_l2_by_method.png`
- `outputs/stage3b/figures/deco_stage3b_speedup_by_method.png`
- `outputs/stage3b/figures/deco_stage3b_cache_hit_rate_by_method.png`
- `outputs/stage3b/figures/deco_stage3b_frequency_delta_by_method.png`

## Interpretation

- `speedup` compares a cached or reduced-step method against the same-seed no-cache reference.
- `same_seed_rel_l2`, MSE, MAE, and PSNR compare final tensors from identical labels and initial noise.
- `frequency_delta_bands` summarizes the frequency content of the output difference.
- A useful DeCo cache should beat reduced-step no-cache baselines at similar speed.

## Limitations

- Debug sample count only.
- No FID or large-scale evaluation.
- No adaptive online policy.
- No calibration.
- No token cache.
- The direct model path loads DeCo EMA denoiser weights and compares tensors before any full dataset evaluation.

## Next Step

Compare JiT and DeCo Stage 3 findings, then design a parameterization-aware PixelFlowCache policy instead of assuming one cache schedule transfers across x-pred and v-pred models.
