# Stage 3B2 DeCo Cache Decomposition

Stage 3B2 decomposes the Stage 3B DeCo result into output/final, decoder, and backbone cache units.

This stage does not implement token cache, adaptive policy, calibration, solver-aware cache, or full ImageNet-scale FID.

## Motivation

Stage 3B found that DeCo `all_candidates` cache beats reduced-step no-cache in a 20-step, 8-sample debug benchmark. It also found that `final`, `decoder`, and `all_candidates` rows had identical rel-L2 in that run.

Stage 3B2 tests whether final/output velocity cache dominates quality while upstream backbone cache mainly provides speed.

## Explicit Cache Specs

- `final_only`: only `dec_net.final_layer`
- `decoder_only_no_final`: only `dec_net.res_blocks.N`
- `decoder_plus_final`: decoder blocks plus final head
- `backbone_only`: only `blocks.N`
- `backbone_plus_final`: backbone blocks plus final head
- `backbone_plus_decoder_no_final`: backbone blocks plus decoder blocks, no final
- `all_candidates`: backbone blocks, decoder blocks, and final head
- `late_backbone_only:<n>`: last `n` backbone blocks
- `late_backbone_plus_final:<n>`: last `n` backbone blocks plus final head

Norm, adaLN/modulation, embedding, linear-only, tiny modules, and arbitrary nested submodules stay excluded.

## Commands

Use one GPU by default:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
export PFC_CUDA_DEVICES=0
bash scripts/run_deco_stage3b2_decomposition.sh
```

Plot and generate report tables:

```bash
DECOMP_DIR="$(ls -td logs/stage3b2/deco_decomposition/* | head -n 1)"
python scripts/plot_stage3b2_deco.py --decomposition-dir "$DECOMP_DIR"
python scripts/make_stage3b2_report_tables.py --decomposition-dir "$DECOMP_DIR"
```

Optional seed sweep and validation:

```bash
bash scripts/run_deco_stage3b2_seed_sweep.sh
bash scripts/run_deco_stage3b2_validate.sh
SWEEP_DIR="$(ls -td logs/stage3b2/deco_seed_sweep/* | head -n 1)"
VAL_DIR="$(ls -td logs/stage3b2/deco_validate/* | head -n 1)"
python scripts/plot_stage3b2_deco.py \
  --decomposition-dir "$DECOMP_DIR" \
  --seed-sweep-dir "$SWEEP_DIR" \
  --validation-dir "$VAL_DIR"
python scripts/make_stage3b2_report_tables.py \
  --decomposition-dir "$DECOMP_DIR" \
  --validation-dir "$VAL_DIR"
```

Fast validation:

```bash
export PFC_STAGE3B2_VALIDATE_FAST=1
bash scripts/run_deco_stage3b2_validate.sh
```

## Interpretation

- If `final_only` and `all_candidates` have the same rel-L2, final/output velocity cache likely dominates quality.
- If `backbone_plus_final` matches `all_candidates` quality and speed, decoder cache is likely unnecessary.
- If `backbone_only` is worse, stale backbone features with a fresh output boundary may be mismatched.
- Cache rows should be compared against reduced-step no-cache at similar speed.

## Outputs

Decomposition:

- `logs/stage3b2/deco_decomposition/<run_id>/decomposition_results.csv`
- `logs/stage3b2/deco_decomposition/<run_id>/decomposition_results.json`
- `logs/stage3b2/deco_decomposition/<run_id>/decomposition_aggregate.csv`
- `logs/stage3b2/deco_decomposition/<run_id>/summary.md`
- `logs/stage3b2/deco_decomposition/<run_id>/runs/<method_name>_seed<seed>/`

Validation and seed sweep use the same layout under `logs/stage3b2/deco_validate/` and `logs/stage3b2/deco_seed_sweep/`.

Figures are written under ignored `outputs/stage3b2/figures/`.

## Limitations

- Debug sample count only unless validation is expanded.
- No FID.
- No adaptive policy.
- No calibration.
- No token cache.

## Next Step

Use the Stage 3B2 decomposition to summarize how JiT x-pred and DeCo v-pred differ, then design a parameterization-aware PixelFlowCache policy instead of using one fixed rule across model types.
