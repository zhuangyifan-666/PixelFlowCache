# Stage 4A Full Inference And FID-Ready Evaluation

Stage 4A adds an end-to-end generation and FID-ready evaluation pipeline for BoundaryFlowCache on JiT and DeCo. It provides scripts, dry-run checks, command-plan generation, result collection, and plotting.

Codex does not launch long inference or compute FID in this stage. The user manually runs generated commands after reviewing GPU availability and storage.

## What Stage 4A Implements

- JiT full-generation script for no-cache, BoundaryFlowCache, and reduced-step methods.
- DeCo full-generation script for no-cache, BoundaryFlowCache, and reduced-step methods.
- FID/IS/KID evaluation entry point with backend auto-detection.
- ImageNet reference validation/preparation helper.
- Command-plan generator that prints shell commands without executing them.
- Result collector and plotting scripts for completed runs.

## What Stage 4A Does Not Do Automatically

- It does not launch 1k, 5k, or 50k generation.
- It does not compute FID/IS/KID by itself.
- It does not submit background jobs.
- It does not download weights.
- It does not implement token cache, adaptive policy, calibration, or a new cache policy.

## Methods

JiT:

- `no_cache_50`
- `bfc_quality_t02_08`
- `bfc_speed_t02_10`
- `reduced_steps_35`
- `reduced_steps_30`

DeCo:

- `no_cache_50`
- `bfc_all_candidates_t02_10`
- `bfc_backbone_plus_final_t02_10`
- `reduced_steps_35`
- `reduced_steps_30`

## Suggested Run Order

1. 100-image smoke generation.
2. 1000-image proxy FID.
3. 5000-image proxy FID.
4. Optional 50k full ImageNet-scale FID.

Small-N FID is only a proxy and should not be reported as paper-scale quality.

## Print Command Plans

Smoke:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
bash scripts/print_stage4a_smoke_commands.sh
```

Proxy:

```bash
bash scripts/print_stage4a_proxy_fid_commands.sh
```

Full 50k:

```bash
bash scripts/print_stage4a_full_50k_commands.sh
```

Write a launch script without running it:

```bash
python scripts/run_stage4a_full_eval_plan.py \
  --models jit,deco \
  --num-images 100 \
  --out-script scripts/launch_stage4a_smoke_100.sh
```

Review the generated script, set `PFC_CUDA_DEVICES`, then run commands manually.

## Manual Generation Examples

JiT:

```bash
export PFC_CUDA_DEVICES=0
CUDA_VISIBLE_DEVICES=$PFC_CUDA_DEVICES conda run -n jit python scripts/run_jit_stage4a_generate.py \
  --method bfc_speed_t02_10 \
  --num-images 100 \
  --batch-size 8 \
  --seed 0 \
  --save-png \
  --no-save-npz \
  --resume
```

DeCo:

```bash
export PFC_CUDA_DEVICES=0
CUDA_VISIBLE_DEVICES=$PFC_CUDA_DEVICES conda run -n deco python scripts/run_deco_stage4a_generate.py \
  --method bfc_all_candidates_t02_10 \
  --num-images 100 \
  --batch-size 4 \
  --seed 0 \
  --save-png \
  --no-save-npz \
  --resume
```

## FID Evaluation

Use a generated image folder as `--fake-dir`. Use either ImageNet val images or a precomputed FID stats file as the reference.

```bash
conda run -n jit python scripts/evaluate_stage4a_fid.py \
  --fake-dir outputs/stage4a/full_generation/jit/stage4a_n100_seed0/bfc_speed_t02_10/images \
  --real-dir /mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC/val \
  --backend auto \
  --metrics fid,is \
  --out logs/stage4a/fid/stage4a_n100_seed0/jit/bfc_speed_t02_10/fid_results.json
```

The script refuses to use the Stage 0 JiT `torch_fidelity` stub. Install a real backend if none is available:

```bash
pip install torch-fidelity
```

or:

```bash
pip install clean-fid
```

## Reference Preparation

Validate ImageNet reference layout:

```bash
python scripts/prepare_stage4a_imagenet_reference.py --dry-run
```

Prepare symlinked reference images only when explicitly needed:

```bash
python scripts/prepare_stage4a_imagenet_reference.py --symlink
```

## Collect And Plot Results

After generation and FID JSON files exist:

```bash
python scripts/collect_stage4a_fid_results.py \
  --root outputs/stage4a/full_generation \
  --fid-root logs/stage4a/fid

SUMMARY_DIR="$(ls -td logs/stage4a/summary/* | head -n 1)"
python scripts/plot_stage4a_full_eval.py --summary-dir "$SUMMARY_DIR"
```

## Output Layout

Generation:

- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/images/`
- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/labels.json`
- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/labels.csv`
- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/manifest.jsonl`
- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/generation_meta.json`
- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/latency.json`
- `outputs/stage4a/full_generation/<model>/<run_id>/<method>/cache_stats.json`

FID:

- `logs/stage4a/fid/<run_id>/<model>/<method>/fid_results.json`
- `logs/stage4a/fid/<run_id>/<model>/<method>/fid_results.csv`

Summary:

- `logs/stage4a/summary/<run_id>/stage4a_results.csv`
- `logs/stage4a/summary/<run_id>/stage4a_results.json`
- `logs/stage4a/summary/<run_id>/summary.md`

## Known Risks

- FID is unstable at 100/1000 images.
- 5000 images is still a proxy run.
- 50k images is the paper-scale target and needs storage planning.
- The real ImageNet val reference may require a compatible folder layout or precomputed stats.
- Generated classes should stay balanced; Stage 4A uses deterministic class-balanced labels.
- Do not put `scripts/jit_stubs` on `PYTHONPATH` for real FID.

## Storage Notes

PNG-only generation is the default. `--save-npz` is intended for smoke/proxy debugging and is guarded against large 50k runs.

All generation, FID, and summary artifacts are under ignored `outputs/` and `logs/`.

## Next Steps

After Stage 4A, add perceptual metrics such as LPIPS/DINO and produce paper-ready full evaluation tables from the completed user-run outputs.
