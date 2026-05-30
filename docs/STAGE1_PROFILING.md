# Stage 1 Profiling

Stage 1 collects profiling evidence for future cache design. It does not implement cache acceleration, skip block computation, or change the official model outputs.

## Research Questions

- JiT: which Transformer blocks are smooth across adjacent sampling steps?
- JiT: how do x0 prediction norms, converted velocity norms, and `1 / (1 - t)` amplification evolve?
- DeCo: which block or branch candidates are temporally smooth?
- DeCo: how does direct velocity output evolve over time?
- Both models: how do low, mid, and high FFT frequency ratios change across sampling?
- Both models: how stable are conditional, unconditional, and CFG-combined velocity outputs?

## Server Setup

- Project: `/mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache`
- ImageNet root: `/mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC`
- JiT env: `conda activate jit`
- DeCo env: `conda activate deco`
- Default GPU policy: one GPU for profiling

## Run Commands

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
python scripts/run_stage0_smoke.py
pytest -q

export PFC_CUDA_DEVICES=0
bash scripts/run_profile_jit_stage1.sh
bash scripts/run_profile_deco_stage1.sh
```

Summarize and plot a run:

```bash
python scripts/summarize_stage1_profiles.py --run-dir logs/stage1/jit/<run_id>
python scripts/plot_stage1_profiles.py --run-dir logs/stage1/jit/<run_id>

python scripts/summarize_stage1_profiles.py --run-dir logs/stage1/deco/<run_id>
python scripts/plot_stage1_profiles.py --run-dir logs/stage1/deco/<run_id>
```

## Expected Outputs

Each run writes:

- `meta.json`
- `step_stats.jsonl`
- `feature_stats.jsonl`
- `velocity_stats.jsonl`
- `frequency_stats.jsonl`
- `summary.json`

Runtime outputs are ignored:

- `logs/stage1/...`
- `outputs/stage1/previews/...`
- `outputs/stage1/figures/...`

## Interpretation

Lower feature `rel_l2_delta` means smoother module outputs across adjacent steps. Smooth modules are later cache candidates, but Stage 1 does not cache them.

High velocity norm or high high-frequency energy ratio marks cache-sensitive steps. For JiT, large `1 / max(1 - t, eps)` warns that x-pred errors can be amplified in late steps after conversion to velocity.

CFG records split conditional, unconditional, and CFG-combined velocity where accessible. Large CFG residual changes suggest future cache policies should treat guidance carefully.

## Limitations

- Small sample count by default.
- No FID or quality metrics.
- Hook timing is diagnostic only and should not be used as final latency.
- DeCo module selection is name-based and may need refinement after reading `module_candidates.json`.
- No cache speedup is implemented.

## Next Step

Analyze Stage 1 profiles, then design Stage 2 fixed-interval and baseline cache policies using the observed smoothness, velocity, CFG, and frequency evidence.

