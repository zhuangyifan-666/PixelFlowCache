#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate deco
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PFC_CUDA_DEVICES="${PFC_CUDA_DEVICES:-0}"
resume_args=()
if [[ "${PFC_RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi

python scripts/run_deco_stage4a_generate.py \
  --method seacache_style \
  --dynamic-cache-threshold 0.06 \
  --num-images 50000 \
  --batch-size 4 \
  --seed 0 \
  --run-id stage4a_deco_seacache_theta0p06_n50000_seed0 \
  --output-root outputs/stage4a/full_generation \
  --save-png \
  --no-save-npz \
  "${resume_args[@]}"

conda run -n jit python scripts/evaluate_stage4a_fid.py \
  --fake-dir outputs/stage4a/full_generation/deco/stage4a_deco_seacache_theta0p06_n50000_seed0/seacache_style/images \
  --fid-stats third_party/JiT/fid_stats/jit_in256_stats.npz \
  --backend torch_fidelity \
  --metrics fid,is \
  --batch-size 64 \
  --expected-images 50000 \
  --out logs/stage4a/fid/stage4a_deco_seacache_theta0p06_n50000_seed0/seacache_style/fid_results.json
