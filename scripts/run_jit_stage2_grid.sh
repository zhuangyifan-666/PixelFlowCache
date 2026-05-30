#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage2
LOG="${ROOT}/logs/stage2/jit_stage2_grid_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 2 JiT fixed-interval block cache grid"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

if [[ -z "${PFC_CUDA_DEVICES:-}" ]]; then
  PFC_CUDA_DEVICES="$(pfc_select_cuda_devices | cut -d',' -f1)"
fi
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PFC_STAGE2_GRID_FAST="${PFC_STAGE2_GRID_FAST:-1}"
export PFC_STAGE2_NUM_SAMPLES="${PFC_STAGE2_NUM_SAMPLES:-8}"
export PFC_STAGE2_BATCH_SIZE="${PFC_STAGE2_BATCH_SIZE:-4}"
export PFC_STAGE2_STEPS="${PFC_STAGE2_STEPS:-20}"
export PFC_STAGE2_WARMUP_RUNS="${PFC_STAGE2_WARMUP_RUNS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jit

CMD=(python scripts/run_jit_stage2_grid.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PFC_STAGE2_GRID_FAST=${PFC_STAGE2_GRID_FAST}"
echo "Final command: ${CMD[*]}"
"${CMD[@]}"
