#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage2c
LOG="${ROOT}/logs/stage2c/jit_stage2c_validate_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 2C JiT validation"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "nvidia-smi failed; cannot run GPU Stage 2C validation safely."
    exit 1
  fi
else
  echo "nvidia-smi not found; cannot run GPU Stage 2C validation safely."
  exit 1
fi

if [[ -z "${PFC_CUDA_DEVICES:-}" ]]; then
  PFC_CUDA_DEVICES="$(pfc_select_cuda_devices | cut -d',' -f1)"
fi
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PFC_STAGE2C_NUM_SAMPLES="${PFC_STAGE2C_NUM_SAMPLES:-32}"
export PFC_STAGE2C_BATCH_SIZE="${PFC_STAGE2C_BATCH_SIZE:-4}"
export PFC_STAGE2C_STEPS="${PFC_STAGE2C_STEPS:-50}"
export PFC_STAGE2C_SEED="${PFC_STAGE2C_SEED:-0}"
export PFC_STAGE2C_TIMING_REPEATS="${PFC_STAGE2C_TIMING_REPEATS:-3}"
export PFC_STAGE2C_WARMUP_RUNS="${PFC_STAGE2C_WARMUP_RUNS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jit

CMD=(python scripts/run_jit_stage2c_validate.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Final command: ${CMD[*]}"
"${CMD[@]}"
