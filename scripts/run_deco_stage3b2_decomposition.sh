#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage3b2
LOG="${ROOT}/logs/stage3b2/deco_stage3b2_decomposition_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 3B2 DeCo cache decomposition"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "nvidia-smi failed; cannot run GPU Stage 3B2 decomposition safely."
    exit 1
  fi
else
  echo "nvidia-smi not found; cannot run GPU Stage 3B2 decomposition safely."
  exit 1
fi

if [[ -z "${PFC_CUDA_DEVICES:-}" ]]; then
  PFC_CUDA_DEVICES="$(pfc_select_cuda_devices | cut -d',' -f1)"
fi
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PFC_STAGE3B2_NUM_SAMPLES="${PFC_STAGE3B2_NUM_SAMPLES:-8}"
export PFC_STAGE3B2_BATCH_SIZE="${PFC_STAGE3B2_BATCH_SIZE:-4}"
export PFC_STAGE3B2_REFERENCE_STEPS="${PFC_STAGE3B2_REFERENCE_STEPS:-20}"
export PFC_STAGE3B2_TIMING_REPEATS="${PFC_STAGE3B2_TIMING_REPEATS:-2}"
export PFC_STAGE3B2_WARMUP_RUNS="${PFC_STAGE3B2_WARMUP_RUNS:-1}"
export PFC_STAGE3B2_SAVE_DIAGNOSTICS="${PFC_STAGE3B2_SAVE_DIAGNOSTICS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deco

CMD=(python scripts/run_deco_stage3b2_decomposition.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Final command: ${CMD[*]}"
"${CMD[@]}"
