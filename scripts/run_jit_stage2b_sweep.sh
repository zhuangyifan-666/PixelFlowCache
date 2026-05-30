#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage2b
LOG="${ROOT}/logs/stage2b/jit_stage2b_sweep_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 2B JiT timestep-window cache sweep"
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
export PFC_STAGE2B_SWEEP_FAST="${PFC_STAGE2B_SWEEP_FAST:-1}"
export PFC_STAGE2B_NUM_SAMPLES="${PFC_STAGE2B_NUM_SAMPLES:-8}"
export PFC_STAGE2B_BATCH_SIZE="${PFC_STAGE2B_BATCH_SIZE:-4}"
export PFC_STAGE2B_STEPS="${PFC_STAGE2B_STEPS:-20}"
export PFC_STAGE2B_TIMING_REPEATS="${PFC_STAGE2B_TIMING_REPEATS:-3}"
export PFC_STAGE2B_WARMUP_RUNS="${PFC_STAGE2B_WARMUP_RUNS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jit

CMD=(python scripts/run_jit_stage2b_sweep.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PFC_STAGE2B_SWEEP_FAST=${PFC_STAGE2B_SWEEP_FAST}"
echo "Final command: ${CMD[*]}"
"${CMD[@]}"
