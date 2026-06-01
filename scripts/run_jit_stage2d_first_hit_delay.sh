#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage2d
LOG="${ROOT}/logs/stage2d/jit_stage2d_first_hit_delay_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 2D JiT first-hit delay"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"
if command -v nvidia-smi >/dev/null 2>&1; then
  set +e
  nvidia-smi
  nvidia_smi_status=$?
  set -e
  if [[ "${nvidia_smi_status}" -ne 0 ]]; then
    echo "nvidia-smi failed; cannot run GPU Stage 2D first-hit delay safely."
    exit 1
  fi
else
  echo "nvidia-smi not found; cannot run GPU Stage 2D first-hit delay safely."
  exit 1
fi

if [[ -z "${PFC_CUDA_DEVICES:-}" ]]; then
  PFC_CUDA_DEVICES="$(pfc_select_cuda_devices | cut -d',' -f1)"
fi
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PFC_STAGE2D_FIRST_HIT_NUM_SAMPLES="${PFC_STAGE2D_FIRST_HIT_NUM_SAMPLES:-8}"
export PFC_STAGE2D_BATCH_SIZE="${PFC_STAGE2D_BATCH_SIZE:-4}"
export PFC_STAGE2D_FIRST_HIT_STEPS="${PFC_STAGE2D_FIRST_HIT_STEPS:-20}"
export PFC_STAGE2D_SEED="${PFC_STAGE2D_SEED:-0}"
export PFC_STAGE2D_FIRST_HIT_TIMING_REPEATS="${PFC_STAGE2D_FIRST_HIT_TIMING_REPEATS:-3}"
export PFC_STAGE2D_WARMUP_RUNS="${PFC_STAGE2D_WARMUP_RUNS:-1}"
export PFC_STAGE2D_FIRST_HIT_WARMUPS="${PFC_STAGE2D_FIRST_HIT_WARMUPS:-0,1,2}"
export PFC_STAGE2D_FIRST_HIT_CACHE_LAYERS="${PFC_STAGE2D_FIRST_HIT_CACHE_LAYERS:-all}"
export PFC_STAGE2D_FIRST_HIT_CACHE_INTERVAL="${PFC_STAGE2D_FIRST_HIT_CACHE_INTERVAL:-2}"
export PFC_STAGE2D_FIRST_HIT_ACTIVE_T_MIN="${PFC_STAGE2D_FIRST_HIT_ACTIVE_T_MIN:-0.1}"
export PFC_STAGE2D_FIRST_HIT_ACTIVE_T_MAX="${PFC_STAGE2D_FIRST_HIT_ACTIVE_T_MAX:-0.8}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jit

CMD=(python scripts/run_jit_stage2d_first_hit_delay.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Final command: ${CMD[*]}"
"${CMD[@]}"
