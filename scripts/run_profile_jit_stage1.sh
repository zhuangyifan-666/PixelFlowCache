#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage1
LOG="${ROOT}/logs/stage1/jit_profile_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 1 JiT profiling"
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
export PFC_PROFILE_NUM_SAMPLES="${PFC_PROFILE_NUM_SAMPLES:-4}"
export PFC_PROFILE_BATCH_SIZE="${PFC_PROFILE_BATCH_SIZE:-4}"
export PFC_PROFILE_STEPS="${PFC_PROFILE_STEPS:-10}"
export PFC_PROFILE_SEED="${PFC_PROFILE_SEED:-0}"

CMD=(python scripts/profile_jit_stage1.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Final command: conda run --no-capture-output -n jit ${CMD[*]}"
conda run --no-capture-output -n jit "${CMD[@]}"

