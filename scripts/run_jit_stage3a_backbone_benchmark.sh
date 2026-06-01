#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"
cd "${ROOT}"

mkdir -p logs/stage3a
LOG="${ROOT}/logs/stage3a/jit_stage3a_backbone_benchmark_stdout.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 3A JiT BackboneCache benchmark"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "nvidia-smi failed; cannot run GPU Stage 3A benchmark safely."
    exit 1
  fi
else
  echo "nvidia-smi not found; cannot run GPU Stage 3A benchmark safely."
  exit 1
fi

if [[ -z "${PFC_CUDA_DEVICES:-}" ]]; then
  PFC_CUDA_DEVICES="$(pfc_select_cuda_devices | cut -d',' -f1)"
fi
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PFC_STAGE3A_SEEDS="${PFC_STAGE3A_SEEDS:-0,1,2}"
export PFC_STAGE3A_NUM_SAMPLES="${PFC_STAGE3A_NUM_SAMPLES:-16}"
export PFC_STAGE3A_BATCH_SIZE="${PFC_STAGE3A_BATCH_SIZE:-4}"
export PFC_STAGE3A_REFERENCE_STEPS="${PFC_STAGE3A_REFERENCE_STEPS:-50}"
export PFC_STAGE3A_TIMING_REPEATS="${PFC_STAGE3A_TIMING_REPEATS:-2}"
export PFC_STAGE3A_WARMUP_RUNS="${PFC_STAGE3A_WARMUP_RUNS:-1}"
export PFC_STAGE3A_REDUCED_STEPS="${PFC_STAGE3A_REDUCED_STEPS:-30,35,40}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jit

CMD=(python scripts/run_jit_stage3a_backbone_benchmark.py)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Final command: ${CMD[*]}"
"${CMD[@]}"
