#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"

mkdir -p "${ROOT}/logs/stage0"
LOG="${ROOT}/logs/stage0/deco_official_baseline.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 0 DeCo official no-cache debug baseline"
echo "Repository: ${ROOT}"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot run DeCo GPU baseline."
  exit 1
fi
nvidia-smi

PFC_DECO_DIR="${PFC_DECO_DIR:-${ROOT}/third_party/DeCo}"
PFC_DECO_CKPT="$(pfc_detect_deco_ckpt "${ROOT}")" || {
  echo "Missing DeCo checkpoint. Expected ${ROOT}/ckpts/DeCo/imagenet256_epoch800.ckpt"
  exit 1
}
PFC_DECO_CONFIG="${PFC_DECO_CONFIG:-${ROOT}/configs/deco_stage0_debug.yaml}"
PFC_CUDA_DEVICES="$(pfc_select_cuda_devices)"
PFC_DECO_OUT_DIR="${PFC_DECO_OUT_DIR:-${ROOT}/outputs/stage0/deco_official_debug}"

if [[ ! -f "${PFC_DECO_CKPT}" ]]; then
  echo "DeCo checkpoint not found: ${PFC_DECO_CKPT}"
  exit 1
fi
if [[ ! -f "${PFC_DECO_CONFIG}" ]]; then
  echo "DeCo debug config not found: ${PFC_DECO_CONFIG}"
  exit 1
fi

DINO_CKPT="${PFC_DINOV2_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}"
if [[ ! -f "${DINO_CKPT}" ]]; then
  echo "Warning: DINOv2 checkpoint is not present at ${DINO_CKPT}."
  echo "If DeCo predict instantiates the training encoder, this run may fail. Not downloading automatically."
fi

mkdir -p "${PFC_DECO_OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1

echo "Selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DeCo repo=${PFC_DECO_DIR}"
echo "DeCo checkpoint=${PFC_DECO_CKPT}"
echo "DeCo config=${PFC_DECO_CONFIG}"
echo "Output dir=${PFC_DECO_OUT_DIR}"
echo "Log=${LOG}"

RUNNER=()
if command -v conda >/dev/null 2>&1 && conda run -n deco python -c "import sys" >/dev/null 2>&1; then
  RUNNER=(conda run --no-capture-output -n deco)
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found."
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate deco
fi

CMD=(
  python
  main.py
  predict
  -c "${PFC_DECO_CONFIG}"
  --ckpt_path "${PFC_DECO_CKPT}"
  --trainer.default_root_dir "${PFC_DECO_OUT_DIR}"
)

echo "Final command:"
printf ' %q' "${RUNNER[@]}" "${CMD[@]}"
echo

cd "${PFC_DECO_DIR}"
"${RUNNER[@]}" "${CMD[@]}"

echo "DeCo debug baseline completed."
