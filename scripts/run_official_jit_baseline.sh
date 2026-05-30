#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/stage0_common.sh"

mkdir -p "${ROOT}/logs/stage0"
LOG="${ROOT}/logs/stage0/jit_official_baseline.log"
exec > >(tee "${LOG}") 2>&1

echo "Stage 0 JiT official no-cache debug baseline"
echo "Repository: ${ROOT}"
date -u +"UTC timestamp: %Y-%m-%dT%H:%M:%SZ"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot run JiT GPU baseline."
  exit 1
fi
nvidia-smi

PFC_JIT_DIR="${PFC_JIT_DIR:-${ROOT}/third_party/JiT}"
PFC_JIT_CKPT_DIR="$(pfc_detect_jit_ckpt_dir "${ROOT}")" || {
  echo "Missing JiT checkpoint. Expected ${ROOT}/ckpts/JiT/JiT-B-16-256/checkpoint-last.pth"
  exit 1
}
PFC_IMAGENET_PATH="$(pfc_detect_imagenet_root)" || {
  echo "Could not detect ImageNet ImageFolder root with train/."
  echo "Checked /mnt/iset/nfs-main/public/datasets/ILSVRC and nested candidates."
  exit 1
}
PFC_CUDA_DEVICES="$(pfc_select_cuda_devices)"
PFC_NPROC="${PFC_NPROC:-$(pfc_count_cuda_devices "${PFC_CUDA_DEVICES}")}"
PFC_NUM_IMAGES="${PFC_NUM_IMAGES:-16}"
PFC_GEN_BSZ="${PFC_GEN_BSZ:-8}"
PFC_JIT_MODEL="${PFC_JIT_MODEL:-JiT-B/16}"
PFC_IMG_SIZE="${PFC_IMG_SIZE:-256}"
PFC_NOISE_SCALE="${PFC_NOISE_SCALE:-1.0}"
PFC_CFG="${PFC_CFG:-3.0}"
PFC_CLASS_NUM="${PFC_CLASS_NUM:-1000}"
PFC_JIT_OUT_DIR="${PFC_JIT_OUT_DIR:-${ROOT}/outputs/stage0/jit_official_debug}"
PFC_JIT_ENTRYPOINT="${PFC_JIT_ENTRYPOINT:-${ROOT}/scripts/jit_official_debug_no_fid.py}"

if [[ ! -f "${PFC_JIT_CKPT_DIR}/checkpoint-last.pth" ]]; then
  echo "JiT checkpoint directory does not contain checkpoint-last.pth: ${PFC_JIT_CKPT_DIR}"
  exit 1
fi
if [[ ! -d "${PFC_IMAGENET_PATH}/train" ]]; then
  echo "PFC_IMAGENET_PATH must point to a root containing train/: ${PFC_IMAGENET_PATH}"
  exit 1
fi
if [[ "${PFC_NPROC}" -lt 1 || "${PFC_NPROC}" -gt 2 ]]; then
  echo "PFC_NPROC must be 1 or 2 for Stage 0, got ${PFC_NPROC}"
  exit 1
fi

mkdir -p "${PFC_JIT_OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${PFC_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PFC_JIT_DIR

echo "Selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PFC_NPROC=${PFC_NPROC}"
echo "JiT repo=${PFC_JIT_DIR}"
echo "JiT checkpoint dir=${PFC_JIT_CKPT_DIR}"
echo "ImageNet root=${PFC_IMAGENET_PATH}"
echo "Output dir=${PFC_JIT_OUT_DIR}"
echo "Log=${LOG}"

RUNNER=()
if command -v conda >/dev/null 2>&1 && conda run -n jit python -c "import sys" >/dev/null 2>&1; then
  RUNNER=(conda run --no-capture-output -n jit)
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found."
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate jit
fi

CMD=(
  torchrun
  "--nproc_per_node=${PFC_NPROC}"
  "--nnodes=1"
  "--node_rank=0"
  "${PFC_JIT_ENTRYPOINT}"
  --model "${PFC_JIT_MODEL}"
  --img_size "${PFC_IMG_SIZE}"
  --noise_scale "${PFC_NOISE_SCALE}"
  --gen_bsz "${PFC_GEN_BSZ}"
  --batch_size "${PFC_GEN_BSZ}"
  --num_images "${PFC_NUM_IMAGES}"
  --class_num "${PFC_CLASS_NUM}"
  --cfg "${PFC_CFG}"
  --interval_min 0.1
  --interval_max 1.0
  --num_workers 2
  --output_dir "${PFC_JIT_OUT_DIR}"
  --resume "${PFC_JIT_CKPT_DIR}"
  --data_path "${PFC_IMAGENET_PATH}"
  --evaluate_gen
)

echo "Final command:"
printf ' %q' "${RUNNER[@]}" "${CMD[@]}"
echo

cd "${PFC_JIT_DIR}"
"${RUNNER[@]}" "${CMD[@]}"

echo "JiT debug baseline completed."
