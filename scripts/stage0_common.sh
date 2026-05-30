#!/usr/bin/env bash

pfc_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." >/dev/null 2>&1
  pwd
}

pfc_count_cuda_devices() {
  local devices="$1"
  if [[ -z "${devices}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${devices}"
}

pfc_select_cuda_devices() {
  if [[ -n "${PFC_CUDA_DEVICES:-}" ]]; then
    echo "${PFC_CUDA_DEVICES}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0,1"
    return
  fi

  local max_used_mb="${PFC_GPU_MEMORY_USED_MAX_MB:-1024}"
  local selected
  selected="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' -v max_used="${max_used_mb}" '
          {
            gsub(/ /, "", $1);
            gsub(/ /, "", $2);
            if ($2 + 0 <= max_used + 0) {
              if (count < 2) {
                if (count > 0) {
                  printf ",";
                }
                printf "%s", $1;
                count += 1;
              }
            }
          }
        '
  )"

  if [[ -n "${selected}" ]]; then
    echo "${selected}"
  else
    echo "0,1"
  fi
}

pfc_detect_imagenet_root() {
  local candidates=(
    "${PFC_IMAGENET_PATH:-}"
    "/mnt/iset/nfs-main/public/datasets/ILSVRC"
    "/mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC"
    "/mnt/iset/nfs-main/public/datasets/ILSVRC2012"
    "/mnt/iset/nfs-main/public/datasets/ImageNet"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -d "${candidate}/train" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

pfc_detect_jit_ckpt_dir() {
  local root="$1"
  if [[ -n "${PFC_JIT_CKPT_DIR:-}" && -f "${PFC_JIT_CKPT_DIR}/checkpoint-last.pth" ]]; then
    echo "${PFC_JIT_CKPT_DIR}"
    return 0
  fi

  local expected="${root}/ckpts/JiT/JiT-B-16-256"
  if [[ -f "${expected}/checkpoint-last.pth" ]]; then
    echo "${expected}"
    return 0
  fi

  local found
  found="$(find "${root}/ckpts/JiT" "${root}/ckpts" -type f -name 'checkpoint-last.pth' -print 2>/dev/null | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    dirname "${found}"
    return 0
  fi

  found="$(find "${root}/ckpts/JiT" "${root}/ckpts" -type f -name '*.pth' -print 2>/dev/null | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    mkdir -p "${expected}"
    ln -sfn "${found}" "${expected}/checkpoint-last.pth"
    echo "${expected}"
    return 0
  fi

  return 1
}

pfc_detect_deco_ckpt() {
  local root="$1"
  if [[ -n "${PFC_DECO_CKPT:-}" && -f "${PFC_DECO_CKPT}" ]]; then
    echo "${PFC_DECO_CKPT}"
    return 0
  fi

  local expected="${root}/ckpts/DeCo/imagenet256_epoch800.ckpt"
  if [[ -f "${expected}" ]]; then
    echo "${expected}"
    return 0
  fi

  local found
  found="$(find "${root}/ckpts/DeCo" "${root}/ckpts" -type f -name '*imagenet*256*800*.ckpt' -print 2>/dev/null | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  found="$(find "${root}/ckpts/DeCo" "${root}/ckpts" -type f -name '*.ckpt' -print 2>/dev/null | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

