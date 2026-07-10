#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

thresholds=(0.02 0.06 0.10 0.20 0.40 0.80)

slug_delta() {
  local value="$1"
  value="${value/./p}"
  echo "$value"
}

print_jit() {
  local method="$1"
  local label="$2"
  for threshold in "${thresholds[@]}"; do
    local slug
    slug="$(slug_delta "$threshold")"
    echo "CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/run_jit_stage4a_generate.py --method ${method} --dynamic-cache-threshold ${threshold} --num-images 1000 --batch-size 8 --seed 0 --run-id stage4a_jit_${label}_n1000_delta${slug}_seed0 --save-png --no-save-npz"
  done
}

print_deco() {
  local method="$1"
  local label="$2"
  for threshold in "${thresholds[@]}"; do
    local slug
    slug="$(slug_delta "$threshold")"
    echo "CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n deco python scripts/run_deco_stage4a_generate.py --method ${method} --dynamic-cache-threshold ${threshold} --num-images 1000 --batch-size 4 --seed 0 --run-id stage4a_deco_${label}_n1000_delta${slug}_seed0 --save-png --no-save-npz"
  done
}

echo "# JiT SeaCache-style 1000-image threshold sweep"
print_jit "seacache_style" "seacache"
echo
echo "# JiT TeaCache-style 1000-image threshold sweep"
print_jit "teacache_style" "teacache"
echo
echo "# DeCo SeaCache-style 1000-image threshold sweep"
print_deco "seacache_style" "seacache"
echo
echo "# DeCo TeaCache-style 1000-image threshold sweep"
print_deco "teacache_style" "teacache"
echo
echo "# 50k templates after selecting a threshold"
echo "CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/run_jit_stage4a_generate.py --method seacache_style --dynamic-cache-threshold 0.06 --num-images 50000 --batch-size 8 --seed 0 --run-id stage4a_jit_seacache_theta0p06_n50000_seed0 --save-png --no-save-npz"
echo "CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n deco python scripts/run_deco_stage4a_generate.py --method seacache_style --dynamic-cache-threshold 0.06 --num-images 50000 --batch-size 4 --seed 0 --run-id stage4a_deco_seacache_theta0p06_n50000_seed0 --save-png --no-save-npz"
