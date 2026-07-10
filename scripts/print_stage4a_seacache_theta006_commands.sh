#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "# Deprecated compatibility entrypoint: commands now come from the canonical registry-backed planner."
python scripts/run_stage4a_full_eval_plan.py \
  --models jit,deco \
  --methods seacache_style \
  --num-images "${PFC_NUM_IMAGES:-50000}" \
  "$@"
