#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/run_stage4a_full_eval_plan.py --models jit,deco --num-images "${PFC_STAGE4A_PROXY_NUM_IMAGES:-1000}"
