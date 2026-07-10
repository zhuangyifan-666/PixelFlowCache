#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python scripts/run_stage4a_pixelgen_eval_plan.py --num-images 50000 "$@"
