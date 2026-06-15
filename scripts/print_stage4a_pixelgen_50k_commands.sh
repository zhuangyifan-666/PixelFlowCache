#!/usr/bin/env bash
set -euo pipefail

python scripts/run_stage4a_pixelgen_eval_plan.py --num-images 50000 "$@"
