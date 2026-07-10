#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The planner prints Safe-BFC, TaylorSeer, adapted SpeCa/DiCache-style, and comparison commands.
python "${SCRIPT_DIR}/run_jit_safe_1000_eval_plan.py" "$@"
