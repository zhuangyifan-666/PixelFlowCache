#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_density_fn():
    module_path = ROOT / "pfc/cache/safe_map_policy.py"
    spec = importlib.util.spec_from_file_location("safe_map_policy_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_safe_map_density


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print Safe-BFC safe-map density.")
    parser.add_argument("--safe-map", type=Path, required=True)
    parser.add_argument("--min-density", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.safe_map.read_text(encoding="utf-8"))
    compute_safe_map_density = _load_density_fn()
    density = compute_safe_map_density(payload)
    print(json.dumps(density, indent=2, sort_keys=True))
    if args.min_density is not None and density["safe_density"] < args.min_density:
        print(
            f"Safe density {density['safe_density']:.6g} is below --min-density {args.min_density:.6g}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
