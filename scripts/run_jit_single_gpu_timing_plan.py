#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import (  # noqa: E402
    list_methods_for_model,
    method_cli_overrides,
    method_supports_model,
)


DEFAULT_METHODS = list_methods_for_model(
    "jit", tags={"reference", "main_baseline", "proxy_default"}
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_commands(args: argparse.Namespace) -> list[str]:
    methods = _csv(args.methods) if args.methods else DEFAULT_METHODS
    unknown = [method for method in methods if not method_supports_model("jit", method)]
    if unknown:
        raise ValueError(f"Unsupported JiT methods: {unknown}")
    commands = [
        "# JiT synchronized single-GPU timing suite (print-only).",
        "# Sampling speedups from this suite are comparable; four-GPU proxy wall time is not.",
    ]
    for repeat in range(1, args.repeats + 1):
        for method in methods:
            run_id = f"{args.run_id}_r{repeat:02d}"
            argv = [
                "conda", "run", "-n", args.env,
                "python", "scripts/run_jit_stage4a_generate.py",
                "--method", method,
                "--num-images", str(args.num_images),
                "--batch-size", str(args.batch_size),
                "--warmup-batches", str(args.warmup_batches),
                "--seed", str(args.seed),
                "--run-id", run_id,
                "--output-root", str(args.output_root),
                "--jit-ckpt-dir", str(args.jit_ckpt_dir),
                "--device", "cuda",
                "--no-save-png",
                "--no-save-npz",
                "--num-shards", "1",
                *method_cli_overrides("jit", method),
            ]
            if method == "safe_bfc_quality":
                argv.extend(["--safe-map", str(args.safe_map_quality)])
            elif method == "safe_bfc_speed":
                argv.extend(["--safe-map", str(args.safe_map_speed)])
            commands.append(f"CUDA_VISIBLE_DEVICES={args.gpu} {shlex.join(argv)}")
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the JiT single-GPU timing suite; never executes it.")
    parser.add_argument("--methods")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--env", default="jit")
    parser.add_argument("--run-id", default="stage5a_jit_single_gpu_timing_seed0")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/timing"))
    parser.add_argument("--jit-ckpt-dir", type=Path, default=Path("ckpts/JiT/JiT-B-16-256"))
    parser.add_argument(
        "--safe-map-quality",
        type=Path,
        default=Path("calibrations/jit_safe/stage5a_jit_safe_calib128_seed123/safe_map_quality.json"),
    )
    parser.add_argument(
        "--safe-map-speed",
        type=Path,
        default=Path("calibrations/jit_safe/stage5a_jit_safe_calib128_seed123/safe_map_speed.json"),
    )
    parser.add_argument("--print-only", action="store_true", help="Accepted for symmetry; always print-only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.num_images, args.batch_size, args.repeats) <= 0 or args.warmup_batches < 0:
        raise ValueError("num-images, batch-size and repeats must be positive; warmup-batches must be non-negative")
    print("\n".join(build_commands(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
