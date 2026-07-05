#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


METHODS = [
    "no_cache_50",
    "safe_bfc_quality",
    "safe_bfc_speed",
    "seacache_style",
    "reduced_steps_35",
    "reduced_steps_30",
]
PAIR_METHODS = [method for method in METHODS if method != "no_cache_50"]


def _cmd(lines: list[str]) -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return " \\\n  ".join(lines)


def _device_prefix(prefix: str, gpu: int | str) -> str:
    return f"{prefix}={gpu}"


def _parallel_generation_command(args: argparse.Namespace, method: str, extra: list[str] | None = None) -> str:
    lines = [
        "conda run -n jit python scripts/run_jit_parallel_generate.py",
        "--execute",
        f"--method {method}",
        f"--num-images {args.num_images}",
        f"--batch-size {args.batch_size}",
        f"--seed {args.seed}",
        "--run-id ${RUN_ID}",
        "--output-root ${OUT_ROOT}",
        "--jit-ckpt-dir ${JIT_CKPT_DIR}",
        f"--gpus {args.gpus}",
        f"--num-shards {args.num_shards}",
    ]
    if extra:
        lines.extend(extra)
    lines.extend(["--save-png", "--no-save-npz", "--resume"])
    return _cmd(lines)


def _fid_command(args: argparse.Namespace, method: str, *, use_real_dir: bool = False) -> str:
    metric_source = f"--real-dir {args.real_dir}" if use_real_dir else "--fid-stats ${FID_STATS}"
    return _cmd(
        [
            "conda run -n jit python scripts/evaluate_stage4a_fid.py",
            f"--fake-dir ${{OUT_ROOT}}/jit/${{RUN_ID}}/{method}/images",
            metric_source,
            "--backend auto",
            "--metrics fid,is",
            "--batch-size 64",
            "--device cuda",
            f"--out logs/stage5a/fid/${{RUN_ID}}/jit/{method}/fid_results.json",
        ]
    )


def _pair_command(method: str) -> str:
    return _cmd(
        [
            "conda run -n jit python scripts/evaluate_stage4b_pair_metrics.py",
            "--reference-dir ${REF_DIR}",
            f"--method-dir ${{OUT_ROOT}}/jit/${{RUN_ID}}/{method}/images",
            "--metrics psnr,ssim,lpips,rel_l2",
            "--batch-size 32",
            "--device cuda",
            f"--out logs/stage5a/pair_metrics/${{RUN_ID}}/jit/{method}/pair_metrics.json",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print JiT Safe-BFC 1000-image proxy experiment commands.")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="stage5a_jit_safe_n1000_seed0")
    parser.add_argument("--calibration-run-id", default="stage5a_jit_safe_calib128_seed123")
    parser.add_argument("--calibration-images", type=int, default=128)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=123)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--quality-lambda", type=float, default=0.5)
    parser.add_argument("--speed-lambda", type=float, default=1.0)
    parser.add_argument("--out-root", default="outputs/stage4a/full_generation")
    parser.add_argument("--calib-root", default="calibrations/jit_safe")
    parser.add_argument("--jit-ckpt-dir", default="ckpts/JiT/JiT-B-16-256")
    parser.add_argument("--fid-stats", default="third_party/JiT/fid_stats/jit_in256_stats.npz")
    parser.add_argument("--real-dir")
    parser.add_argument("--device-prefix", default="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--print-only", action="store_true", help="Accepted for symmetry; commands are always print-only.")
    parser.add_argument("--dry-run", action="store_true", help="Accepted for symmetry; commands are always print-only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calib_dir = str(Path(args.calib_root) / args.calibration_run_id)

    print("# Part 1: Safe calibration command")
    print(f"export RUN_ID={args.run_id}")
    print(f"export CALIB_ID={args.calibration_run_id}")
    print("export CALIB_DIR=${CALIB_ROOT}/${CALIB_ID}" if args.calib_root == "${CALIB_ROOT}" else f"export CALIB_DIR={calib_dir}")
    print(f"export JIT_CKPT_DIR={args.jit_ckpt_dir}")
    print()
    print(
        _cmd(
            [
                f"{_device_prefix(args.device_prefix, 0)} conda run -n jit python scripts/run_jit_safe_calibration.py",
                f"--num-calibration-images {args.calibration_images}",
                f"--batch-size {args.calibration_batch_size}",
                f"--seed {args.calibration_seed}",
                "--run-id ${CALIB_ID}",
                "--jit-ckpt-dir ${JIT_CKPT_DIR}",
                "--out-dir ${CALIB_DIR}",
                "--boundary-groups whole_backbone",
                f"--max-age {args.max_age}",
                f"--quantile {args.quantile}",
                f"--quality-lambda {args.quality_lambda}",
                f"--speed-lambda {args.speed_lambda}",
                "--lte-floor 1e-3",
            ]
        )
    )
    print()

    print("# Part 2: Safe map density check")
    print("python scripts/check_safe_map_density.py \\")
    print("  --safe-map ${CALIB_DIR}/safe_map_quality.json")
    print()
    print("python scripts/check_safe_map_density.py \\")
    print("  --safe-map ${CALIB_DIR}/safe_map_speed.json")
    print()

    print("# Part 3: forced-safe smoke test command")
    print(
        _cmd(
            [
                "python scripts/make_forced_safe_map.py",
                "--out ${CALIB_DIR}/forced_safe_smoke.json",
                "--jit-blocks 12",
                "--steps 50",
                "--max-age 1",
                "--branches global,cond,uncond",
                "--solver-stages euler",
            ]
        )
    )
    print()
    print(
        _cmd(
            [
                f"{_device_prefix(args.device_prefix, 0)} conda run -n jit python scripts/run_jit_stage4a_generate.py",
                "--method safe_bfc_speed",
                "--num-images 16",
                "--batch-size 8",
                f"--seed {args.seed}",
                "--run-id stage5a_jit_forced_safe_smoke16_seed0",
                "--output-root outputs/stage4a/full_generation",
                "--jit-ckpt-dir ${JIT_CKPT_DIR}",
                "--safe-map ${CALIB_DIR}/forced_safe_smoke.json",
                "--safe-debug-jsonl logs/stage5a/debug/forced_safe_smoke16.jsonl",
                "--save-png",
                "--no-save-npz",
                "--resume",
                "--allow-empty-safe-map",
            ]
        )
    )
    print()

    print("# Part 4: calibrated safe smoke test command")
    print(
        _cmd(
            [
                f"{_device_prefix(args.device_prefix, 0)} conda run -n jit python scripts/run_jit_stage4a_generate.py",
                "--method safe_bfc_speed",
                "--num-images 32",
                "--batch-size 8",
                f"--seed {args.seed}",
                "--run-id stage5a_jit_calibrated_safe_smoke32_seed0",
                "--output-root outputs/stage4a/full_generation",
                "--jit-ckpt-dir ${JIT_CKPT_DIR}",
                "--safe-map ${CALIB_DIR}/safe_map_speed.json",
                "--safe-debug-jsonl logs/stage5a/debug/calibrated_safe_smoke32.jsonl",
                "--save-png",
                "--no-save-npz",
                "--resume",
            ]
        )
    )
    print()

    print("# Part 5: 1000 image four-card generation commands")
    print(f"export RUN_ID={args.run_id}")
    print(f"export OUT_ROOT={args.out_root}")
    print(f"export JIT_CKPT_DIR={args.jit_ckpt_dir}")
    print(f"export CALIB_DIR={calib_dir}")
    print()
    print("# no_cache_50")
    print(_parallel_generation_command(args, "no_cache_50"))
    print()
    print("# safe_bfc_quality")
    print(_parallel_generation_command(args, "safe_bfc_quality", ["--safe-map ${CALIB_DIR}/safe_map_quality.json"]))
    print()
    print("# safe_bfc_speed")
    print(_parallel_generation_command(args, "safe_bfc_speed", ["--safe-map ${CALIB_DIR}/safe_map_speed.json"]))
    print()
    print("# seacache_style")
    print(
        _parallel_generation_command(
            args,
            "seacache_style",
            ["--dynamic-cache-threshold 0.06", "--sea-beta 2.0", "--sea-proxy-downsample 64"],
        )
    )
    print()
    print("# reduced_steps_35")
    print(_parallel_generation_command(args, "reduced_steps_35"))
    print()
    print("# reduced_steps_30")
    print(_parallel_generation_command(args, "reduced_steps_30"))
    print()

    print("# Part 6: FID/IS evaluation commands")
    print(f"export RUN_ID={args.run_id}")
    print(f"export OUT_ROOT={args.out_root}")
    print(f"export FID_STATS={args.fid_stats}")
    print()
    for method in METHODS:
        print(_fid_command(args, method))
        print()
    if args.real_dir:
        print("# Optional real-dir FID/IS variants")
        for method in METHODS:
            print(_fid_command(args, method, use_real_dir=True))
            print()

    print("# Part 7: paired PSNR/SSIM/LPIPS commands")
    print(f"export RUN_ID={args.run_id}")
    print(f"export OUT_ROOT={args.out_root}")
    print("export REF_DIR=${OUT_ROOT}/jit/${RUN_ID}/no_cache_50/images")
    print()
    for method in PAIR_METHODS:
        print(_pair_command(method))
        print()

    print("# Part 8: summary collection command")
    print(
        _cmd(
            [
                "conda run -n jit python scripts/collect_jit_safe_1000_results.py",
                f"--run-id {args.run_id}",
                f"--output-root {args.out_root}",
                "--fid-root logs/stage5a/fid",
                "--pair-root logs/stage5a/pair_metrics",
                f"--methods {','.join(METHODS)}",
                f"--out-dir logs/stage5a/summary/{args.run_id}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
