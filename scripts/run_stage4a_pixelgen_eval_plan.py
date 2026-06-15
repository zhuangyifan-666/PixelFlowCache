#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PIXELGEN_METHODS = [
    "no_cache_50",
    "bfc_quality_t02_08",
    "bfc_speed_t02_10",
    "reduced_steps_30",
    "reduced_steps_35",
    "bfc_speed_t02_09",
]


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _method_filter(requested: list[str] | None) -> list[str]:
    if requested is None:
        return list(PIXELGEN_METHODS)
    unknown = sorted(set(requested) - set(PIXELGEN_METHODS))
    if unknown:
        raise ValueError(f"Unknown PixelGen Stage 4A methods: {unknown}")
    return [method for method in PIXELGEN_METHODS if method in requested]


def _reference_arg(args: argparse.Namespace) -> str:
    if args.fid_stats:
        return f"--fid-stats {_q(args.fid_stats)}"
    if args.real_dir:
        return f"--real-dir {_q(args.real_dir)}"
    return "--real-dir /path/to/imagenet/val"


def build_plan(args: argparse.Namespace) -> list[str]:
    requested_methods = _split_csv(args.methods) if args.methods else None
    methods = _method_filter(requested_methods)
    run_id = args.run_id or f"stage4a_pixelgen_n{args.num_images}_seed{args.seed}"
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# PixelGen Stage 4A command plan. Review and run commands manually.",
        f"cd {_q(ROOT)}",
        f"export PIXELGEN_CKPT={_q(args.pixelgen_ckpt)}",
        f"export RUN_ID={_q(run_id)}",
        f"export OUT_ROOT={_q(args.output_root)}",
        "",
        "# First round: four independent single-GPU processes, no DDP.",
    ]
    eval_refs = []
    for index, method in enumerate(methods):
        if index == 4:
            commands.extend(["", "# Second round: remaining PixelGen methods."])
        gpu = index % 4
        commands.append(
            f"CUDA_VISIBLE_DEVICES={gpu} conda run -n {_q(args.pixelgen_env)} python "
            "scripts/run_pixelgen_stage4a_generate.py "
            f"--method {_q(method)} --num-images {args.num_images} "
            f"--batch-size {args.batch_size_pixelgen} --seed {args.seed} "
            "--run-id \"${RUN_ID}\" --output-root \"${OUT_ROOT}\" "
            "--pixelgen-ckpt \"${PIXELGEN_CKPT}\" "
            f"--cfg {args.cfg} --timeshift {args.timeshift} "
            f"--guidance-interval-min {args.guidance_interval_min} "
            f"--guidance-interval-max {args.guidance_interval_max} "
            "--save-png --no-save-npz --resume"
        )
        fake_dir = args.output_root / "pixelgen" / run_id / method / "images"
        out = ROOT / "logs/stage4a/fid" / run_id / "pixelgen" / method / "fid_results.json"
        eval_refs.append((fake_dir, out))

    commands.extend(["", "# FID/IS evaluation commands. Run after generation completes."])
    reference = _reference_arg(args)
    for fake_dir, out in eval_refs:
        commands.append(
            "CUDA_VISIBLE_DEVICES=${PFC_CUDA_DEVICES:-0} "
            f"conda run -n {_q(args.fid_env)} python scripts/evaluate_stage4a_fid.py "
            f"--fake-dir {_q(fake_dir)} {reference} "
            f"--backend auto --metrics fid,is --batch-size {args.fid_batch_size} --out {_q(out)}"
        )

    commands.extend(
        [
            "",
            "# Summarize after FID JSON files are written.",
            f"conda run -n {_q(args.fid_env)} python scripts/collect_stage4a_fid_results.py "
            f"--root {_q(args.output_root)} --fid-root {_q(ROOT / 'logs/stage4a/fid')} "
            f"--run-id {_q(run_id)} --num-images {args.num_images} --model pixelgen "
            f"--out-dir {_q(ROOT / 'logs/stage4a/summary' / run_id)}",
            f"conda run -n {_q(args.fid_env)} python scripts/plot_stage4a_full_eval.py "
            f"--summary-dir {_q(ROOT / 'logs/stage4a/summary' / run_id)} --num-images {args.num_images}",
        ]
    )
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print PixelGen Stage 4A generation and FID commands without running them.")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size-pixelgen", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/stage4a/full_generation")
    parser.add_argument("--pixelgen-ckpt", type=Path, default=ROOT / "ckpts/PixelGen/PixelGen_XL.ckpt")
    parser.add_argument("--pixelgen-env", default="pixelgen")
    parser.add_argument("--fid-env", default="jit")
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--fid-stats", type=Path)
    parser.add_argument("--methods")
    parser.add_argument("--cfg", type=float, default=2.25)
    parser.add_argument("--timeshift", type=float, default=2.0)
    parser.add_argument("--guidance-interval-min", type=float, default=0.1)
    parser.add_argument("--guidance-interval-max", type=float, default=0.9)
    parser.add_argument("--out-script", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.batch_size_pixelgen <= 0:
        raise ValueError("--batch-size-pixelgen must be positive")
    commands = build_plan(args)
    text = "\n".join(commands) + "\n"
    print(text)
    if args.out_script:
        args.out_script.parent.mkdir(parents=True, exist_ok=True)
        args.out_script.write_text(text, encoding="utf-8")
        args.out_script.chmod(0o755)
        print(f"Wrote command plan: {args.out_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
