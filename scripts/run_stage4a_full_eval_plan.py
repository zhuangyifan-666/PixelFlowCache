#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import list_methods_for_model, method_cli_overrides  # noqa: E402


PLAN_TAGS = {"reference", "main_baseline", "proxy_default", "final_50k"}
JIT_METHODS = list_methods_for_model("jit", tags=PLAN_TAGS)
DECO_METHODS = list_methods_for_model("deco", tags=PLAN_TAGS)


def _q(value: str | Path) -> str:
    text = value.as_posix() if isinstance(value, Path) else str(value)
    return shlex.quote(text)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _method_filter(methods: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return methods
    selected = [method for method in methods if method in requested]
    unknown = sorted(set(requested) - set(JIT_METHODS) - set(DECO_METHODS))
    if unknown:
        raise ValueError(f"Unknown Stage 4A methods: {unknown}")
    return selected


def build_plan(args: argparse.Namespace) -> list[str]:
    models = set(_split_csv(args.models))
    requested_methods = _split_csv(args.methods) if args.methods else None
    run_id = f"stage4a_n{args.num_images}_seed{args.seed}"
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Stage 4A command plan. Review and run commands manually.",
        "# Set PFC_CUDA_DEVICES before running, for example: export PFC_CUDA_DEVICES=0",
        f"cd {_q(ROOT)}",
        "",
    ]
    eval_refs = []
    if "jit" in models:
        jit_methods = _method_filter(JIT_METHODS, requested_methods)
        required_safe_maps = {
            "safe_bfc_quality": args.safe_map_quality,
            "safe_bfc_speed": args.safe_map_speed,
        }
        missing_safe_maps = [
            method for method in jit_methods
            if method in required_safe_maps and required_safe_maps[method] is None
        ]
        if missing_safe_maps:
            raise ValueError(f"Selected Safe-BFC methods require safe maps: {missing_safe_maps}")
        for method in jit_methods:
            overrides = method_cli_overrides("jit", method)
            if method in required_safe_maps:
                overrides.extend(["--safe-map", required_safe_maps[method].as_posix()])
            method_args = "" if not overrides else " " + shlex.join(overrides)
            commands.append(
                "CUDA_VISIBLE_DEVICES=${PFC_CUDA_DEVICES:-0} conda run -n jit python "
                f"scripts/run_jit_stage4a_generate.py --method {_q(method)} "
                f"--num-images {args.num_images} --batch-size {args.batch_size_jit} --seed {args.seed} "
                f"--run-id {_q(run_id)} --output-root {_q(args.output_root)}{method_args} --save-png --no-save-npz"
                + (" --resume" if args.resume else "")
            )
            fake_dir = args.output_root / "jit" / run_id / method / "images"
            out = ROOT / "logs/stage4a/fid" / run_id / "jit" / method / "fid_results.json"
            eval_refs.append((fake_dir, out))
    if "deco" in models:
        for method in _method_filter(DECO_METHODS, requested_methods):
            commands.append(
                "CUDA_VISIBLE_DEVICES=${PFC_CUDA_DEVICES:-0} conda run -n deco python "
                f"scripts/run_deco_stage4a_generate.py --method {_q(method)} "
                f"--num-images {args.num_images} --batch-size {args.batch_size_deco} --seed {args.seed} "
                f"--run-id {_q(run_id)} --output-root {_q(args.output_root)} --save-png --no-save-npz"
                + (" --resume" if args.resume else "")
            )
            fake_dir = args.output_root / "deco" / run_id / method / "images"
            out = ROOT / "logs/stage4a/fid" / run_id / "deco" / method / "fid_results.json"
            eval_refs.append((fake_dir, out))
    commands.extend(["", "# FID/IS evaluation commands. Run after generation completes."])
    for fake_dir, out in eval_refs:
        reference = ""
        if args.fid_stats:
            reference = f" --fid-stats {_q(args.fid_stats)}"
        elif args.real_dir:
            reference = f" --real-dir {_q(args.real_dir)}"
        else:
            reference = " --real-dir \"${IMAGENET_VAL_DIR:?set IMAGENET_VAL_DIR}\""
        proxy_args = f" --expected-images {args.num_images}" + (" --proxy-result" if args.num_images < 50000 else "")
        commands.append(
            "CUDA_VISIBLE_DEVICES=${PFC_CUDA_DEVICES:-0} conda run -n jit python "
            f"scripts/evaluate_stage4a_fid.py --fake-dir {_q(fake_dir)}{reference} "
            f"--backend auto --metrics fid,is --batch-size 64{proxy_args} --out {_q(out)}"
        )
    commands.extend(
        [
            "",
            "# Summarize after FID JSON files are written.",
            "conda run -n jit python scripts/collect_stage4a_fid_results.py "
            f"--root {_q(args.output_root)} --fid-root {_q(ROOT / 'logs/stage4a/fid')} "
            f"--run-id {_q(run_id)} --num-images {args.num_images} "
            f"--out-dir {_q(ROOT / 'logs/stage4a/summary' / run_id)}",
            "conda run -n jit python scripts/plot_stage4a_full_eval.py "
            f"--summary-dir {_q(ROOT / 'logs/stage4a/summary' / run_id)} --num-images {args.num_images}",
        ]
    )
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print Stage 4A full generation and FID command plan without running it.")
    parser.add_argument("--models", default="jit,deco")
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--batch-size-jit", type=int, default=8)
    parser.add_argument("--batch-size-deco", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/stage4a/full_generation")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--fid-stats", type=Path)
    parser.add_argument("--methods")
    parser.add_argument("--safe-map-quality", type=Path)
    parser.add_argument("--safe-map-speed", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-script", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
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
