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
SCRIPT = {
    "jit": "scripts/run_jit_stage4a_generate.py",
    "deco": "scripts/run_deco_stage4a_generate.py",
    "pixelgen": "scripts/run_pixelgen_stage4a_generate.py",
}


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _path(value: Path | str) -> str:
    return Path(value).as_posix()


def _command(argv: list[str], *, gpu: str | None = None) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} " if gpu is not None else ""
    return prefix + shlex.join(argv)


def _env(args: argparse.Namespace, model: str) -> str:
    return str(getattr(args, f"env_{model}"))


def _batch_size(args: argparse.Namespace, model: str) -> int:
    return int(getattr(args, f"batch_size_{model}"))


def _checkpoint(args: argparse.Namespace, model: str) -> tuple[str, str]:
    if model == "jit":
        return "--jit-ckpt-dir", _path(args.jit_ckpt_dir)
    if model == "deco":
        return "--deco-ckpt", _path(args.deco_ckpt)
    return "--pixelgen-ckpt", _path(args.pixelgen_ckpt)


def _safe_map(args: argparse.Namespace, method: str) -> Path | None:
    if method == "safe_bfc_quality":
        return args.safe_map_quality
    if method == "safe_bfc_speed":
        return args.safe_map_speed
    return None


def _generation(
    args: argparse.Namespace,
    model: str,
    method: str,
    *,
    count: int,
    run_id: str,
    extra: list[str] | None = None,
) -> str:
    ckpt_flag, ckpt = _checkpoint(args, model)
    argv = [
        "conda", "run", "-n", _env(args, model), "python", SCRIPT[model],
        "--method", method,
        "--num-images", str(count),
        "--batch-size", str(_batch_size(args, model)),
        "--seed", "0",
        "--run-id", run_id,
        "--output-root", _path(args.output_root),
        ckpt_flag, ckpt,
        "--save-png", "--no-save-npz",
        *method_cli_overrides(model, method),
    ]
    safe_map = _safe_map(args, method)
    if safe_map is not None:
        argv.extend(["--safe-map", _path(safe_map)])
    argv.extend(extra or [])
    return _command(argv, gpu="0")


def _validate_plan(args: argparse.Namespace, models: list[str], methods: list[str]) -> None:
    unknown = [
        method for method in methods
        if not any(method_supports_model(model, method) for model in models)
    ]
    if unknown:
        raise ValueError(f"Methods are unsupported by the requested models: {unknown}")
    if "jit" in models:
        missing = [
            method for method in ("safe_bfc_quality", "safe_bfc_speed")
            if method in methods and _safe_map(args, method) is None
        ]
        if missing:
            raise ValueError(f"Selected Safe-BFC methods require safe maps: {missing}")
    if any(_batch_size(args, model) <= 0 for model in models):
        raise ValueError("model batch sizes must be positive")
    if any(not _env(args, model) for model in models):
        raise ValueError("requested model environments must not be empty")


def build_plan(args: argparse.Namespace) -> list[str]:
    models = _csv(args.models)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    gpus = _csv(args.gpus)
    _validate_plan(args, models, methods)
    smoke_run = f"{args.run_id}_smoke8"
    equivalence_run = f"{args.run_id}_dicache_equivalence8"
    baseline_smoke_run = f"{args.run_id}_smoke16"
    proxy_run = f"{args.run_id}_proxy1000"

    preflight = [
        "python", "scripts/preflight_experiments.py",
        "--models", ",".join(models),
        "--methods", ",".join(methods),
        "--env-jit", args.env_jit,
        "--env-deco", args.env_deco,
        "--env-pixelgen", args.env_pixelgen,
        "--jit-ckpt-dir", _path(args.jit_ckpt_dir),
        "--deco-ckpt", _path(args.deco_ckpt),
        "--pixelgen-ckpt", _path(args.pixelgen_ckpt),
        "--fid-stats", _path(args.fid_stats),
        "--required-gpus", str(len(gpus)),
        "--min-free-disk-gb", "100",
        "--strict",
        "--out", f"logs/preflight/{args.run_id}.json",
    ]
    if args.safe_map_quality is not None:
        preflight.extend(["--safe-map-quality", _path(args.safe_map_quality)])
    if args.safe_map_speed is not None:
        preflight.extend(["--safe-map-speed", _path(args.safe_map_speed)])

    lines = [
        "# PixelFlowCache server readiness plan (print-only; review every gate).",
        "# Do not advance to the next gate until the current gate passes.",
        "",
        "# Gate 0: strict base/static preflight plus independent model-environment probes",
        _command(preflight),
    ]
    probe_code = (
        "import json,sys,torch; "
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count()}))"
    )
    for model in models:
        lines.append(f"# {model} environment probe")
        lines.append(_command(["conda", "run", "-n", _env(args, model), "python", "-c", probe_code]))

    lines.extend(["", "# Gate 1: 8-image no-cache correctness smoke"])
    for model in models:
        lines.append(_generation(args, model, "no_cache_50", count=8, run_id=smoke_run))

    lines.extend([
        "",
        "# Gate 2: DiCache force-full equivalence and static manifest/index tests",
        _command([
            "conda", "run", "-n", args.env_jit, "pytest", "-q",
            "tests/test_jit_dicache_sampler_equivalence.py",
            "tests/test_resume_manifest.py",
            "tests/test_pixelgen_sampler_equivalence.py",
        ]),
    ])
    if "jit" in models and method_supports_model("jit", "dicache_style"):
        lines.append(_generation(args, "jit", "no_cache_50", count=8, run_id=equivalence_run))
        lines.append(_generation(
            args, "jit", "dicache_style", count=8, run_id=equivalence_run,
            extra=["--dicache-force-full"],
        ))
        lines.append(_command([
            "conda", "run", "-n", args.env_jit, "python",
            "scripts/evaluate_stage4b_pair_metrics.py",
            "--reference-dir", _path(args.output_root / "jit" / equivalence_run / "no_cache_50" / "images"),
            "--method-dir", _path(args.output_root / "jit" / equivalence_run / "dicache_style" / "images"),
            "--metrics", "psnr,ssim,lpips,rel_l2",
            "--out", f"logs/server_readiness/{equivalence_run}/dicache_force_full_pair_metrics.json",
        ]))
        lines.append("# Expected equivalence: PSNR inf; SSIM 1; LPIPS 0; rel_l2 0.")
    if not args.skip_safe_calibration and "jit" in models:
        lines.extend([
            "# Optional reviewed Safe calibration command; this planner never executes it.",
            _command([
                "conda", "run", "-n", args.env_jit, "python", "scripts/run_jit_safe_calibration.py",
                "--num-calibration-images", "128", "--batch-size", str(args.batch_size_jit),
                "--seed", "123", "--run-id", f"{args.run_id}_safe_calib",
                "--out-dir", f"calibrations/jit_safe/{args.run_id}_safe_calib",
            ], gpu="0"),
        ])

    lines.extend(["", "# Gate 3: 16-image baseline smoke with model-specific environments and Safe maps"])
    for model in models:
        for method in methods:
            if method == "no_cache_50" or not method_supports_model(model, method):
                continue
            lines.append(_generation(args, model, method, count=16, run_id=baseline_smoke_run))

    jit_methods = [method for method in methods if method_supports_model("jit", method)]
    timing = [
        "python", "scripts/run_jit_single_gpu_timing_plan.py",
        "--methods", ",".join(jit_methods),
        "--num-images", "64", "--batch-size", str(args.batch_size_jit),
        "--warmup-batches", "2", "--repeats", "3", "--gpu", "0",
        "--env", args.env_jit,
        "--jit-ckpt-dir", _path(args.jit_ckpt_dir),
        "--run-id", f"{args.run_id}_timing",
        "--print-only",
    ]
    if args.safe_map_quality is not None:
        timing.extend(["--safe-map-quality", _path(args.safe_map_quality)])
    if args.safe_map_speed is not None:
        timing.extend(["--safe-map-speed", _path(args.safe_map_speed)])
    lines.extend([
        "",
        "# Gate 4: synchronized single-GPU JiT timing (64 images, two warmups, three repeats)",
        _command(timing),
        "",
        "# Gate 5: four-GPU 1k proxy generation; throughput only, never algorithm speedup",
    ])
    for model in models:
        for method in methods:
            if not method_supports_model(model, method):
                continue
            launch = [
                "python", "scripts/run_parallel_generate.py",
                "--model", model, "--method", method,
                "--gpus", ",".join(gpus), "--num-shards", str(len(gpus)),
                "--num-images", "1000", "--batch-size", str(_batch_size(args, model)),
                "--run-id", proxy_run, "--output-root", _path(args.output_root),
                "--env-jit", args.env_jit, "--env-deco", args.env_deco,
                "--env-pixelgen", args.env_pixelgen,
                "--jit-ckpt-dir", _path(args.jit_ckpt_dir),
                "--deco-ckpt", _path(args.deco_ckpt),
                "--pixelgen-ckpt", _path(args.pixelgen_ckpt),
            ]
            safe_map = _safe_map(args, method)
            if safe_map is not None:
                launch.extend(["--safe-map", _path(safe_map)])
            launch.append("--execute")
            lines.append(_command(launch))

    lines.extend(["", "# Gate 6: FID/IS proxy evaluation in the JiT evaluation environment"])
    for model in models:
        for method in methods:
            if not method_supports_model(model, method):
                continue
            lines.append(_command([
                "conda", "run", "-n", args.env_jit, "python", "scripts/evaluate_stage4a_fid.py",
                "--fake-dir", _path(args.output_root / model / proxy_run / method / "images"),
                "--fid-stats", _path(args.fid_stats),
                "--backend", "torch_fidelity", "--metrics", "fid,is",
                "--expected-images", "1000", "--proxy-result",
                "--out", f"logs/server_readiness/{proxy_run}/fid/{model}/{method}/fid_results.json",
            ]))

    lines.extend(["", "# Gate 7: JiT paired metrics and cross-model FID collection"])
    if "jit" in models:
        for method in jit_methods:
            if method == "no_cache_50":
                continue
            lines.append(_command([
                "conda", "run", "-n", args.env_jit, "python",
                "scripts/evaluate_stage4b_pair_metrics.py",
                "--reference-dir", _path(args.output_root / "jit" / proxy_run / "no_cache_50" / "images"),
                "--method-dir", _path(args.output_root / "jit" / proxy_run / method / "images"),
                "--metrics", "psnr,ssim,lpips,rel_l2",
                "--out", f"logs/server_readiness/{proxy_run}/pair_metrics/jit/{method}/pair_metrics.json",
            ]))
    lines.append(_command([
        "conda", "run", "-n", args.env_jit, "python", "scripts/collect_stage4a_fid_results.py",
        "--root", _path(args.output_root),
        "--fid-root", f"logs/server_readiness/{proxy_run}/fid",
        "--run-id", proxy_run, "--num-images", "1000",
        "--out-dir", f"logs/server_readiness/{proxy_run}/summary",
    ]))
    for model in models:
        if model != "jit":
            lines.append(f"# {model}: no generic paired-metric collector is defined; use FID collection above.")
    lines.extend([
        "",
        "# Go/No-Go: only after Gates 0-7 pass may a reviewed 50k plan be considered.",
    ])
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the server readiness gates; never executes commands.")
    parser.add_argument("--models", default="jit,deco,pixelgen")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--run-id", default="stage5a_server_readiness_seed0")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/server_readiness"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--env-jit", default="jit")
    parser.add_argument("--env-deco", default="deco")
    parser.add_argument("--env-pixelgen", default="pixelgen")
    parser.add_argument("--batch-size-jit", type=int, default=8)
    parser.add_argument("--batch-size-deco", type=int, default=4)
    parser.add_argument("--batch-size-pixelgen", type=int, default=4)
    parser.add_argument("--safe-map-quality", type=Path)
    parser.add_argument("--safe-map-speed", type=Path)
    parser.add_argument("--jit-ckpt-dir", type=Path, default=Path("ckpts/JiT/JiT-B-16-256"))
    parser.add_argument("--deco-ckpt", type=Path, default=Path("ckpts/DeCo/DeCo_XL.ckpt"))
    parser.add_argument("--pixelgen-ckpt", type=Path, default=Path("ckpts/PixelGen/PixelGen_XL_160ep.ckpt"))
    parser.add_argument("--fid-stats", type=Path, default=Path("third_party/JiT/fid_stats/jit_in256_stats.npz"))
    parser.add_argument("--skip-safe-calibration", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="Accepted for clarity; planner is always print-only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = _csv(args.models)
    if not models or any(model not in SCRIPT for model in models):
        raise ValueError("--models must contain jit, deco, and/or pixelgen")
    if not _csv(args.gpus):
        raise ValueError("--gpus must not be empty")
    print("\n".join(build_plan(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
