#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import (  # noqa: E402
    get_method_metadata,
    method_cli_overrides,
    method_supports_model,
)


SCRIPT_BY_MODEL = {
    "jit": "scripts/run_jit_stage4a_generate.py",
    "deco": "scripts/run_deco_stage4a_generate.py",
    "pixelgen": "scripts/run_pixelgen_stage4a_generate.py",
}
DEFAULT_BATCH_SIZE = {"jit": 8, "deco": 4, "pixelgen": 4}
DEBUG_FLAG_BY_METHOD_TYPE = {
    "safe_cache": "--safe-debug-jsonl",
    "dynamic_cache": "--dynamic-cache-debug-jsonl",
    "forecast_cache": "--taylorseer-debug-jsonl",
    "speculative_cache": "--speca-debug-jsonl",
    "probe_cache": "--dicache-debug-jsonl",
}


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shard_debug_path(path: Path | None, shard_index: int, num_shards: int) -> Path | None:
    if path is None or num_shards <= 1:
        return path
    return path.with_name(f"{path.stem}_shard{shard_index}{path.suffix}")


def worker_command(
    args: argparse.Namespace,
    shard_index: int,
) -> list[str]:
    command = (
        [args.python_executable]
        if args.python_executable
        else ["conda", "run", "-n", args.resolved_env, "python"]
    )
    command.extend([
        SCRIPT_BY_MODEL[args.model],
        "--method",
        args.method,
        "--num-images",
        str(args.num_images),
        "--batch-size",
        str(args.resolved_batch_size),
        "--seed",
        str(args.seed),
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(shard_index),
        "--shard-mode",
        args.shard_mode,
    ])
    if args.model == "jit":
        command.extend(["--jit-ckpt-dir", str(args.jit_ckpt_dir)])
    elif args.model == "deco":
        command.extend(["--deco-ckpt", str(args.deco_ckpt)])
    else:
        command.extend(["--pixelgen-ckpt", str(args.pixelgen_ckpt)])
    command.append("--save-png" if args.save_png else "--no-save-png")
    command.append("--save-npz" if args.save_npz else "--no-save-npz")
    if args.resume:
        command.append("--resume")
    command.extend(method_cli_overrides(args.model, args.method))
    if args.resolved_safe_map is not None:
        command.extend(["--safe-map", str(args.resolved_safe_map)])
    debug = shard_debug_path(args.debug_jsonl, shard_index, args.num_shards)
    if debug is not None:
        command.extend([DEBUG_FLAG_BY_METHOD_TYPE[args.method_type], str(debug)])
    command.extend(args.resolved_worker_args)
    return command


def command_text(command: list[str], gpu: str) -> str:
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {shlex.join(command)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print or run sharded generation for JiT, DeCo, or PixelGen.")
    parser.add_argument("--model", choices=tuple(SCRIPT_BY_MODEL), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage4a/full_generation"))
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--jit-ckpt-dir", type=Path, default=Path("ckpts/JiT/JiT-B-16-256"))
    parser.add_argument("--deco-ckpt", type=Path, default=Path("ckpts/DeCo/DeCo_XL.ckpt"))
    parser.add_argument("--pixelgen-ckpt", type=Path, default=Path("ckpts/PixelGen/PixelGen_XL_160ep.ckpt"))
    parser.add_argument("--env-jit", default="jit")
    parser.add_argument("--env-deco", default="deco")
    parser.add_argument("--env-pixelgen", default="pixelgen")
    parser.add_argument("--python-executable")
    parser.add_argument("--safe-map", type=Path)
    parser.add_argument("--safe-map-quality", type=Path)
    parser.add_argument("--safe-map-speed", type=Path)
    parser.add_argument("--shard-mode", choices=("strided", "contiguous"), default="strided")
    parser.add_argument("--debug-jsonl", type=Path)
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--worker-arg",
        action="append",
        default=[],
        help="Append one complete argv token per occurrence; do not pass a shell-style compound string.",
    )
    parser.add_argument("--worker-args-json", help="JSON list of additional worker argv tokens.")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def resolve_launch_options(args: argparse.Namespace) -> None:
    if not method_supports_model(args.model, args.method):
        raise ValueError(f"Method {args.method!r} is not supported for model {args.model!r}")
    metadata = get_method_metadata(args.model, args.method)
    args.method_type = str(metadata["method_type"])
    args.resolved_batch_size = (
        DEFAULT_BATCH_SIZE[args.model] if args.batch_size is None else args.batch_size
    )
    if args.resolved_batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.resolved_env = getattr(args, f"env_{args.model}")
    if not args.python_executable and not args.resolved_env:
        raise ValueError(f"--env-{args.model} must not be empty")

    args.resolved_safe_map = None
    if args.method == "safe_bfc_quality":
        args.resolved_safe_map = args.safe_map or args.safe_map_quality
    elif args.method == "safe_bfc_speed":
        args.resolved_safe_map = args.safe_map or args.safe_map_speed
    if args.method_type == "safe_cache" and args.resolved_safe_map is None:
        raise ValueError(f"{args.method} requires --safe-map or its matching preset-specific safe map")

    if args.debug_jsonl is not None and args.method_type not in DEBUG_FLAG_BY_METHOD_TYPE:
        raise ValueError(
            f"--debug-jsonl is unsupported for {args.method} ({args.method_type}); "
            "no method-specific debug flag exists"
        )
    json_args: list[str] = []
    if args.worker_args_json:
        try:
            parsed = json.loads(args.worker_args_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--worker-args-json must be valid JSON: {exc}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("--worker-args-json must be a JSON list of strings")
        json_args = parsed
    args.resolved_worker_args = [*args.worker_arg, *json_args]


def main() -> int:
    args = build_parser().parse_args()
    resolve_launch_options(args)
    gpus = split_csv(args.gpus)
    if not gpus:
        raise SystemExit("--gpus must not be empty")
    args.num_shards = args.num_shards or len(gpus)
    if args.num_shards != len(gpus):
        raise SystemExit("--num-shards must equal the number of GPUs")
    if args.print_only and args.execute:
        raise ValueError("--print-only and --execute are mutually exclusive")
    commands = [worker_command(args, index) for index in range(args.num_shards)]
    if not args.execute:
        for gpu, command in zip(gpus, commands):
            print(command_text(command, gpu))
        return 0

    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    model_dir = {"jit": "jit", "deco": "deco", "pixelgen": "pixelgen"}[args.model]
    run_dir = args.output_root / model_dir / args.run_id / args.method
    log_root = args.log_root or run_dir / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    launcher_start = datetime.now(timezone.utc).isoformat()
    processes: list[dict[str, Any]] = []
    try:
        for shard_index, (gpu, command) in enumerate(zip(gpus, commands)):
            stdout_path = log_root / f"worker_shard{shard_index}.stdout.log"
            stderr_path = log_root / f"worker_shard{shard_index}.stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            processes.append(
                {
                    "process": process,
                    "gpu_id": gpu,
                    "command": command,
                    "command_text": command_text(command, gpu),
                    "pid": process.pid,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                    "start_time_utc": datetime.now(timezone.utc).isoformat(),
                    "start_monotonic": time.perf_counter(),
                }
            )
        failure = False
        while any(item["process"].poll() is None for item in processes):
            for item in processes:
                code = item["process"].poll()
                if code not in (None, 0):
                    failure = True
                    _terminate_running(processes)
                    break
            if failure:
                break
            time.sleep(0.1)
        for item in processes:
            item["return_code"] = item["process"].wait()
            item["end_time_utc"] = datetime.now(timezone.utc).isoformat()
            item["wall_time_sec"] = time.perf_counter() - item["start_monotonic"]
        failure = failure or any(item["return_code"] != 0 for item in processes)
    except (KeyboardInterrupt, SystemExit):
        _terminate_running(processes)
        raise
    finally:
        for item in processes:
            item["stdout_handle"].close()
            item["stderr_handle"].close()

    wall = time.perf_counter() - started
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_time_utc": launcher_start,
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "parallel_orchestration_wall_time_sec": wall,
        "model": args.model,
        "method": args.method,
        "method_type": args.method_type,
        "resolved_env": None if args.python_executable else args.resolved_env,
        "python_executable": args.python_executable,
        "resolved_batch_size": args.resolved_batch_size,
        "resolved_safe_map": str(args.resolved_safe_map) if args.resolved_safe_map else None,
        "num_shards": args.num_shards,
        "comparable_for_algorithm_speedup": False,
        "workers": [
            {key: value for key, value in item.items() if key not in {"process", "stdout_handle", "stderr_handle", "start_monotonic"}}
            for item in processes
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "parallel_launcher_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if failure:
        return 1
    merge = [
        sys.executable,
        "scripts/merge_parallel_shards.py",
        "--run-dir",
        str(run_dir),
        "--num-shards",
        str(args.num_shards),
        "--expected-images",
        str(args.num_images),
        "--method",
        args.method,
        "--model",
        args.model,
        "--launcher-meta",
        str(run_dir / "parallel_launcher_meta.json"),
        "--strict",
    ]
    try:
        merge_result = subprocess.run(
            merge, cwd=ROOT, check=False, text=True, capture_output=True, timeout=300
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Shard merge timed out: {shlex.join(merge)}\nstdout={exc.stdout}\nstderr={exc.stderr}"
        ) from exc
    if merge_result.stdout:
        print(merge_result.stdout, end="")
    if merge_result.stderr:
        print(merge_result.stderr, end="", file=sys.stderr)
    return int(merge_result.returncode)


def _terminate_running(processes: list[dict[str, Any]]) -> None:
    for item in processes:
        process = item["process"]
        if process.poll() is None:
            process.terminate()
    for item in processes:
        process = item["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
