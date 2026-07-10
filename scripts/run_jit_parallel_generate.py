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


ROOT = Path(__file__).resolve().parents[1]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _cmd_text(command: list[str], gpu: str) -> str:
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {shlex.join(command)}"


def make_shard_debug_path(
    path: Path | None,
    shard_index: int,
    num_shards: int = 2,
) -> Path | None:
    if path is None or num_shards <= 1:
        return path
    return path.with_name(f"{path.stem}_shard{shard_index}{path.suffix}")


def _base_worker_command(args: argparse.Namespace, shard_index: int, python_bin: str = "python") -> list[str]:
    command = [
        python_bin,
        "scripts/run_jit_stage4a_generate.py",
        "--method",
        args.method,
        "--num-images",
        str(args.num_images),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
        "--jit-ckpt-dir",
        str(args.jit_ckpt_dir),
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(shard_index),
        "--shard-mode",
        args.shard_mode,
    ]
    if args.safe_map:
        command.extend(["--safe-map", str(args.safe_map)])
    if args.allow_empty_safe_map:
        command.append("--allow-empty-safe-map")
    if args.dynamic_cache_threshold is not None:
        command.extend(["--dynamic-cache-threshold", str(args.dynamic_cache_threshold)])
    dynamic_debug = make_shard_debug_path(
        args.dynamic_cache_debug_jsonl,
        shard_index,
        args.num_shards,
    )
    if dynamic_debug is not None:
        command.extend(["--dynamic-cache-debug-jsonl", str(dynamic_debug)])
    safe_debug = make_shard_debug_path(args.safe_debug_jsonl, shard_index, args.num_shards)
    if safe_debug is not None:
        command.extend(["--safe-debug-jsonl", str(safe_debug)])
    if args.taylorseer_interval is not None:
        command.extend(["--taylorseer-interval", str(args.taylorseer_interval)])
    if args.taylorseer_max_order is not None:
        command.extend(["--taylorseer-max-order", str(args.taylorseer_max_order)])
    if args.taylorseer_refresh_first_n_steps is not None:
        command.extend(["--taylorseer-refresh-first-n-steps", str(args.taylorseer_refresh_first_n_steps)])
    if args.taylorseer_refresh_last_n_steps is not None:
        command.extend(["--taylorseer-refresh-last-n-steps", str(args.taylorseer_refresh_last_n_steps)])
    taylorseer_debug = make_shard_debug_path(
        args.taylorseer_debug_jsonl,
        shard_index,
        args.num_shards,
    )
    if taylorseer_debug is not None:
        command.extend(["--taylorseer-debug-jsonl", str(taylorseer_debug)])
    if args.taylorseer_min_history is not None:
        command.extend(["--taylorseer-min-history", str(args.taylorseer_min_history)])
    if args.taylorseer_clone_forecast:
        command.append("--taylorseer-clone-forecast")
    for option, value in (
        ("--speca-max-order", args.speca_max_order),
        ("--speca-first-full-steps", args.speca_first_full_steps),
        ("--speca-base-threshold", args.speca_base_threshold),
        ("--speca-decay-rate", args.speca_decay_rate),
        ("--speca-min-threshold", args.speca_min_threshold),
        ("--speca-min-forecast-steps", args.speca_min_forecast_steps),
        ("--speca-max-forecast-steps", args.speca_max_forecast_steps),
        ("--speca-error-metric", args.speca_error_metric),
        ("--speca-branch-aggregation", args.speca_branch_aggregation),
        ("--speca-verifier-module", args.speca_verifier_module),
        ("--speca-min-history", args.speca_min_history),
        (
            "--speca-debug-jsonl",
            make_shard_debug_path(args.speca_debug_jsonl, shard_index, args.num_shards),
        ),
        ("--speca-eps", args.speca_eps if args.method == "speca_style" else None),
        (
            "--speca-max-error-samples",
            args.speca_max_error_samples if args.method == "speca_style" else None,
        ),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if args.method == "speca_style" and args.speca_clone_forecast:
        command.append("--speca-clone-forecast")
    for option, value in (
        ("--dicache-probe-depth", args.dicache_probe_depth),
        ("--dicache-reuse-threshold", args.dicache_reuse_threshold),
        ("--dicache-error-choice", args.dicache_error_choice),
        ("--dicache-branch-aggregation", args.dicache_branch_aggregation),
        ("--dicache-ret-ratio", args.dicache_ret_ratio),
        ("--dicache-gamma-min", args.dicache_gamma_min),
        ("--dicache-gamma-max", args.dicache_gamma_max),
        ("--dicache-eps", args.dicache_eps),
        ("--dicache-max-stat-samples", args.dicache_max_stat_samples),
        (
            "--dicache-debug-jsonl",
            make_shard_debug_path(args.dicache_debug_jsonl, shard_index, args.num_shards),
        ),
    ):
        if value is not None:
            command.extend([option, str(value)])
    for enabled, positive, negative in (
        (
            args.dicache_force_last_step_full,
            "--dicache-force-last-step-full",
            "--no-dicache-force-last-step-full",
        ),
        (args.dicache_dcta_enabled, "--dicache-dcta", "--no-dicache-dcta"),
    ):
        if enabled is not None:
            command.append(positive if enabled else negative)
    if args.dicache_clone_history:
        command.append("--dicache-clone-history")
    if args.dicache_force_full:
        command.append("--dicache-force-full")
    if args.method == "dicache_style":
        command.append(
            "--dicache-share-cfg-prefix"
            if args.dicache_share_cfg_prefix
            else "--no-dicache-share-cfg-prefix"
        )
        command.extend(
            ["--dicache-schedule-variant", args.dicache_schedule_variant]
        )
    command.extend(["--sea-beta", str(args.sea_beta), "--sea-proxy-downsample", str(args.sea_proxy_downsample)])
    command.append("--save-png" if args.save_png else "--no-save-png")
    command.append("--save-npz" if args.save_npz else "--no-save-npz")
    if args.resume:
        command.append("--resume")
    return command


def _merge_command(args: argparse.Namespace, python_bin: str = "python") -> list[str]:
    run_dir = Path(args.output_root) / "jit" / args.run_id / args.method
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    command = [
        python_bin,
        "scripts/merge_jit_parallel_shards.py",
        "--run-dir",
        str(run_dir),
        "--num-shards",
        str(args.num_shards),
        "--expected-images",
        str(args.num_images),
        "--method",
        args.method,
        "--launcher-meta",
        str(run_dir / "parallel_launcher_meta.json"),
        "--strict",
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or print sharded JiT generation workers.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage4a/full_generation"))
    parser.add_argument("--jit-ckpt-dir", type=Path, default=Path("ckpts/JiT/JiT-B-16-256"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-mode", choices=("strided", "contiguous"), default="strided")
    parser.add_argument("--safe-map", type=Path)
    parser.add_argument("--safe-debug-jsonl", type=Path)
    parser.add_argument("--dynamic-cache-threshold", type=float)
    parser.add_argument("--dynamic-cache-debug-jsonl", type=Path)
    parser.add_argument("--taylorseer-interval", type=int)
    parser.add_argument("--taylorseer-max-order", type=int)
    parser.add_argument("--taylorseer-debug-jsonl", type=Path)
    parser.add_argument("--taylorseer-refresh-first-n-steps", type=int)
    parser.add_argument("--taylorseer-refresh-last-n-steps", type=int)
    parser.add_argument("--taylorseer-min-history", type=int)
    parser.add_argument("--taylorseer-clone-forecast", action="store_true")
    parser.add_argument("--speca-max-order", type=int)
    parser.add_argument("--speca-first-full-steps", type=int)
    parser.add_argument("--speca-base-threshold", type=float)
    parser.add_argument("--speca-decay-rate", type=float)
    parser.add_argument("--speca-min-threshold", type=float)
    parser.add_argument("--speca-min-forecast-steps", type=int)
    parser.add_argument("--speca-max-forecast-steps", type=int)
    parser.add_argument(
        "--speca-error-metric",
        choices=("l1", "l2", "relative_l1", "relative_l2", "cosine_error"),
    )
    parser.add_argument("--speca-branch-aggregation", choices=("mean", "max"))
    parser.add_argument("--speca-verifier-module")
    parser.add_argument("--speca-min-history", type=int)
    parser.add_argument("--speca-debug-jsonl", type=Path)
    parser.add_argument("--speca-clone-forecast", action="store_true")
    parser.add_argument("--speca-eps", type=float, default=1e-10)
    parser.add_argument("--speca-max-error-samples", type=int, default=4096)
    parser.add_argument("--dicache-probe-depth", type=int)
    parser.add_argument("--dicache-reuse-threshold", type=float)
    parser.add_argument("--dicache-error-choice", choices=("delta_y", "delta_minus"))
    parser.add_argument("--dicache-branch-aggregation", choices=("mean", "max"))
    parser.add_argument("--dicache-ret-ratio", type=float)
    parser.add_argument(
        "--dicache-force-last-step-full",
        dest="dicache_force_last_step_full",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-dicache-force-last-step-full",
        dest="dicache_force_last_step_full",
        action="store_false",
    )
    parser.add_argument("--dicache-dcta", dest="dicache_dcta_enabled", action="store_true", default=None)
    parser.add_argument("--no-dicache-dcta", dest="dicache_dcta_enabled", action="store_false")
    parser.add_argument("--dicache-gamma-min", type=float)
    parser.add_argument("--dicache-gamma-max", type=float)
    parser.add_argument("--dicache-eps", type=float)
    parser.add_argument("--dicache-max-stat-samples", type=int)
    parser.add_argument("--dicache-debug-jsonl", type=Path)
    parser.add_argument("--dicache-clone-history", action="store_true")
    parser.add_argument("--dicache-force-full", action="store_true")
    parser.add_argument(
        "--dicache-share-cfg-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dicache-schedule-variant",
        choices=("released_flux_compat",),
        default="released_flux_compat",
    )
    parser.add_argument("--sea-beta", type=float, default=2.0)
    parser.add_argument("--sea-proxy-downsample", type=int, default=64)
    parser.add_argument("--save-png", dest="save_png", action="store_true", default=True)
    parser.add_argument("--no-save-png", dest="save_png", action="store_false")
    parser.add_argument("--save-npz", dest="save_npz", action="store_true", default=False)
    parser.add_argument("--no-save-npz", dest="save_npz", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-empty-safe-map", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="Print commands without executing; this is the default.")
    parser.add_argument("--execute", action="store_true", help="Execute workers and merge shards.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    gpus = _split_csv(args.gpus)
    if not gpus:
        parser.error("--gpus must not be empty")
    if args.num_shards is None:
        args.num_shards = len(gpus)
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if len(gpus) != args.num_shards:
        parser.error("--gpus count must equal --num-shards")
    if args.save_npz and args.num_shards > 1:
        parser.error("--save-npz is not supported with --num-shards > 1")
    if args.taylorseer_interval is not None and args.taylorseer_interval <= 0:
        parser.error("--taylorseer-interval must be positive")
    if args.taylorseer_max_order is not None and args.taylorseer_max_order < 0:
        parser.error("--taylorseer-max-order must be non-negative")
    if args.taylorseer_refresh_first_n_steps is not None and args.taylorseer_refresh_first_n_steps < 0:
        parser.error("--taylorseer-refresh-first-n-steps must be non-negative")
    if args.taylorseer_refresh_last_n_steps is not None and args.taylorseer_refresh_last_n_steps < 0:
        parser.error("--taylorseer-refresh-last-n-steps must be non-negative")
    if args.taylorseer_min_history is not None and args.taylorseer_min_history <= 0:
        parser.error("--taylorseer-min-history must be positive")
    if args.speca_max_order is not None and args.speca_max_order < 0:
        parser.error("--speca-max-order must be non-negative")
    if args.speca_first_full_steps is not None and args.speca_first_full_steps < 0:
        parser.error("--speca-first-full-steps must be non-negative")
    if args.speca_base_threshold is not None and args.speca_base_threshold <= 0.0:
        parser.error("--speca-base-threshold must be positive")
    if args.speca_decay_rate is not None and not 0.0 < args.speca_decay_rate <= 1.0:
        parser.error("--speca-decay-rate must satisfy 0 < value <= 1")
    if args.speca_min_threshold is not None and args.speca_min_threshold <= 0.0:
        parser.error("--speca-min-threshold must be positive")
    if args.speca_min_forecast_steps is not None and args.speca_min_forecast_steps <= 0:
        parser.error("--speca-min-forecast-steps must be positive")
    if args.speca_max_forecast_steps is not None and args.speca_max_forecast_steps <= 0:
        parser.error("--speca-max-forecast-steps must be positive")
    if args.speca_min_history is not None and args.speca_min_history <= 0:
        parser.error("--speca-min-history must be positive")
    if args.speca_eps <= 0.0:
        parser.error("--speca-eps must be positive")
    if args.speca_max_error_samples <= 0:
        parser.error("--speca-max-error-samples must be positive")
    if args.dicache_probe_depth is not None and args.dicache_probe_depth <= 0:
        parser.error("--dicache-probe-depth must be positive")
    if args.dicache_reuse_threshold is not None and args.dicache_reuse_threshold <= 0.0:
        parser.error("--dicache-reuse-threshold must be positive")
    if args.dicache_ret_ratio is not None and not 0.0 <= args.dicache_ret_ratio < 1.0:
        parser.error("--dicache-ret-ratio must satisfy 0 <= value < 1")
    if args.dicache_gamma_min is not None and args.dicache_gamma_min < 0.0:
        parser.error("--dicache-gamma-min must be non-negative")
    if (
        args.dicache_gamma_min is not None
        and args.dicache_gamma_max is not None
        and args.dicache_gamma_min > args.dicache_gamma_max
    ):
        parser.error("--dicache-gamma-min must not exceed --dicache-gamma-max")
    if args.dicache_eps is not None and args.dicache_eps <= 0.0:
        parser.error("--dicache-eps must be positive")
    if args.dicache_max_stat_samples is not None and args.dicache_max_stat_samples <= 0:
        parser.error("--dicache-max-stat-samples must be positive")
    if (
        args.speca_min_forecast_steps is not None
        and args.speca_max_forecast_steps is not None
        and args.speca_max_forecast_steps < args.speca_min_forecast_steps
    ):
        parser.error("--speca-max-forecast-steps must be >= --speca-min-forecast-steps")
    if (
        args.speca_base_threshold is not None
        and args.speca_min_threshold is not None
        and args.speca_min_threshold > args.speca_base_threshold
    ):
        parser.error("--speca-min-threshold must not exceed --speca-base-threshold")

    worker_commands = [_base_worker_command(args, idx, python_bin="python") for idx in range(args.num_shards)]
    merge_command = _merge_command(args, python_bin="python")
    if not args.execute:
        for gpu, command in zip(gpus, worker_commands):
            print(_cmd_text(command, gpu))
        print(shlex.join(merge_command))
        return 0

    run_dir = Path(args.output_root) / "jit" / args.run_id / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    start_time = datetime.now(timezone.utc).isoformat()
    processes = []
    worker_meta = []
    log_dir = run_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, gpu in enumerate(gpus):
            command = _base_worker_command(args, idx, python_bin=sys.executable)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            stdout_path = log_dir / f"worker_shard{idx}.stdout.log"
            stderr_path = log_dir / f"worker_shard{idx}.stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            processes.append(process)
            worker_meta.append(
                {
                    "command": command,
                    "command_text": _cmd_text(command, gpu),
                    "gpu_id": gpu,
                    "pid": process.pid,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                    "start_time_utc": datetime.now(timezone.utc).isoformat(),
                    "start_monotonic": time.perf_counter(),
                }
            )
        failed = False
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                failed = True
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(0.1)
        return_codes = [process.wait() for process in processes]
        failed = failed or any(code != 0 for code in return_codes)
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise
    finally:
        for item in worker_meta:
            item["stdout_handle"].close()
            item["stderr_handle"].close()
    end = time.perf_counter()
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_time_utc": start_time,
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "parallel_orchestration_wall_time_sec": end - start,
        "worker_return_codes": return_codes,
        "workers": [
            {
                **{key: value for key, value in item.items() if key not in {"stdout_handle", "stderr_handle"}},
                "return_code": return_codes[index],
                "end_time_utc": datetime.now(timezone.utc).isoformat(),
                "wall_time_sec": end - item["start_monotonic"],
            }
            for index, item in enumerate(worker_meta)
        ],
        "method": args.method,
        "num_shards": args.num_shards,
        "gpus": gpus,
        "comparable_for_algorithm_speedup": False,
    }
    launcher_meta = run_dir / "parallel_launcher_meta.json"
    for worker in meta["workers"]:
        worker.pop("start_monotonic", None)
    launcher_meta.write_text(
        json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    if failed:
        return 1
    merge = _merge_command(args, python_bin=sys.executable)
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


if __name__ == "__main__":
    raise SystemExit(main())
