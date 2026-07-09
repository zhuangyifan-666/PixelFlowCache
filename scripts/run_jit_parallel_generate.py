#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _cmd_text(command: list[str], gpu: str) -> str:
    return "CUDA_VISIBLE_DEVICES={} {}".format(gpu, " ".join(command))


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
    if args.taylorseer_interval is not None:
        command.extend(["--taylorseer-interval", str(args.taylorseer_interval)])
    if args.taylorseer_max_order is not None:
        command.extend(["--taylorseer-max-order", str(args.taylorseer_max_order)])
    if args.taylorseer_refresh_first_n_steps is not None:
        command.extend(["--taylorseer-refresh-first-n-steps", str(args.taylorseer_refresh_first_n_steps)])
    if args.taylorseer_refresh_last_n_steps is not None:
        command.extend(["--taylorseer-refresh-last-n-steps", str(args.taylorseer_refresh_last_n_steps)])
    if args.taylorseer_debug_jsonl is not None:
        command.extend(["--taylorseer-debug-jsonl", str(args.taylorseer_debug_jsonl)])
    if args.taylorseer_min_history is not None:
        command.extend(["--taylorseer-min-history", str(args.taylorseer_min_history)])
    if args.taylorseer_clone_forecast:
        command.append("--taylorseer-clone-forecast")
    command.extend(["--sea-beta", str(args.sea_beta), "--sea-proxy-downsample", str(args.sea_proxy_downsample)])
    command.append("--save-png" if args.save_png else "--no-save-png")
    command.append("--save-npz" if args.save_npz else "--no-save-npz")
    if args.resume:
        command.append("--resume")
    return command


def _merge_command(args: argparse.Namespace, python_bin: str = "python") -> list[str]:
    run_dir = Path(args.output_root) / "jit" / args.run_id / args.method
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
    parser.add_argument("--dynamic-cache-threshold", type=float)
    parser.add_argument("--taylorseer-interval", type=int)
    parser.add_argument("--taylorseer-max-order", type=int)
    parser.add_argument("--taylorseer-debug-jsonl", type=Path)
    parser.add_argument("--taylorseer-refresh-first-n-steps", type=int)
    parser.add_argument("--taylorseer-refresh-last-n-steps", type=int)
    parser.add_argument("--taylorseer-min-history", type=int)
    parser.add_argument("--taylorseer-clone-forecast", action="store_true")
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

    worker_commands = [_base_worker_command(args, idx, python_bin="python") for idx in range(args.num_shards)]
    merge_command = _merge_command(args, python_bin="python")
    if not args.execute:
        for gpu, command in zip(gpus, worker_commands):
            print(_cmd_text(command, gpu))
        print(" ".join(merge_command))
        return 0

    run_dir = Path(args.output_root) / "jit" / args.run_id / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    start_time = datetime.now(timezone.utc).isoformat()
    processes = []
    executed_commands = []
    for idx, gpu in enumerate(gpus):
        command = _base_worker_command(args, idx, python_bin=sys.executable)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        executed_commands.append(_cmd_text(command, gpu))
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env))
    return_codes = [process.wait() for process in processes]
    end = time.perf_counter()
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_time_utc": start_time,
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_sec": end - start,
        "worker_return_codes": return_codes,
        "worker_commands": executed_commands,
        "method": args.method,
        "num_shards": args.num_shards,
        "gpus": gpus,
    }
    launcher_meta = run_dir / "parallel_launcher_meta.json"
    launcher_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(code != 0 for code in return_codes):
        return 1
    merge = _merge_command(args, python_bin=sys.executable)
    merge_result = subprocess.run(merge, cwd=ROOT, check=False)
    return int(merge_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
