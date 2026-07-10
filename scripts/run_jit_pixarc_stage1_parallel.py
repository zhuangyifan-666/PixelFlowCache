#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for entry in (ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from pfc.risk.io import strict_json_dumps, write_json_atomic  # noqa: E402
from run_jit_pixarc_stage1_instrument import add_stage1_arguments, resolve_stage1_config  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or launch independent JiT PixARC Stage-1 sample shards."
    )
    add_stage1_arguments(parser, parallel_defaults=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--env", default="jit")
    parser.add_argument("--python-executable")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--print-only", action="store_true")
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _interpreter_prefix(args: argparse.Namespace) -> list[str]:
    if args.python_executable:
        return [str(args.python_executable)]
    return ["conda", "run", "-n", str(args.env), "python"]


def _common_worker_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--jit-dir", str(args.jit_dir),
        "--jit-ckpt-dir", str(args.jit_ckpt_dir),
        "--run-id", args.run_id,
        "--output-root", str(args.output_root),
        "--num-images", str(args.num_images),
        "--batch-size", str(args.batch_size),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--cfg", str(args.cfg),
        "--cfg-interval-min", str(args.cfg_interval_min),
        "--cfg-interval-max", str(args.cfg_interval_max),
        "--img-size", str(args.img_size),
        "--noise-scale", str(args.noise_scale),
        "--plans", args.plans,
        "--actions", args.actions,
        "--risk-atol", str(args.risk_atol),
        "--risk-rtol", str(args.risk_rtol),
        "--frequency-low-ratio", str(args.frequency_low_ratio),
        "--frequency-high-ratio", str(args.frequency_high_ratio),
        "--equivalence-steps", args.equivalence_steps,
        "--equivalence-atol", str(args.equivalence_atol),
        "--equivalence-rtol", str(args.equivalence_rtol),
        "--num-shards", str(args.num_shards),
        "--shard-mode", args.shard_mode,
    ]
    arguments.append(
        "--measure-action-latency" if args.measure_action_latency else "--no-measure-action-latency"
    )
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.strict_correctness, "--strict-correctness"),
        (args.correctness_only, "--correctness-only"),
        (args.hash_checkpoint, "--hash-checkpoint"),
        (args.save_final_png, "--save-final-png"),
    ):
        if enabled:
            arguments.append(flag)
    return arguments


def build_parallel_plan(args: argparse.Namespace) -> dict[str, Any]:
    resolved = resolve_stage1_config(args)
    gpus = _csv(args.gpus)
    if len(gpus) != args.num_shards:
        raise ValueError(
            f"GPU count ({len(gpus)}) must equal num-shards ({args.num_shards})"
        )
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU identifiers must be unique")
    prefix = _interpreter_prefix(args)
    common = _common_worker_arguments(args)
    workers = []
    for shard_index, gpu in enumerate(gpus):
        argv = [
            *prefix,
            "scripts/run_jit_pixarc_stage1_instrument.py",
            *common,
            "--shard-index",
            str(shard_index),
        ]
        workers.append(
            {
                "shard_index": shard_index,
                "gpu": gpu,
                "environment": {"CUDA_VISIBLE_DEVICES": gpu},
                "argv": argv,
                "command": shlex.join(argv),
            }
        )
    merge_argv = [
        *prefix,
        "scripts/merge_jit_pixarc_stage1.py",
        "--run-dir",
        str(resolved["run_dir"]),
        "--expected-images",
        str(args.num_images),
        "--strict",
    ]
    return {
        "run_id": args.run_id,
        "run_dir": str(resolved["run_dir"]),
        "num_shards": args.num_shards,
        "gpus": gpus,
        "conda_environment": None if args.python_executable else args.env,
        "python_executable_override": args.python_executable,
        "workers": workers,
        "merge_enabled": bool(args.merge),
        "merge_argv": merge_argv,
        "merge_command": shlex.join(merge_argv),
        "parallel_wall_time_is_algorithm_speedup": False,
    }


def execute_parallel_plan(plan: dict[str, Any]) -> int:
    if _is_windows():
        raise RuntimeError("Stage-1 parallel execution is disabled on Windows; use --print-only.")
    run_dir = Path(plan["run_dir"])
    log_dir = run_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "parallel_launcher.json"
    metadata = {**plan, "status": "running", "worker_results": []}
    write_json_atomic(metadata_path, metadata)
    processes: list[tuple[dict[str, Any], subprocess.Popen[Any], Any, Any]] = []
    started = time.perf_counter()
    try:
        for worker in plan["workers"]:
            stdout_handle = (log_dir / f"shard_{worker['shard_index']}.stdout.log").open(
                "w", encoding="utf-8", newline="\n"
            )
            stderr_handle = (log_dir / f"shard_{worker['shard_index']}.stderr.log").open(
                "w", encoding="utf-8", newline="\n"
            )
            environment = os.environ.copy()
            environment.update(worker["environment"])
            try:
                process = subprocess.Popen(
                    worker["argv"],
                    cwd=ROOT,
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                )
            except Exception:
                stdout_handle.close()
                stderr_handle.close()
                for _worker, running, _stdout, _stderr in processes:
                    if running.poll() is None:
                        running.terminate()
                for _worker, running, _stdout, _stderr in processes:
                    try:
                        running.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        running.kill()
                        running.wait()
                metadata["status"] = "worker_launch_failed"
                write_json_atomic(metadata_path, metadata)
                raise
            processes.append((worker, process, stdout_handle, stderr_handle))

        failed: tuple[dict[str, Any], int] | None = None
        while True:
            running = False
            for worker, process, _stdout, _stderr in processes:
                returncode = process.poll()
                if returncode is None:
                    running = True
                elif returncode != 0 and failed is None:
                    failed = (worker, int(returncode))
            if failed is not None or not running:
                break
            time.sleep(0.2)
        if failed is not None:
            for _worker, process, _stdout, _stderr in processes:
                if process.poll() is None:
                    process.terminate()
            for _worker, process, _stdout, _stderr in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        else:
            for _worker, process, _stdout, _stderr in processes:
                process.wait()
    finally:
        for _worker, _process, stdout_handle, stderr_handle in processes:
            stdout_handle.close()
            stderr_handle.close()

    results = [
        {
            "shard_index": worker["shard_index"],
            "gpu": worker["gpu"],
            "returncode": process.returncode,
        }
        for worker, process, _stdout, _stderr in processes
    ]
    failed_results = [result for result in results if result["returncode"] != 0]
    metadata.update(
        {
            "status": "worker_failed" if failed_results else "workers_succeeded",
            "worker_results": results,
            "parallel_wall_time_sec": time.perf_counter() - started,
        }
    )
    write_json_atomic(metadata_path, metadata)
    if failed_results:
        return 1
    if plan["merge_enabled"]:
        merge = subprocess.run(plan["merge_argv"], cwd=ROOT, check=False, shell=False)
        metadata["merge_returncode"] = int(merge.returncode)
        metadata["status"] = "complete" if merge.returncode == 0 else "merge_failed"
        write_json_atomic(metadata_path, metadata)
        return int(merge.returncode)
    metadata["status"] = "complete_without_merge"
    write_json_atomic(metadata_path, metadata)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    plan = build_parallel_plan(args)
    if not args.execute:
        print(strict_json_dumps({"print_only": True, **plan}, indent=2))
        return 0
    return execute_parallel_plan(plan)


def _is_windows() -> bool:
    return os.name == "nt"


if __name__ == "__main__":
    raise SystemExit(main())
