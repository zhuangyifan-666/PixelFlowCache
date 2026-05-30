#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pfc.profiling.run_meta import collect_env_info, collect_git_status, write_run_meta  # noqa: E402


def _make_run_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _detect_deco_ckpt() -> Path:
    env_path = os.environ.get("PFC_DECO_CKPT")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    expected = ROOT / "ckpts/DeCo/imagenet256_epoch800/imagenet256_epoch800.ckpt"
    if expected.is_file():
        return expected
    matches = sorted((ROOT / "ckpts").glob("**/*.ckpt"))
    if matches:
        return matches[0]
    raise FileNotFoundError("DeCo checkpoint not found under ckpts/DeCo")


def main() -> int:
    deco_dir = Path(os.environ.get("PFC_DECO_DIR", ROOT / "third_party/DeCo")).resolve()
    config = Path(os.environ.get("PFC_DECO_PROFILE_CONFIG", ROOT / "configs/deco_stage1_profile.yaml")).resolve()
    ckpt = _detect_deco_ckpt()
    seed = int(os.environ.get("PFC_PROFILE_SEED", 0))
    steps = int(os.environ.get("PFC_PROFILE_STEPS", 10))
    num_samples = int(os.environ.get("PFC_PROFILE_NUM_SAMPLES", 4))
    batch_size = int(os.environ.get("PFC_PROFILE_BATCH_SIZE", 4))
    run_id = os.environ.get("PFC_STAGE1_RUN_ID", _make_run_id(seed, steps))
    run_dir = Path(os.environ.get("PFC_DECO_PROFILE_LOG_DIR", ROOT / "logs/stage1/deco" / run_id)).resolve()
    preview_root = Path(os.environ.get("PFC_STAGE1_PREVIEW_DIR", ROOT / "outputs/stage1/previews/deco" / run_id)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    os.environ["PFC_DECO_PROFILE_LOG_DIR"] = str(run_dir)
    os.environ["PYTHONPATH"] = f"{ROOT}:{deco_dir}:{os.environ.get('PYTHONPATH', '')}"
    sys.path.insert(0, str(deco_dir))

    meta = {
        **collect_git_status(ROOT, ROOT / "third_party/JiT", deco_dir),
        "env": collect_env_info(),
        "script": "scripts/profile_deco_stage1.py",
        "model_name": "DeCo",
        "checkpoint": str(ckpt),
        "config": str(config),
        "seed": seed,
        "steps": steps,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "preview_root": str(preview_root),
    }
    write_run_meta(run_dir / "meta.json", meta)

    sys.argv = [
        str(deco_dir / "main.py"),
        "predict",
        "-c",
        str(config),
        "--ckpt_path",
        str(ckpt),
        "--trainer.default_root_dir",
        str(preview_root),
        "--model.diffusion_sampler.init_args.num_steps",
        str(steps),
        "--data.pred_batch_size",
        str(batch_size),
        "--data.pred_dataset.init_args.max_num_instances",
        str(num_samples),
        "--data.eval_dataset.init_args.max_num_instances",
        str(num_samples),
    ]
    print("Running DeCo profile command:")
    print(" ".join(sys.argv))
    runpy.run_path(str(deco_dir / "main.py"), run_name="__main__")

    summary = {
        "model_name": "DeCo",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "preview_root": str(preview_root),
        "status": "completed",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DeCo Stage 1 profile run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

