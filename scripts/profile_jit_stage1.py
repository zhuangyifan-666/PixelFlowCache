#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pfc.profiling.feature_recorder import FeatureRecorder  # noqa: E402
from pfc.profiling.frequency import fft_frequency_bands, frequency_delta_bands  # noqa: E402
from pfc.profiling.jsonl import JsonlWriter  # noqa: E402
from pfc.profiling.run_meta import collect_env_info, collect_git_status, write_run_meta  # noqa: E402
from pfc.profiling.velocity_recorder import VelocityRecorder  # noqa: E402
from pfc.utils.seeding import set_seed  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _detect_jit_ckpt_dir() -> Path:
    env_path = os.environ.get("PFC_JIT_CKPT_DIR")
    if env_path:
        path = Path(env_path)
        if (path / "checkpoint-last.pth").is_file():
            return path
    expected = ROOT / "ckpts/JiT/JiT-B-16-256"
    if (expected / "checkpoint-last.pth").is_file():
        return expected
    matches = sorted((ROOT / "ckpts").glob("**/checkpoint-last.pth"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError("JiT checkpoint not found. Expected ckpts/JiT/JiT-B-16-256/checkpoint-last.pth")


def _detect_imagenet_root() -> Path:
    candidates = [
        os.environ.get("PFC_IMAGENET_PATH"),
        "/mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC",
        "/mnt/iset/nfs-main/public/datasets/ILSVRC",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "train").is_dir():
            return Path(candidate)
    raise FileNotFoundError("Could not detect ImageNet root containing train/")


def _make_run_id(seed: int, steps: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_seed{seed}_steps{steps}"


def _save_previews(x: torch.Tensor, preview_dir: Path, max_images: int = 4) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    images = ((x.detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    for idx, image in enumerate(images[:max_images]):
        arr = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        Image.fromarray(arr).save(preview_dir / f"{idx:03d}.png")


def _load_jit_model(jit_dir: Path, ckpt_dir: Path, device: torch.device, args: Namespace):
    sys.path.insert(0, str(jit_dir))
    from denoiser import Denoiser  # type: ignore

    model = Denoiser(args)
    checkpoint_path = ckpt_dir / "checkpoint-last.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if "model_ema1" in checkpoint:
        ema_state = model.state_dict()
        for name, _param in model.named_parameters():
            if name in checkpoint["model_ema1"]:
                ema_state[name] = checkpoint["model_ema1"][name]
        model.load_state_dict(ema_state)
    model.to(device)
    model.eval()
    return model


def main() -> int:
    jit_dir = Path(os.environ.get("PFC_JIT_DIR", ROOT / "third_party/JiT")).resolve()
    ckpt_dir = _detect_jit_ckpt_dir()
    imagenet_root = _detect_imagenet_root()
    seed = _env_int("PFC_PROFILE_SEED", 0)
    num_samples = _env_int("PFC_PROFILE_NUM_SAMPLES", 4)
    batch_size = _env_int("PFC_PROFILE_BATCH_SIZE", num_samples)
    steps = _env_int("PFC_PROFILE_STEPS", 10)
    img_size = _env_int("PFC_PROFILE_IMG_SIZE", 256)
    cfg = _env_float("PFC_PROFILE_CFG", 3.0)
    interval_min = _env_float("PFC_PROFILE_CFG_INTERVAL_MIN", 0.1)
    interval_max = _env_float("PFC_PROFILE_CFG_INTERVAL_MAX", 1.0)
    solver = os.environ.get("PFC_PROFILE_SOLVER", "euler")
    if solver != "euler":
        raise NotImplementedError("JiT Stage 1 profiling currently supports only euler")

    run_id = os.environ.get("PFC_STAGE1_RUN_ID", _make_run_id(seed, steps))
    run_dir = Path(os.environ.get("PFC_STAGE1_OUT_DIR", ROOT / "logs/stage1/jit" / run_id)).resolve()
    preview_dir = Path(os.environ.get("PFC_STAGE1_PREVIEW_DIR", ROOT / "outputs/stage1/previews/jit" / run_id)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Namespace(
        model=os.environ.get("PFC_PROFILE_JIT_MODEL", "JiT-B/16"),
        img_size=img_size,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=1.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="euler",
        num_sampling_steps=steps,
        cfg=cfg,
        interval_min=interval_min,
        interval_max=interval_max,
    )

    meta = {
        **collect_git_status(ROOT, jit_dir, ROOT / "third_party/DeCo"),
        "env": collect_env_info(),
        "script": "scripts/profile_jit_stage1.py",
        "model_name": "JiT",
        "checkpoint_dir": str(ckpt_dir),
        "imagenet_root": str(imagenet_root),
        "seed": seed,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "steps": steps,
        "cfg_scale": cfg,
        "cfg_interval": [interval_min, interval_max],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "preview_dir": str(preview_dir),
    }
    write_run_meta(run_dir / "meta.json", meta)

    model = _load_jit_model(jit_dir, ckpt_dir, device, args)
    feature_writer = JsonlWriter(run_dir / "feature_stats.jsonl")
    velocity_recorder = VelocityRecorder(JsonlWriter(run_dir / "velocity_stats.jsonl"))
    frequency_writer = JsonlWriter(run_dir / "frequency_stats.jsonl")
    step_writer = JsonlWriter(run_dir / "step_stats.jsonl")

    recorder = FeatureRecorder(
        module_filter=lambda name, _module: name.startswith("blocks.") and name.count(".") == 1,
        writer=feature_writer,
        model_name="JiT",
        previous_on_cpu=True,
        previous_dtype="float16",
    )
    recorder.attach(model.net)

    start_time = time.time()
    labels = torch.arange(num_samples, device=device, dtype=torch.long) % 1000
    z = args.noise_scale * torch.randn(num_samples, 3, img_size, img_size, device=device)
    timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=z.dtype)
    prev_v_cfg: torch.Tensor | None = None

    with torch.no_grad():
        for step_idx in range(steps):
            t_scalar = timesteps[step_idx]
            t_next_scalar = timesteps[step_idx + 1]
            dt = t_next_scalar - t_scalar
            t_value = float(t_scalar.detach().float().cpu().item())
            t_next_value = float(t_next_scalar.detach().float().cpu().item())
            dt_value = float(dt.detach().float().cpu().item())
            t = t_scalar.expand(num_samples, 1, 1, 1)
            null_labels = torch.full_like(labels, 1000)
            cfg_enabled = bool((t_value < interval_max) and ((interval_min == 0.0) or (t_value > interval_min)))
            cfg_scale_interval = cfg if cfg_enabled else 1.0

            recorder.set_context(step_idx, t_value, solver_stage="euler", cfg_branch="cond")
            x_cond = model.net(z, t.flatten(), labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(args.t_eps)
            velocity_recorder.log_xpred_conversion(
                model_name="JiT",
                step_idx=step_idx,
                t=t_value,
                t_next=t_next_value,
                dt=dt_value,
                branch="cond",
                x0_pred=x_cond,
                v=v_cond,
                x_current=z,
                cfg_scale=cfg,
                cfg_enabled=cfg_enabled,
                eps=args.t_eps,
            )

            recorder.set_context(step_idx, t_value, solver_stage="euler", cfg_branch="uncond")
            x_uncond = model.net(z, t.flatten(), null_labels)
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(args.t_eps)
            velocity_recorder.log_xpred_conversion(
                model_name="JiT",
                step_idx=step_idx,
                t=t_value,
                t_next=t_next_value,
                dt=dt_value,
                branch="uncond",
                x0_pred=x_uncond,
                v=v_uncond,
                x_current=z,
                cfg_scale=cfg,
                cfg_enabled=cfg_enabled,
                eps=args.t_eps,
            )

            v_cfg = v_uncond + cfg_scale_interval * (v_cond - v_uncond)
            velocity_recorder.log_velocity(
                model_name="JiT",
                step_idx=step_idx,
                t=t_value,
                t_next=t_next_value,
                dt=dt_value,
                branch="cfg",
                v=v_cfg,
                cfg_scale=cfg,
                cfg_enabled=cfg_enabled,
                extra={"amplification": 1.0 / max(1.0 - t_value, args.t_eps)},
            )

            frequency_record: dict[str, Any] = {
                "record_type": "frequency",
                "model_name": "JiT",
                "step_idx": step_idx,
                "t": t_value,
                "t_next": t_next_value,
                "dt": dt_value,
                "branch": "cfg",
                "cfg_scale": cfg,
                "cfg_enabled": cfg_enabled,
                "frequency": fft_frequency_bands(v_cfg),
            }
            if prev_v_cfg is not None:
                frequency_record["frequency_delta"] = frequency_delta_bands(v_cfg, prev_v_cfg)
            frequency_writer.write(frequency_record)
            prev_v_cfg = v_cfg.detach().to(dtype=torch.float16, device="cpu")

            step_writer.write(
                {
                    "record_type": "step",
                    "model_name": "JiT",
                    "step_idx": step_idx,
                    "t": t_value,
                    "t_next": t_next_value,
                    "dt": dt_value,
                    "cfg_enabled": cfg_enabled,
                    "cfg_scale": cfg,
                    "amplification": 1.0 / max(1.0 - t_value, args.t_eps),
                }
            )
            z = z + dt * v_cfg

    recorder.remove()
    feature_writer.close()
    velocity_recorder.close()
    frequency_writer.close()
    step_writer.close()
    _save_previews(z, preview_dir)

    summary = {
        "model_name": "JiT",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "preview_dir": str(preview_dir),
        "num_steps": steps,
        "num_samples": num_samples,
        "feature_records": recorder.record_count,
        "runtime_seconds": time.time() - start_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JiT Stage 1 profile run dir: {run_dir}")
    print(f"JiT Stage 1 preview dir: {preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

