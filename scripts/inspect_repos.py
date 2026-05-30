#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pfc.utils.repo import repo_status  # noqa: E402


IMAGENET_CANDIDATES = [
    Path(os.environ.get("PFC_IMAGENET_PATH", "")) if os.environ.get("PFC_IMAGENET_PATH") else None,
    Path("/mnt/iset/nfs-main/public/datasets/ILSVRC"),
    Path("/mnt/iset/nfs-main/public/datasets/ILSVRC/Data/CLS-LOC"),
    Path("/mnt/iset/nfs-main/public/datasets/ILSVRC2012"),
    Path("/mnt/iset/nfs-main/public/datasets/ImageNet"),
]


def detect_imagenet_root() -> dict[str, Any]:
    checked: list[str] = []
    for candidate in IMAGENET_CANDIDATES:
        if candidate is None:
            continue
        checked.append(str(candidate))
        if (candidate / "train").is_dir():
            return {
                "detected": str(candidate),
                "has_train": True,
                "has_val": (candidate / "val").is_dir(),
                "checked": checked,
            }
    return {"detected": None, "has_train": False, "has_val": False, "checked": checked}


def detect_jit_checkpoint() -> dict[str, Any]:
    env_dir = os.environ.get("PFC_JIT_CKPT_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(ROOT / "ckpts/JiT/JiT-B-16-256")

    for candidate in candidates:
        ckpt = candidate / "checkpoint-last.pth"
        if ckpt.is_file() or ckpt.is_symlink():
            return {"detected_dir": str(candidate), "checkpoint": str(ckpt), "created_symlink": False}

    pths = sorted((ROOT / "ckpts").glob("**/*.pth"))
    if pths:
        expected_dir = ROOT / "ckpts/JiT/JiT-B-16-256"
        expected_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = expected_dir / "checkpoint-last.pth"
        if not symlink_path.exists():
            symlink_path.symlink_to(pths[0])
            return {"detected_dir": str(expected_dir), "checkpoint": str(symlink_path), "created_symlink": True}
        return {"detected_dir": str(expected_dir), "checkpoint": str(symlink_path), "created_symlink": False}

    return {
        "detected_dir": None,
        "checkpoint": None,
        "expected": str(ROOT / "ckpts/JiT/JiT-B-16-256/checkpoint-last.pth"),
    }


def detect_deco_checkpoint() -> dict[str, Any]:
    env_ckpt = os.environ.get("PFC_DECO_CKPT")
    candidates = []
    if env_ckpt:
        candidates.append(Path(env_ckpt))
    candidates.append(ROOT / "ckpts/DeCo/imagenet256_epoch800.ckpt")

    for candidate in candidates:
        if candidate.is_file():
            return {"detected": str(candidate)}

    preferred = sorted((ROOT / "ckpts").glob("**/*imagenet*256*800*.ckpt"))
    if preferred:
        return {"detected": str(preferred[0]), "expected": str(ROOT / "ckpts/DeCo/imagenet256_epoch800.ckpt")}

    ckpts = sorted((ROOT / "ckpts").glob("**/*.ckpt"))
    if ckpts:
        return {"detected": str(ckpts[0]), "expected": str(ROOT / "ckpts/DeCo/imagenet256_epoch800.ckpt")}

    return {"detected": None, "expected": str(ROOT / "ckpts/DeCo/imagenet256_epoch800.ckpt")}


def get_submodule_status() -> str:
    result = subprocess.run(
        ["git", "submodule", "status", "third_party/JiT", "third_party/DeCo"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return f"ERROR: {result.stderr.strip()}"


def append_repro_log(status: dict[str, Any]) -> None:
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    log_path = docs_dir / "repro_log.md"
    ts = status["timestamp_utc"]
    text = [
        f"## Stage 0 Inspect - {ts}",
        "",
        f"- root commit: {status['repos']['root'].get('commit')}",
        f"- JiT commit: {status['repos']['jit'].get('commit')}",
        f"- DeCo commit: {status['repos']['deco'].get('commit')}",
        f"- detected ImageNet root: {status['imagenet'].get('detected')}",
        f"- detected JiT checkpoint path: {status['checkpoints']['jit'].get('checkpoint')}",
        f"- detected DeCo checkpoint path: {status['checkpoints']['deco'].get('detected')}",
        "- selected GPUs for each run: not selected by inspect",
        "- smoke test passed: not run by inspect",
        "- JiT official debug baseline ran: not run by inspect",
        "- DeCo official debug baseline ran: not run by inspect",
        "- known blockers: see missing fields above",
        "",
    ]
    if not log_path.exists():
        log_path.write_text("# PixelFlowCache Repro Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(text))


def main() -> int:
    logs_dir = ROOT / "logs/stage0"
    logs_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "repos": {
            "root": repo_status(ROOT),
            "jit": repo_status(ROOT / "third_party/JiT"),
            "deco": repo_status(ROOT / "third_party/DeCo"),
        },
        "submodule_status": get_submodule_status(),
        "imagenet": detect_imagenet_root(),
        "checkpoints": {
            "jit": detect_jit_checkpoint(),
            "deco": detect_deco_checkpoint(),
        },
    }

    out_path = logs_dir / "repo_status.json"
    out_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    append_repro_log(status)

    print(json.dumps(status, indent=2, sort_keys=True))
    if not status["repos"]["jit"]["exists"] or not status["repos"]["deco"]["exists"]:
        print("Missing third-party repo. Run: bash scripts/setup_third_party.sh")
    if not status["imagenet"]["detected"]:
        print("ImageNet ImageFolder root not detected. Candidates checked:")
        for candidate in status["imagenet"]["checked"]:
            print(f"  {candidate}")
    if not status["checkpoints"]["jit"].get("checkpoint"):
        print("Missing JiT checkpoint. Expected ckpts/JiT/JiT-B-16-256/checkpoint-last.pth")
    if not status["checkpoints"]["deco"].get("detected"):
        print("Missing DeCo checkpoint. Expected ckpts/DeCo/imagenet256_epoch800.ckpt")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

