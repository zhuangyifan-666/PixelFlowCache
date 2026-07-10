from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IMPORTANT_PACKAGES = (
    "numpy",
    "scipy",
    "Pillow",
    "PyYAML",
    "timm",
    "torch-fidelity",
    "clean-fid",
    "lpips",
    "scikit-image",
    "pytest",
)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_provenance(repo_root: Path | str, include_diff_stat: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    result: dict[str, Any] = {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
    }
    if include_diff_stat:
        result["git_diff_stat"] = _git(root, "diff", "--stat")
    return result


def collect_submodule_provenance(repo_root: Path | str) -> list[dict[str, Any]]:
    output = _git(Path(repo_root).resolve(), "submodule", "status", "--recursive")
    rows: list[dict[str, Any]] = []
    for line in output.splitlines() if output else []:
        fields = line.strip().split()
        if len(fields) >= 2:
            commit = fields[0].lstrip("-+")
            rows.append(
                {
                    "path": fields[1],
                    "commit": commit,
                    "initialized": not line.startswith("-"),
                    "dirty_or_mismatched": line.startswith("+"),
                }
            )
    return rows


def collect_runtime_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": {name: _package_version(name) for name in IMPORTANT_PACKAGES},
    }
    try:
        import torch

        result.update(
            {
                "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            }
        )
    except Exception:
        result.update({"torch_version": None, "cuda_runtime_version": None, "cudnn_version": None})
    try:
        import torchvision

        result["torchvision_version"] = torchvision.__version__
    except Exception:
        result["torchvision_version"] = None
    return result


def collect_gpu_provenance() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        devices = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
        return {
            "cuda_available": available,
            "gpu_count": count,
            "gpu_names": [device["name"] for device in devices],
            "gpus": devices,
            "driver_version": None,
        }
    except Exception as exc:
        return {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "gpus": [],
            "driver_version": None,
            "gpu_probe_error": str(exc),
        }


def collect_command_provenance(
    argv: Iterable[str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    return {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd() if cwd is None else Path(cwd).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def collect_file_provenance(
    path: Path | str | None,
    *,
    hash_file: bool = False,
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size": None, "sha256": None}
    source = Path(path).resolve()
    exists = source.is_file()
    return {
        "path": str(source),
        "exists": exists,
        "size": source.stat().st_size if exists else None,
        "sha256": sha256_file(source) if exists and hash_file else None,
    }


def collect_generation_provenance(
    repo_root: Path | str,
    *,
    checkpoint_path: Path | str | None = None,
    hash_checkpoint: bool = False,
    argv: Iterable[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        **collect_git_provenance(repo_root),
        **collect_runtime_provenance(),
        **collect_gpu_provenance(),
        **collect_command_provenance(argv=argv, cwd=repo_root),
        "checkpoint": collect_file_provenance(
            checkpoint_path,
            hash_file=hash_checkpoint,
        ),
        "third_party_submodules": collect_submodule_provenance(repo_root),
    }


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json_strict(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
