from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def is_git_repo(path: str | Path) -> bool:
    repo_path = Path(path)
    if not repo_path.exists():
        return False
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--is-inside-work-tree"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def get_git_commit(path: str | Path, short: bool = False) -> str | None:
    repo_path = Path(path)
    if not is_git_repo(repo_path):
        return None
    cmd = ["git", "-C", str(repo_path), "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repo_status(path: str | Path) -> dict[str, Any]:
    repo_path = Path(path)
    status: dict[str, Any] = {
        "path": str(repo_path),
        "exists": repo_path.exists(),
        "is_git_repo": is_git_repo(repo_path),
        "commit": None,
        "dirty": None,
        "status_short": None,
    }
    if not status["is_git_repo"]:
        return status

    status["commit"] = get_git_commit(repo_path)
    result = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--short"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        status_short = result.stdout.strip()
        status["status_short"] = status_short
        status["dirty"] = bool(status_short)
    return status

