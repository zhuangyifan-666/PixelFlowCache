from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tracked_shells() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.sh"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def test_all_tracked_shell_scripts_are_lf() -> None:
    scripts = _tracked_shells()
    assert scripts
    crlf = [str(path.relative_to(ROOT)) for path in scripts if b"\r\n" in path.read_bytes()]
    assert not crlf, f"tracked shell scripts with CRLF: {crlf}"


def test_all_tracked_shell_scripts_pass_bash_n_when_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    failures = []
    for path in _tracked_shells():
        result = subprocess.run(
            [bash, "-n", str(path)],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode:
            failures.append(f"{path.relative_to(ROOT)}: {result.stderr}")
    assert not failures, "\n".join(failures)
