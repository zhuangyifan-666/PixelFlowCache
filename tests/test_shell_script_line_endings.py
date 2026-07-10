from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = (
    "scripts/print_stage4a_seacache_theta006_commands.sh",
    "scripts/print_stage4a_seacache_baseline_commands.sh",
    "scripts/print_jit_safe_1000_commands.sh",
)


def test_critical_shell_scripts_use_lf_line_endings() -> None:
    for relative_path in SHELL_SCRIPTS:
        payload = (ROOT / relative_path).read_bytes()
        assert b"\r\n" not in payload, f"CRLF found in {relative_path}"
