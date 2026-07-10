from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pfc.eval.provenance import (
    collect_command_provenance,
    collect_file_provenance,
    collect_gpu_provenance,
    collect_runtime_provenance,
    sha256_file,
    write_json_strict,
)


def test_sha256_and_optional_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"pixel-flow-cache")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected
    assert collect_file_provenance(path, hash_file=False)["sha256"] is None
    assert collect_file_provenance(path, hash_file=True)["sha256"] == expected


def test_missing_file_and_cpu_runtime_are_safe(tmp_path: Path) -> None:
    missing = collect_file_provenance(tmp_path / "missing")
    assert missing["exists"] is False
    assert "python_version" in collect_runtime_provenance()
    assert "gpu_count" in collect_gpu_provenance()
    assert collect_command_provenance(argv=["python", "x.py"])["argv"] == ["python", "x.py"]


def test_strict_json_writer_rejects_nan(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    try:
        write_json_strict(out, {"bad": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("strict provenance JSON must reject NaN")
    write_json_strict(out, {"ok": None})
    assert json.loads(out.read_text(encoding="utf-8")) == {"ok": None}
