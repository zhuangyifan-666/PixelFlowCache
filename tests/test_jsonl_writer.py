from __future__ import annotations

import json
from pathlib import Path

from pfc.profiling.jsonl import JsonlWriter


def test_jsonl_writer_creates_parent_and_writes(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"a": 1, "path": tmp_path})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"a": 1, "path": str(tmp_path)}]

