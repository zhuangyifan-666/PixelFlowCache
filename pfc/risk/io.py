from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable

from pfc.risk.schema import PIXARC_STAGE1_SCHEMA_VERSION, ensure_strict_json_value


def strict_json_dumps(payload: Any, *, indent: int | None = None) -> str:
    ensure_strict_json_value(payload)
    return json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False)


def config_signature(payload: dict[str, Any]) -> str:
    encoded = strict_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path | str, payload: Any) -> None:
    write_text_atomic(path, strict_json_dumps(payload, indent=2) + "\n")


def write_text_atomic(path: Path | str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in records:
                handle.write(strict_json_dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
            ensure_strict_json_value(row, f"{source}:{line_number}")
            rows.append(row)
    return rows


def sample_directory(run_dir: Path | str, global_index: int) -> Path:
    return Path(run_dir) / "samples" / f"sample_{int(global_index):06d}"


def reconcile_sample_output(
    run_dir: Path | str,
    global_index: int,
    expected_config_signature: str,
    *,
    resume: bool,
) -> str:
    target = sample_directory(run_dir, global_index)
    done_path = target / "DONE.json"
    if done_path.is_file():
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"sample DONE is unreadable: {done_path}: {exc}") from exc
        if int(done.get("schema_version", -1)) != PIXARC_STAGE1_SCHEMA_VERSION:
            raise ValueError(f"sample DONE schema mismatch: {done_path}")
        if done.get("config_signature") != expected_config_signature:
            raise ValueError(f"sample DONE belongs to a different run configuration: {done_path}")
        if int(done.get("global_index", -1)) != int(global_index):
            raise ValueError(f"sample DONE global index mismatch: {done_path}")
        if resume:
            return "skip"
        raise FileExistsError(f"completed Stage-1 sample already exists; use --resume: {target}")
    if target.exists():
        shutil.rmtree(target)
    samples_dir = target.parent
    if samples_dir.is_dir():
        for temporary in samples_dir.glob(f".{target.name}.*.tmp"):
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()
    return "run"


class AtomicSampleWriter:
    def __init__(
        self,
        run_dir: Path | str,
        global_index: int,
        expected_config_signature: str,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.global_index = int(global_index)
        self.expected_config_signature = str(expected_config_signature)
        self.target = sample_directory(self.run_dir, self.global_index)
        samples_dir = self.target.parent
        samples_dir.mkdir(parents=True, exist_ok=True)
        self.temporary = samples_dir / f".{self.target.name}.{uuid.uuid4().hex}.tmp"
        self.temporary.mkdir()
        self._committed = False

    def commit(
        self,
        *,
        risk_records: list[dict[str, Any]],
        correctness_records: list[dict[str, Any]],
        sample_summary: dict[str, Any],
    ) -> Path:
        if self._committed:
            raise RuntimeError("sample output is already committed")
        write_jsonl(self.temporary / "risk_records.jsonl", risk_records)
        write_jsonl(self.temporary / "correctness_records.jsonl", correctness_records)
        summary = {
            **sample_summary,
            "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
            "global_index": self.global_index,
            "config_signature": self.expected_config_signature,
        }
        write_json_atomic(self.temporary / "sample_summary.json", summary)
        if self.target.exists():
            raise FileExistsError(f"refusing to overwrite Stage-1 sample: {self.target}")
        os.replace(self.temporary, self.target)
        write_json_atomic(
            self.target / "DONE.json",
            {
                "schema_version": PIXARC_STAGE1_SCHEMA_VERSION,
                "record_type": "pixarc_stage1_sample_done",
                "global_index": self.global_index,
                "config_signature": self.expected_config_signature,
                "risk_record_count": len(risk_records),
                "correctness_record_count": len(correctness_records),
            },
        )
        self._committed = True
        return self.target

    def abort(self) -> None:
        if not self._committed and self.temporary.exists():
            shutil.rmtree(self.temporary)

    def __enter__(self) -> "AtomicSampleWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None:
            self.abort()


def completed_sample_indices(run_dir: Path | str) -> list[int]:
    samples_dir = Path(run_dir) / "samples"
    if not samples_dir.is_dir():
        return []
    indices: list[int] = []
    for path in samples_dir.glob("sample_*"):
        suffix = path.name.removeprefix("sample_")
        if suffix.isdigit() and (path / "DONE.json").is_file():
            indices.append(int(suffix))
    return sorted(indices)
