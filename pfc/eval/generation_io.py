from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ResumeReconciliation:
    complete_indices: list[int]
    pending_indices: list[int]
    orphan_png_indices: list[int]
    stale_manifest_indices: list[int]
    reconstructed_records: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_generation_dir(root: Path | str, model_name: str, method_name: str, run_id: str, create: bool = True) -> dict[str, Path]:
    base = Path(root) / _safe_name(model_name) / run_id / method_name
    paths = {
        "base_dir": base,
        "image_dir": base / "images",
        "samples_npz": base / "samples.npz",
        "labels_json": base / "labels.json",
        "labels_csv": base / "labels.csv",
        "manifest": base / "manifest.jsonl",
        "generation_meta": base / "generation_meta.json",
        "latency": base / "latency.json",
        "cache_stats": base / "cache_stats.json",
        "stdout_log": base / "stdout.log",
    }
    if create:
        paths["image_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def save_image_batch_png(
    tensor: Any,
    labels: list[int] | Any,
    start_idx: int | Sequence[int],
    image_dir: Path | str,
    value_range: str = "auto",
) -> list[dict[str, Any]]:
    import torch
    from PIL import Image

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    batch = tensor.detach().cpu().float()
    if batch.ndim != 4:
        raise ValueError("Expected image tensor with shape [B, C, H, W]")
    if batch.shape[1] not in {1, 3}:
        raise ValueError("Expected 1 or 3 channels")
    batch_size = int(batch.shape[0])
    if value_range == "auto":
        min_value = float(batch.min().item()) if batch.numel() else 0.0
        max_value = float(batch.max().item()) if batch.numel() else 1.0
        if min_value < 0.0:
            batch = (batch.clamp(-1.0, 1.0) + 1.0) / 2.0
        elif max_value > 1.0:
            batch = batch.clamp(0.0, 255.0) / 255.0
        else:
            batch = batch.clamp(0.0, 1.0)
    elif value_range == "minus_one_one":
        batch = (batch.clamp(-1.0, 1.0) + 1.0) / 2.0
    elif value_range == "zero_one":
        batch = batch.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown value_range: {value_range}")
    labels_list = [int(item) for item in labels]
    if len(labels_list) != batch_size:
        raise ValueError("Number of labels must match batch size")
    if isinstance(start_idx, Integral):
        indices = [int(start_idx) + offset for offset in range(batch_size)]
    elif isinstance(start_idx, Sequence) and not isinstance(start_idx, (str, bytes, bytearray)):
        indices = [int(item) for item in start_idx]
        if len(indices) != batch_size:
            raise ValueError("Number of explicit image indices must match batch size")
    else:
        raise TypeError("start_idx must be an int or a sequence of image indices")
    records: list[dict[str, Any]] = []
    for offset, image in enumerate(batch):
        index = indices[offset]
        array = (image * 255.0).round().to(torch.uint8)
        if array.shape[0] == 1:
            pil_array = array.squeeze(0).numpy()
            pil_image = Image.fromarray(pil_array)
        else:
            pil_array = array.permute(1, 2, 0).numpy()
            pil_image = Image.fromarray(pil_array)
        filename = f"{index:06d}.png"
        path = image_dir / filename
        pil_image.save(path)
        records.append({"index": index, "label": labels_list[offset], "path": str(path)})
    return records


def save_npz_samples(samples: Any, labels: list[int] | Any, path: Path | str) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    np.savez_compressed(path, samples=samples, labels=np.asarray(labels, dtype=np.int64))


def append_generation_manifest(manifest_path: Path | str, records: list[dict[str, Any]]) -> None:
    """Compatibility wrapper with upsert semantics rather than duplicate append."""

    upsert_manifest_records(manifest_path, records)


def load_generation_manifest(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest_index_map(
    path: Path | str,
    *,
    reject_duplicates: bool = True,
) -> dict[int, dict[str, Any]]:
    rows = load_generation_manifest(path)
    index_map: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for row in rows:
        if "index" not in row:
            raise ValueError(f"manifest row has no global index: {row}")
        index = int(row["index"])
        if index in index_map:
            duplicates.append(index)
        index_map[index] = dict(row)
    if duplicates and reject_duplicates:
        raise ValueError(f"duplicate manifest indices: {sorted(set(duplicates))}")
    return index_map


def upsert_manifest_records(
    manifest_path: Path | str,
    records: list[dict[str, Any]],
) -> None:
    index_map = load_manifest_index_map(manifest_path, reject_duplicates=True)
    for record in records:
        if "index" not in record:
            raise ValueError(f"manifest row has no global index: {record}")
        index_map[int(record["index"])] = dict(record)
    write_manifest_atomic(manifest_path, list(index_map.values()))


def reconcile_resume_state(
    indices: Sequence[int],
    labels: Sequence[int],
    image_dir: Path | str,
    manifest_path: Path | str,
    *,
    resume: bool,
    save_png: bool,
) -> ResumeReconciliation:
    requested = [int(index) for index in indices]
    if resume and not save_png:
        raise ValueError("Resume requires PNG completion markers in the current implementation.")
    if not resume:
        return ResumeReconciliation([], requested, [], [], [], [])

    root = Path(image_dir)
    manifest = Path(manifest_path)
    index_map = load_manifest_index_map(manifest, reject_duplicates=True)
    complete: list[int] = []
    pending: list[int] = []
    orphan_pngs: list[int] = []
    stale_rows: list[int] = []
    reconstructed: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index in requested:
        if index < 0 or index >= len(labels):
            raise ValueError(f"resume index {index} is outside the label schedule")
        expected_label = int(labels[index])
        expected_path = root / f"{index:06d}.png"
        png_exists = expected_path.is_file()
        row = index_map.get(index)
        if row is not None:
            if "label" not in row or int(row["label"]) != expected_label:
                raise ValueError(
                    f"manifest label mismatch for index {index}: "
                    f"manifest={row.get('label')!r}, expected={expected_label}"
                )
            raw_path = row.get("path")
            path_matches = raw_path is not None and _same_path(Path(str(raw_path)), expected_path)
            if not path_matches:
                repaired_row = dict(row)
                repaired_row["path"] = str(expected_path)
                repaired_row["resume_path_reconciled"] = True
                repaired.append(repaired_row)
                row = repaired_row
                warnings.append(
                    f"manifest path canonicalized for index {index}: "
                    f"{raw_path!r} -> {expected_path}"
                )

        if png_exists and row is not None:
            complete.append(index)
        elif png_exists:
            record = {
                "index": index,
                "label": expected_label,
                "path": str(expected_path),
                "resume_status": "orphan_png_reconciled",
            }
            reconstructed.append(record)
            orphan_pngs.append(index)
            complete.append(index)
            warnings.append(f"orphan PNG reconciled for index {index}")
        else:
            pending.append(index)
            if row is not None:
                stale_rows.append(index)
                warnings.append(f"stale manifest row requires regeneration for index {index}")

    if repaired or reconstructed:
        upsert_manifest_records(manifest, [*repaired, *reconstructed])
    return ResumeReconciliation(
        complete_indices=complete,
        pending_indices=pending,
        orphan_png_indices=orphan_pngs,
        stale_manifest_indices=stale_rows,
        reconstructed_records=reconstructed,
        warnings=warnings,
    )


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def write_manifest_atomic(
    manifest_path: Path | str,
    records: list[dict[str, Any]],
) -> None:
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: int(row["index"]))
    seen: set[int] = set()
    for row in ordered:
        index = int(row["index"])
        if index in seen:
            raise ValueError(f"duplicate manifest index: {index}")
        seen.add(index)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            for row in ordered:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_output_indices(
    image_dir: Path | str,
    expected_indices: Sequence[int] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    root = Path(image_dir)
    names = [child.name for child in root.iterdir() if child.is_file() and child.suffix.lower() == ".png"] if root.is_dir() else []
    invalid_names: list[str] = []
    indices: list[int] = []
    for name in names:
        stem = Path(name).stem
        if not stem.isdigit():
            invalid_names.append(name)
            continue
        indices.append(int(stem))
    counts = Counter(indices)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    duplicate_filenames = sorted(name for name, count in Counter(names).items() if count > 1)
    expected = set(int(index) for index in expected_indices) if expected_indices is not None else set(indices)
    actual = set(indices)
    report = {
        "image_count": len(names),
        "valid_numeric_image_count": len(indices),
        "indices": sorted(actual),
        "duplicate_indices": duplicates,
        "duplicate_png_filenames": duplicate_filenames,
        "missing_indices": sorted(expected - actual),
        "unexpected_indices": sorted(actual - expected),
        "invalid_filenames": sorted(invalid_names),
    }
    report["valid"] = not any(
        report[key]
        for key in (
            "duplicate_indices",
            "duplicate_png_filenames",
            "missing_indices",
            "unexpected_indices",
            "invalid_filenames",
        )
    )
    if strict and not report["valid"]:
        raise ValueError(f"output index validation failed: {report}")
    return report


def write_generation_meta(path: Path | str, meta: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **meta}
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def count_images(path: Path | str) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(1 for child in root.rglob("*") if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
