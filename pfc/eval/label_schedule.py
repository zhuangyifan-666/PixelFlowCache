from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def make_imagenet_class_balanced_labels(
    num_images: int,
    num_classes: int = 1000,
    seed: int = 0,
    shuffle: bool = False,
) -> list[int]:
    if num_images < 0:
        raise ValueError("num_images must be non-negative")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    full_repeats, remainder = divmod(num_images, num_classes)
    labels = list(range(num_classes)) * full_repeats + list(range(remainder))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(labels)
    return labels


def save_label_schedule(labels: list[int], path: Path | str) -> dict[str, Path]:
    target = Path(path)
    if target.suffix.lower() == ".json":
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(target, labels)
        return {"json": target}
    if target.suffix.lower() == ".csv":
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(target, labels)
        return {"csv": target}
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "labels.json"
    csv_path = target / "labels.csv"
    _write_json_atomic(json_path, labels)
    _write_csv(csv_path, labels)
    return {"json": json_path, "csv": csv_path}


def ensure_label_schedule(labels: list[int], path: Path | str) -> dict[str, Path]:
    """Create a schedule or reject a resume run with different labels."""

    target = Path(path)
    existing_path = target / "labels.json" if target.is_dir() or not target.suffix else target
    if existing_path.exists():
        try:
            existing = load_label_schedule(existing_path)
        except Exception as exc:
            raise ValueError(f"Existing label schedule is unreadable: {existing_path}: {exc}") from exc
        if [int(value) for value in existing] != [int(value) for value in labels]:
            raise RuntimeError(f"Existing label schedule differs: {existing_path}")
        if target.suffix:
            return {target.suffix.lstrip(".").lower(): target}
        return {"json": target / "labels.json", "csv": target / "labels.csv"}
    return save_label_schedule(labels, target)


def load_label_schedule(path: Path | str) -> list[int]:
    source = Path(path)
    if source.is_dir():
        source = source / "labels.json"
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        raw = payload["labels"] if isinstance(payload, dict) else payload
        return [int(item) for item in raw]
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return [int(row["label"]) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported label schedule file: {source}")


def label_schedule_summary(labels: list[int]) -> dict[str, Any]:
    counts = Counter(labels)
    if not labels:
        return {
            "num_images": 0,
            "num_classes_present": 0,
            "min_count": 0,
            "max_count": 0,
            "balanced": True,
        }
    values = list(counts.values())
    return {
        "num_images": len(labels),
        "num_classes_present": len(counts),
        "min_label": min(labels),
        "max_label": max(labels),
        "min_count": min(values),
        "max_count": max(values),
        "balanced": max(values) - min(values) <= 1,
    }


def _write_csv(path: Path, labels: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "label"])
        writer.writeheader()
        for idx, label in enumerate(labels):
            writer.writerow({"index": idx, "label": int(label)})


def _write_json_atomic(path: Path, labels: list[int]) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump({"labels": [int(label) for label in labels]}, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

