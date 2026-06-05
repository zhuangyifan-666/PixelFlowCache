from __future__ import annotations

import csv
import json
import random
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
        target.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
        return {"json": target}
    if target.suffix.lower() == ".csv":
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(target, labels)
        return {"csv": target}
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "labels.json"
    csv_path = target / "labels.csv"
    json_path.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
    _write_csv(csv_path, labels)
    return {"json": json_path, "csv": csv_path}


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

