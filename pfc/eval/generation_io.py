from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    start_idx: int | list[int] | tuple[int, ...],
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
    if isinstance(start_idx, (list, tuple)):
        indices = [int(item) for item in start_idx]
        if len(indices) != len(labels_list):
            raise ValueError("Number of explicit image indices must match labels")
    else:
        indices = [int(start_idx) + offset for offset in range(len(labels_list))]
    records: list[dict[str, Any]] = []
    for offset, image in enumerate(batch):
        index = indices[offset]
        array = (image * 255.0).round().to(torch.uint8)
        if array.shape[0] == 1:
            pil_array = array.squeeze(0).numpy()
            pil_image = Image.fromarray(pil_array, mode="L")
        else:
            pil_array = array.permute(1, 2, 0).numpy()
            pil_image = Image.fromarray(pil_array, mode="RGB")
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
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_generation_manifest(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_generation_meta(path: Path | str, meta: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **meta}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def count_images(path: Path | str) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(1 for child in root.rglob("*") if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
