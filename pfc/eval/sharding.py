from __future__ import annotations

from pathlib import Path
from typing import Any

from pfc.eval.generation_io import ResumeReconciliation, reconcile_resume_state


def compute_shard_indices(
    num_images: int,
    num_shards: int,
    shard_index: int,
    shard_mode: str = "strided",
) -> list[int]:
    if num_images < 0:
        raise ValueError("num_images must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if shard_mode == "strided":
        return list(range(shard_index, num_images, num_shards))
    if shard_mode == "contiguous":
        base, extra = divmod(num_images, num_shards)
        start = shard_index * base + min(shard_index, extra)
        length = base + (1 if shard_index < extra else 0)
        return list(range(start, start + length))
    raise ValueError(f"unsupported shard_mode: {shard_mode}")


def pending_indices(
    indices: list[int],
    image_dir: Path | str,
    *,
    resume: bool,
    save_png: bool,
) -> list[int]:
    if resume and not save_png:
        raise ValueError("Resume requires PNG completion markers in the current implementation.")
    if not resume:
        return list(indices)
    root = Path(image_dir)
    return [index for index in indices if not (root / f"{index:06d}.png").is_file()]


def apply_shard_paths(
    paths: dict[str, Path],
    *,
    num_shards: int,
    shard_index: int,
    suffix: str | None = None,
) -> dict[str, Path]:
    if num_shards <= 1:
        return paths
    resolved = dict(paths)
    tag = suffix or f"_shard{shard_index}"
    base = resolved["base_dir"]
    for key, stem in (
        ("manifest", "manifest"),
        ("generation_meta", "generation_meta"),
        ("latency", "latency"),
        ("cache_stats", "cache_stats"),
    ):
        extension = ".jsonl" if key == "manifest" else ".json"
        resolved[key] = base / f"{stem}{tag}{extension}"
    return resolved


def shard_metadata(
    *,
    num_shards: int,
    shard_index: int,
    shard_mode: str,
    indices: list[int],
) -> dict[str, Any]:
    return {
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "shard_mode": shard_mode,
        "shard_indices": list(indices),
    }
