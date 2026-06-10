from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import torch.nn as nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


def parse_layer_list(spec: str, num_blocks: int) -> list[int]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    stripped = spec.strip()
    normalized = stripped.lower()
    if normalized == "none":
        return []
    if normalized == "all":
        return list(range(num_blocks))
    if normalized == "early":
        return list(range(0, max(1, num_blocks // 4)))
    if normalized == "middle":
        return list(range(num_blocks // 4, (num_blocks * 3) // 4))
    if normalized == "late":
        return list(range((num_blocks * 3) // 4, num_blocks))
    if normalized.startswith("prefix:"):
        n = _parse_nonnegative_int(normalized.split(":", 1)[1], spec)
        return list(range(min(n, num_blocks)))
    if normalized.startswith("suffix:"):
        n = _parse_nonnegative_int(normalized.split(":", 1)[1], spec)
        start = max(0, num_blocks - n)
        return list(range(start, num_blocks))
    if normalized.startswith("range:"):
        return _parse_range(normalized, num_blocks)
    if normalized.startswith("every:"):
        stride = _parse_positive_int(normalized.split(":", 1)[1], spec)
        return list(range(0, num_blocks, stride))
    if normalized.startswith("complement:"):
        inner_spec = stripped.split(":", 1)[1]
        inner = set(parse_layer_list(inner_spec, num_blocks))
        return [idx for idx in range(num_blocks) if idx not in inner]
    if normalized.startswith("topk:"):
        return _parse_topk(stripped, num_blocks)
    if "," in normalized or normalized.isdigit():
        layers = []
        for item in normalized.split(","):
            item = item.strip()
            if not item:
                continue
            if not item.isdigit():
                raise ValueError(f"Invalid layer id in spec {spec!r}: {item!r}")
            layer_id = int(item)
            if layer_id < 0 or layer_id >= num_blocks:
                raise ValueError(f"Layer id {layer_id} out of range for {num_blocks} blocks")
            layers.append(layer_id)
        return sorted(dict.fromkeys(layers))
    raise ValueError(f"Unsupported layer spec: {spec}")


def _parse_nonnegative_int(value: str, spec: str) -> int:
    if not value.isdigit():
        raise ValueError(f"Expected non-negative integer in layer spec: {spec}")
    return int(value)


def _parse_positive_int(value: str, spec: str) -> int:
    parsed = _parse_nonnegative_int(value, spec)
    if parsed <= 0:
        raise ValueError(f"Expected positive integer in layer spec: {spec}")
    return parsed


def _parse_range(spec: str, num_blocks: int) -> list[int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("range spec must be range:<start>:<end>")
    start = _parse_nonnegative_int(parts[1], spec)
    end = _parse_nonnegative_int(parts[2], spec)
    if start > end:
        raise ValueError(f"Range start must be <= end in layer spec: {spec}")
    if end > num_blocks:
        raise ValueError(f"Range end {end} out of range for {num_blocks} blocks")
    return list(range(start, end))


def _parse_topk(spec: str, num_blocks: int) -> list[int]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("topk spec must be topk:<csv_path>:<k>")
    csv_path = Path(parts[1]).expanduser()
    k = int(parts[2])
    if k < 0:
        raise ValueError("topk k must be non-negative")
    rows: list[tuple[float, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            module_name = row.get("module_name", "")
            match = re.search(r"blocks\.(\d+)$", module_name)
            if not match:
                continue
            layer_id = int(match.group(1))
            if 0 <= layer_id < num_blocks:
                rows.append((float(row.get("mean_rel_l2_delta", "inf")), layer_id))
    rows.sort(key=lambda item: item[0])
    return sorted(dict.fromkeys(layer_id for _score, layer_id in rows[:k]))


def wrap_jit_blocks(
    denoiser_or_net: Any,
    cache_state: RuntimeCacheState,
    policy: FixedIntervalCachePolicy,
    layers: list[int],
) -> list[str]:
    net = getattr(denoiser_or_net, "net", denoiser_or_net)
    blocks = getattr(net, "blocks", None)
    if blocks is None:
        raise ValueError("Expected JiT net with a blocks ModuleList")
    wrapped: list[str] = []
    for layer_id in layers:
        if layer_id < 0 or layer_id >= len(blocks):
            raise ValueError(f"Layer id {layer_id} out of range for {len(blocks)} blocks")
        module_name = f"blocks.{layer_id}"
        if isinstance(blocks[layer_id], CachedModule):
            continue
        blocks[layer_id] = CachedModule(
            module=blocks[layer_id],
            module_name=module_name,
            cache_state=cache_state,
            policy=policy,
        )
        wrapped.append(module_name)
    return wrapped


def unwrap_jit_blocks(denoiser_or_net: Any) -> list[str]:
    net = getattr(denoiser_or_net, "net", denoiser_or_net)
    blocks = getattr(net, "blocks", None)
    if blocks is None:
        return []
    unwrapped: list[str] = []
    for layer_id, block in enumerate(blocks):
        if isinstance(block, CachedModule):
            blocks[layer_id] = block.module
            unwrapped.append(f"blocks.{layer_id}")
    return unwrapped
