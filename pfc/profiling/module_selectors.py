from __future__ import annotations

from typing import Any

import torch.nn as nn


def select_jit_blocks(model_or_denoiser: Any) -> list[tuple[str, nn.Module]]:
    net = getattr(model_or_denoiser, "net", model_or_denoiser)
    blocks = getattr(net, "blocks", None)
    if blocks is None:
        return []
    return [(f"net.blocks.{idx}", block) for idx, block in enumerate(blocks)]


def generic_transformer_block_filter(name: str, module: nn.Module) -> bool:
    lower_name = name.lower()
    class_name = module.__class__.__name__.lower()
    if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout, nn.Conv2d)):
        return False
    block_like = any(token in lower_name for token in ("blocks.", "block.", "decoder", "head", "final"))
    class_like = "block" in class_name or "decoder" in class_name
    return block_like or class_like


def categorize_deco_module(name: str, module: nn.Module | None = None) -> str:
    lower = name.lower()
    class_name = module.__class__.__name__.lower() if module is not None else ""
    if any(token in lower for token in ("adaln", "modulation", "norm", "q_norm", "k_norm")):
        return "norm_or_modulation"
    if "decoder" in lower or lower.startswith("dec_net") or ".decoder" in lower:
        if "final" in lower or "head" in lower:
            return "final"
        return "decoder"
    if "final" in lower or "head" in lower:
        return "final"
    if "blocks" in lower or "cond_blocks" in lower or "block" in class_name:
        return "block"
    return "other"


def is_deco_stage2_cache_candidate(name: str, module: nn.Module | None = None) -> bool:
    category = categorize_deco_module(name, module)
    lower = name.lower()
    if category == "norm_or_modulation":
        return False
    if any(token in lower for token in ("norm", "adaln", "modulation", "embed", "dropout")):
        return False
    return category in {"block", "decoder", "final"}


def select_deco_candidate_modules(net: nn.Module) -> list[tuple[str, nn.Module]]:
    candidates: list[tuple[str, nn.Module]] = []
    keywords = ("blocks", "cond_blocks", "encoder.blocks", "decoder", "denoiser", "final", "head")
    for name, module in net.named_modules():
        if not name:
            continue
        lower = name.lower()
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout)):
            continue
        if any(keyword in lower for keyword in keywords) or generic_transformer_block_filter(name, module):
            candidates.append((name, module))
    # Keep the hook set bounded; DeCo has many nested modules.
    filtered: list[tuple[str, nn.Module]] = []
    seen_prefixes: set[str] = set()
    for name, module in candidates:
        parts = name.split(".")
        compact_name = ".".join(parts[:3]) if len(parts) > 3 else name
        if compact_name in seen_prefixes:
            continue
        seen_prefixes.add(compact_name)
        filtered.append((name, module))
    return filtered
