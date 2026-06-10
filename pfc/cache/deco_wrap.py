from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import torch.nn as nn

from pfc.cache.cache_state import RuntimeCacheState
from pfc.cache.cached_module import CachedModule
from pfc.cache.fixed_interval_policy import FixedIntervalCachePolicy


_BACKBONE_RE = re.compile(r"^blocks\.\d+$")
_DECODER_RE = re.compile(r"^dec_net\.res_blocks\.\d+$")
_FINAL_NAMES = {"final_layer", "dec_net.final_layer"}


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


def is_deco_cache_candidate(name: str, module: nn.Module | None = None) -> bool:
    category = categorize_deco_module(name, module)
    lower = name.lower()
    if category == "norm_or_modulation":
        return False
    if any(token in lower for token in ("norm", "adaln", "modulation", "embed", "dropout")):
        return False
    return category in {"block", "decoder", "final"}


def deco_cache_unit_category(name: str, module: nn.Module | None = None) -> str:
    lower = name.lower()
    if _is_norm_or_modulation_name(lower):
        return "norm_or_modulation"
    if _BACKBONE_RE.match(name):
        return "backbone_block"
    if _DECODER_RE.match(name):
        return "decoder_block"
    if lower in _FINAL_NAMES or lower.endswith(".final_layer"):
        return "final_head"
    if module is not None and _is_tiny_module(module):
        return "tiny_module"
    category = categorize_deco_module(name, module)
    if category == "norm_or_modulation":
        return "norm_or_modulation"
    if category == "final":
        return "final_head"
    return "other"


def is_safe_deco_cache_unit(name: str, module: nn.Module | None = None) -> bool:
    category = deco_cache_unit_category(name, module)
    if category in {"norm_or_modulation", "tiny_module", "other"}:
        return False
    if _is_norm_or_modulation_name(name.lower()):
        return False
    if module is not None and _is_tiny_module(module):
        return False
    return category in {"backbone_block", "decoder_block", "final_head"} and is_deco_cache_candidate(name, module)


def parse_deco_cache_spec(spec: str, module_candidates: list[str]) -> list[str]:
    candidate_set = set(module_candidates)
    normalized = spec.strip()
    lower = normalized.lower()
    if not normalized:
        raise ValueError("DeCo cache spec must not be empty")
    backbone = [name for name in module_candidates if deco_cache_unit_category(name) == "backbone_block"]
    decoder = [name for name in module_candidates if deco_cache_unit_category(name) == "decoder_block"]
    final = [name for name in module_candidates if deco_cache_unit_category(name) == "final_head"]
    if lower == "none":
        return []
    if lower == "all_candidates":
        return backbone + decoder + final
    if lower in {"backbone_blocks", "backbone_only"}:
        return backbone
    if lower in {"decoder_blocks", "decoder_only_no_final"}:
        return decoder
    if lower in {"final", "final_only"}:
        return final
    if lower == "decoder_plus_final":
        return decoder + final
    if lower == "backbone_plus_final":
        return backbone + final
    if lower == "backbone_plus_decoder_no_final":
        return backbone + decoder
    if lower.startswith("late_backbone_plus_final:"):
        return _late_backbone(lower, normalized, backbone) + final
    if lower.startswith("late_backbone_only:"):
        return _late_backbone(lower, normalized, backbone)
    if lower.startswith("topk:"):
        return _parse_topk_spec(normalized, candidate_set)
    explicit = [item.strip() for item in normalized.split(",") if item.strip()]
    if not explicit:
        return []
    missing = [name for name in explicit if name not in candidate_set]
    if missing:
        raise ValueError(f"Explicit DeCo cache modules are not candidates: {missing}")
    return explicit


def wrap_deco_modules(
    net: nn.Module,
    cache_state: RuntimeCacheState,
    policy: FixedIntervalCachePolicy,
    module_names: list[str],
) -> list[str]:
    module_map = dict(net.named_modules())
    wrapped: list[str] = []
    for module_name in module_names:
        module = module_map.get(module_name)
        if module is None:
            raise ValueError(f"DeCo module not found: {module_name}")
        if isinstance(module, CachedModule):
            continue
        if not is_safe_deco_cache_unit(module_name, module):
            continue
        parent, child_name = _resolve_parent(net, module_name)
        parent._modules[child_name] = CachedModule(
            module=module,
            module_name=module_name,
            cache_state=cache_state,
            policy=policy,
        )
        wrapped.append(module_name)
    return wrapped


def _parse_topk_spec(spec: str, candidate_set: set[str]) -> list[str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("topk spec must be topk:<candidate_csv>:<k>")
    csv_path = Path(parts[1]).expanduser()
    k = int(parts[2])
    if k < 0:
        raise ValueError("topk k must be non-negative")
    rows: list[tuple[float, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            module_name = row.get("module_name", "")
            if module_name not in candidate_set or not _name_matches_safe_unit(module_name):
                continue
            score_text = row.get("mean_rel_l2_delta") or row.get("rel_l2_mean") or "inf"
            rows.append((float(score_text), module_name))
    rows.sort(key=lambda item: item[0])
    return [name for _score, name in rows[:k]]


def _late_backbone(lower: str, original: str, backbone: list[str]) -> list[str]:
    try:
        count = int(lower.rsplit(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Expected integer n in DeCo cache spec: {original}") from exc
    if count <= 0:
        raise ValueError(f"Expected positive integer n in DeCo cache spec: {original}")
    return backbone[-count:]


def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    if not parts:
        raise ValueError("module_name must not be empty")
    parent: Any = root
    for part in parts[:-1]:
        if not isinstance(parent, nn.Module) or part not in parent._modules:
            raise ValueError(f"Could not resolve parent for module {module_name}")
        parent = parent._modules[part]
    child_name = parts[-1]
    if child_name not in parent._modules:
        raise ValueError(f"Could not resolve child {child_name} for module {module_name}")
    return parent, child_name


def _name_matches_safe_unit(name: str) -> bool:
    return deco_cache_unit_category(name) in {"backbone_block", "decoder_block", "final_head"}


def _is_norm_or_modulation_name(lower_name: str) -> bool:
    return any(token in lower_name for token in ("norm", "adaln", "modulation", "q_norm", "k_norm"))


def _is_tiny_module(module: nn.Module) -> bool:
    if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout, nn.Identity)):
        return True
    return sum(param.numel() for param in module.parameters(recurse=False)) == 0 and len(list(module.children())) == 0
