from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class CacheKey:
    model_name: str
    module_name: str
    cfg_branch: str
    solver_stage: str
    batch_signature: str | None = None


@dataclass
class CacheEntry:
    tensor: torch.Tensor
    step_idx: int
    t: float
    hit_count: int = 0
    refresh_count: int = 0


@dataclass
class CacheStats:
    total_calls: int = 0
    hits: int = 0
    misses: int = 0
    refreshes: int = 0
    disabled: int = 0
    by_module: dict[str, dict[str, int]] = field(default_factory=dict)

    def mark(self, module_name: str, field_name: str, count_total: bool = True) -> None:
        if count_total:
            self.total_calls += 1
        if field_name == "hits":
            self.hits += 1
        elif field_name == "misses":
            self.misses += 1
        elif field_name == "refreshes":
            self.refreshes += 1
        elif field_name == "disabled":
            self.disabled += 1
        else:
            raise ValueError(f"Unknown cache stat field: {field_name}")
        module_stats = self.by_module.setdefault(
            module_name,
            {"calls": 0, "hits": 0, "misses": 0, "refreshes": 0, "disabled": 0},
        )
        if count_total:
            module_stats["calls"] += 1
        module_stats[field_name] += 1

    def to_dict(self) -> dict[str, Any]:
        hit_rate = self.hits / self.total_calls if self.total_calls else 0.0
        return {
            "total_calls": self.total_calls,
            "hits": self.hits,
            "misses": self.misses,
            "refreshes": self.refreshes,
            "disabled": self.disabled,
            "hit_rate": hit_rate,
            "by_module": {name: dict(stats) for name, stats in sorted(self.by_module.items())},
        }


class RuntimeCacheState:
    def __init__(
        self,
        model_name: str = "JiT",
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.current_step_idx = -1
        self.current_t = 0.0
        self.cfg_branch = "unknown"
        self.solver_stage = "euler"
        self.enabled = enabled
        self.entries: dict[CacheKey, CacheEntry] = {}
        self.stats = CacheStats()

    def set_context(
        self,
        step_idx: int,
        t: float,
        cfg_branch: str,
        solver_stage: str = "euler",
    ) -> None:
        self.current_step_idx = int(step_idx)
        self.current_t = float(t)
        self.cfg_branch = str(cfg_branch)
        self.solver_stage = str(solver_stage)

    def make_key(self, module_name: str, batch_signature: str | None = None) -> CacheKey:
        return CacheKey(
            model_name=self.model_name,
            module_name=module_name,
            cfg_branch=self.cfg_branch,
            solver_stage=self.solver_stage,
            batch_signature=batch_signature,
        )

    def get(self, key: CacheKey) -> CacheEntry | None:
        return self.entries.get(key)

    def put(self, key: CacheKey, tensor: torch.Tensor) -> CacheEntry:
        entry = CacheEntry(
            tensor=tensor.detach(),
            step_idx=self.current_step_idx,
            t=self.current_t,
            refresh_count=1,
        )
        self.entries[key] = entry
        return entry

    def clear_entries(self) -> None:
        self.entries.clear()

    def clear(self) -> None:
        self.clear_entries()

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    def mark_hit(self, module_name: str) -> None:
        self.stats.mark(module_name, "hits", count_total=True)

    def mark_miss(self, module_name: str) -> None:
        self.stats.mark(module_name, "misses", count_total=True)

    def mark_refresh(self, module_name: str) -> None:
        self.stats.mark(module_name, "refreshes", count_total=False)

    def mark_disabled(self, module_name: str) -> None:
        self.stats.mark(module_name, "disabled", count_total=True)

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "enabled": self.enabled,
            "current_step_idx": self.current_step_idx,
            "current_t": self.current_t,
            "cfg_branch": self.cfg_branch,
            "solver_stage": self.solver_stage,
            "num_entries": len(self.entries),
            **self.stats.to_dict(),
        }
