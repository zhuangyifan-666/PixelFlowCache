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


@dataclass(frozen=True)
class InputSignature:
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None
    batch_size: int | None
    session_id: str | int | None = None

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        session_id: str | int | None = None,
    ) -> "InputSignature":
        return cls(
            shape=tuple(int(size) for size in tensor.shape),
            dtype=str(tensor.dtype),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            batch_size=int(tensor.shape[0]) if tensor.ndim > 0 else None,
            session_id=session_id,
        )


@dataclass
class CacheEntry:
    tensor: torch.Tensor
    step_idx: int
    t: float
    hit_count: int = 0
    refresh_count: int = 0
    input_signature: InputSignature | None = None
    output_shape: tuple[int, ...] | None = None
    output_dtype: str | None = None
    output_device: str | None = None


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
        clone_on_store: bool = False,
    ) -> None:
        self.model_name = model_name
        self.current_step_idx = -1
        self.current_t = 0.0
        self.cfg_branch = "unknown"
        self.solver_stage = "euler"
        self.enabled = enabled
        self.clone_on_store = bool(clone_on_store)
        self.session_id: str | int = 0
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

    def put(
        self,
        key: CacheKey,
        tensor: torch.Tensor,
        *,
        input_signature: InputSignature | None = None,
    ) -> CacheEntry:
        stored = tensor.detach().clone() if self.clone_on_store else tensor.detach()
        entry = CacheEntry(
            tensor=stored,
            step_idx=self.current_step_idx,
            t=self.current_t,
            refresh_count=1,
            input_signature=input_signature,
            output_shape=tuple(int(size) for size in tensor.shape),
            output_dtype=str(tensor.dtype),
            output_device=str(tensor.device),
        )
        self.entries[key] = entry
        return entry

    def begin_batch(
        self,
        *,
        session_id: str | int | None = None,
        batch_signature: str | None = None,
    ) -> None:
        self.entries.clear()
        if session_id is None:
            current = int(self.session_id) if isinstance(self.session_id, int) else 0
            self.session_id = current + 1
        else:
            self.session_id = session_id

    def clear_entries(self) -> None:
        self.begin_batch()

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
            "session_id": self.session_id,
            "clone_on_store": self.clone_on_store,
            "num_entries": len(self.entries),
            **self.stats.to_dict(),
        }
