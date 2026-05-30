from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from pfc.profiling.jsonl import JsonlWriter
from pfc.profiling.tensor_stats import _first_tensor, summarize_delta, summarize_tensor


class FeatureRecorder:
    def __init__(
        self,
        module_filter: Callable[[str, nn.Module], bool],
        writer: JsonlWriter,
        model_name: str,
        keep_previous: bool = True,
        previous_on_cpu: bool = True,
        previous_dtype: str = "float16",
        previous_key_fields: tuple[str, ...] = ("module_name", "cfg_branch"),
        split_batch_dim0: bool = False,
    ) -> None:
        self.module_filter = module_filter
        self.writer = writer
        self.model_name = model_name
        self.keep_previous = keep_previous
        self.previous_on_cpu = previous_on_cpu
        self.previous_dtype = previous_dtype
        self.previous_key_fields = previous_key_fields
        self.split_batch_dim0 = split_batch_dim0
        self.handles: list[Any] = []
        self.previous: dict[str, torch.Tensor] = {}
        self.context: dict[str, Any] = {
            "step_idx": None,
            "t": None,
            "solver_stage": "euler",
            "cfg_branch": "unknown",
            "extra": {},
        }
        self.record_count = 0

    def attach(self, model: nn.Module) -> None:
        self.remove()
        for name, module in model.named_modules():
            if name and self.module_filter(name, module):
                self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def set_context(
        self,
        step_idx: int | None,
        t: float | None,
        solver_stage: str = "euler",
        cfg_branch: str = "unknown",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.context = {
            "step_idx": step_idx,
            "t": t,
            "solver_stage": solver_stage,
            "cfg_branch": cfg_branch,
            "extra": extra or {},
        }

    def clear_previous(self) -> None:
        self.previous.clear()

    def close(self) -> None:
        self.remove()
        self.writer.close()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, module_name: str, module: nn.Module) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                return
            record: dict[str, Any] = {
                "record_type": "feature",
                "model_name": self.model_name,
                "module_name": module_name,
                "module_kind": module.__class__.__name__,
                "step_idx": self.context.get("step_idx"),
                "t": self.context.get("t"),
                "solver_stage": self.context.get("solver_stage"),
                "cfg_branch": self.context.get("cfg_branch"),
                "timestamp": time.time(),
                "tensor": summarize_tensor(tensor, name=module_name),
            }
            if self.context.get("extra"):
                extra = self.context["extra"]
                record_extra = {key: value for key, value in extra.items() if key != "module_categories"}
                if record_extra:
                    record["extra"] = record_extra
                module_categories = extra.get("module_categories")
                if isinstance(module_categories, dict) and module_name in module_categories:
                    record["module_category"] = module_categories[module_name]
            if self.split_batch_dim0:
                split_stats = self._summarize_split_batch(tensor, module_name)
                if split_stats:
                    record["split_batch_dim0"] = split_stats
            previous_key = self._previous_key(module_name)
            previous = self.previous.get(previous_key)
            if previous is not None:
                record["temporal_delta"] = summarize_delta(tensor, previous)
            self.writer.write(record)
            self.record_count += 1
            if self.keep_previous:
                self.previous[previous_key] = self._store_previous(tensor)

        return hook

    def _previous_key(self, module_name: str) -> str:
        values: dict[str, Any] = {
            "module_name": module_name,
            "cfg_branch": self.context.get("cfg_branch", "unknown"),
            "solver_stage": self.context.get("solver_stage", "unknown"),
        }
        return "::".join(str(values.get(field, "")) for field in self.previous_key_fields)

    def _summarize_split_batch(self, tensor: torch.Tensor, module_name: str) -> dict[str, Any] | None:
        extra = self.context.get("extra") or {}
        batch_size = extra.get("cfg_cat_batch_size", extra.get("batch_size"))
        if not isinstance(batch_size, int) or batch_size <= 0:
            return None
        if tensor.ndim == 0 or tensor.shape[0] != batch_size * 2:
            return None
        uncond = tensor[:batch_size]
        cond = tensor[batch_size:]
        return {
            "batch_size": batch_size,
            "uncond": summarize_tensor(uncond, name=f"{module_name}.uncond"),
            "cond": summarize_tensor(cond, name=f"{module_name}.cond"),
            "cond_minus_uncond": summarize_tensor(cond - uncond, name=f"{module_name}.cond_minus_uncond"),
        }

    def _store_previous(self, tensor: torch.Tensor) -> torch.Tensor:
        stored = tensor.detach()
        if self.previous_dtype == "float16":
            stored = stored.to(dtype=torch.float16)
        elif self.previous_dtype == "bfloat16":
            stored = stored.to(dtype=torch.bfloat16)
        elif self.previous_dtype == "float32":
            stored = stored.float()
        if self.previous_on_cpu:
            stored = stored.cpu()
        return stored
