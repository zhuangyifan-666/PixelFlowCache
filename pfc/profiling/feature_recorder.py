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
    ) -> None:
        self.module_filter = module_filter
        self.writer = writer
        self.model_name = model_name
        self.keep_previous = keep_previous
        self.previous_on_cpu = previous_on_cpu
        self.previous_dtype = previous_dtype
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
                record["extra"] = self.context["extra"]
            previous = self.previous.get(module_name)
            if previous is not None:
                record["temporal_delta"] = summarize_delta(tensor, previous)
            self.writer.write(record)
            self.record_count += 1
            if self.keep_previous:
                self.previous[module_name] = self._store_previous(tensor)

        return hook

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

