from __future__ import annotations

import time
from typing import Any

import torch

from pfc.profiling.jsonl import JsonlWriter
from pfc.profiling.tensor_stats import summarize_tensor


class VelocityRecorder:
    def __init__(self, writer: JsonlWriter) -> None:
        self.writer = writer
        self.count = 0

    def log_velocity(
        self,
        *,
        model_name: str,
        step_idx: int,
        t: float,
        t_next: float,
        dt: float,
        branch: str,
        v: torch.Tensor,
        cfg_scale: float = 1.0,
        cfg_enabled: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "record_type": "velocity",
            "model_name": model_name,
            "step_idx": step_idx,
            "t": t,
            "t_next": t_next,
            "dt": dt,
            "branch": branch,
            "cfg_scale": cfg_scale,
            "cfg_enabled": cfg_enabled,
            "v": summarize_tensor(v, name="v"),
            "timestamp": time.time(),
        }
        if extra:
            record["extra"] = extra
        self.writer.write(record)
        self.count += 1

    def log_xpred_conversion(
        self,
        *,
        model_name: str,
        step_idx: int,
        t: float,
        t_next: float,
        dt: float,
        branch: str,
        x0_pred: torch.Tensor,
        v: torch.Tensor,
        x_current: torch.Tensor,
        cfg_scale: float = 1.0,
        cfg_enabled: bool = False,
        eps: float = 1e-4,
        extra: dict[str, Any] | None = None,
    ) -> None:
        amplification = 1.0 / max(1.0 - float(t), eps)
        record = {
            "record_type": "xpred_velocity",
            "model_name": model_name,
            "step_idx": step_idx,
            "t": t,
            "t_next": t_next,
            "dt": dt,
            "branch": branch,
            "cfg_scale": cfg_scale,
            "cfg_enabled": cfg_enabled,
            "amplification": amplification,
            "x0_pred": summarize_tensor(x0_pred, name="x0_pred"),
            "v": summarize_tensor(v, name="v"),
            "x_current": summarize_tensor(x_current, name="x_current"),
            "timestamp": time.time(),
        }
        if extra:
            record["extra"] = extra
        self.writer.write(record)
        self.count += 1

    def close(self) -> None:
        self.writer.close()

