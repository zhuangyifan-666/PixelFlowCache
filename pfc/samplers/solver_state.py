from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolverState:
    solver: str
    step_idx: int
    t: float
    t_next: float
    dt: float
    mode: str = "euler"
    cfg_scale: float = 1.0
    cfg_enabled: bool = False

