from __future__ import annotations

from typing import Any

import torch

from pfc.cache.base_policy import CachePolicy, NoCachePolicy
from pfc.samplers.solver_state import SolverState


class UnifiedPixelFlowSampler:
    def __init__(
        self,
        adapter: Any,
        solver: str = "euler",
        steps: int = 20,
        cfg_scale: float = 1.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-4,
    ) -> None:
        if solver != "euler":
            raise NotImplementedError("Stage 0 UnifiedPixelFlowSampler only supports solver='euler'")
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.adapter = adapter
        self.solver = solver
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.cfg_interval = cfg_interval
        self.eps = eps

    def predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        cond: Any,
        uncond: Any | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        policy = cache_policy or NoCachePolicy()
        if not policy.should_compute():
            return policy.reuse(), {"cache": "reuse"}

        t_float = self._time_to_float(t)
        cfg_enabled = uncond is not None and self.cfg_scale > 1.0 and self._cfg_active(t_float)
        if cfg_enabled:
            v_uncond = self.adapter.forward_velocity(x, t, uncond, eps=self.eps).velocity
            v_cond = self.adapter.forward_velocity(x, t, cond, eps=self.eps).velocity
            velocity = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            diagnostics = {
                "cfg_enabled": True,
                "cfg_scale": self.cfg_scale,
                "v_cond": v_cond,
                "v_uncond": v_uncond,
            }
        else:
            velocity = self.adapter.forward_velocity(x, t, cond, eps=self.eps).velocity
            diagnostics = {"cfg_enabled": False, "cfg_scale": self.cfg_scale}

        policy.update(velocity=velocity, diagnostics=diagnostics)
        return velocity, diagnostics

    def sample(
        self,
        noise: torch.Tensor,
        cond: Any,
        uncond: Any | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = noise.clone()
        policy = cache_policy or NoCachePolicy()
        step_diagnostics: list[dict[str, Any]] = []
        time_grid = torch.linspace(0.0, 1.0, self.steps + 1, device=noise.device, dtype=noise.dtype)

        for step_idx in range(self.steps):
            t = time_grid[step_idx]
            t_next = time_grid[step_idx + 1]
            dt = t_next - t
            solver_state = SolverState(
                solver=self.solver,
                step_idx=step_idx,
                t=float(t.item()),
                t_next=float(t_next.item()),
                dt=float(dt.item()),
                mode="euler",
                cfg_scale=self.cfg_scale,
                cfg_enabled=uncond is not None and self.cfg_scale > 1.0 and self._cfg_active(float(t.item())),
            )
            policy.begin_step(solver_state=solver_state, x=x, cond=cond)
            velocity, diagnostics = self.predict_velocity(x, t, cond, uncond=uncond, cache_policy=policy)
            x = x + dt * velocity
            step_diagnostics.append(
                {
                    "solver_state": solver_state,
                    "cfg_enabled": diagnostics.get("cfg_enabled", False),
                    "velocity_shape": tuple(velocity.shape),
                }
            )

        return x, {"steps": step_diagnostics, "num_steps": self.steps}

    def _cfg_active(self, t: float) -> bool:
        interval_min, interval_max = self.cfg_interval
        return interval_min <= t <= interval_max

    @staticmethod
    def _time_to_float(t: torch.Tensor | float) -> float:
        if torch.is_tensor(t):
            return float(t.detach().float().mean().item())
        return float(t)

