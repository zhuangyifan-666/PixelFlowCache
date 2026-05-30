from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pfc.adapters.base import ModelAdapter


class JiTAdapter(ModelAdapter):
    """Lightweight JiT adapter skeleton.

    The official JiT model construction is intentionally lazy so importing this
    module does not require the JiT environment or a checkpoint.
    """

    def __init__(self, jit_repo: str | Path | None = None, checkpoint_dir: str | Path | None = None) -> None:
        super().__init__(name="JiT", model_type="xpred")
        self.jit_repo = Path(jit_repo) if jit_repo is not None else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.model: Any | None = None

    def build(self, **kwargs: Any) -> Any:
        # TODO: Construct the official JiT Denoiser from third_party/JiT and load
        # checkpoint-last.pth. Keep this lazy for Stage 0 smoke tests.
        if self.jit_repo is None:
            raise ValueError("jit_repo must be provided to build the official JiT model")
        try:
            import importlib

            return importlib.import_module("denoiser")
        except ImportError as exc:
            raise ImportError(
                "Could not import JiT official code. Add third_party/JiT to PYTHONPATH "
                "or run through scripts/run_official_jit_baseline.sh."
            ) from exc

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any, **kwargs: Any) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("JiTAdapter.build() has not been wired to construct the official model yet")
        return self.model(x, t, cond, **kwargs)

