from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pfc.adapters.base import ModelAdapter


class DeCoAdapter(ModelAdapter):
    """Lightweight DeCo adapter skeleton with lazy official imports."""

    def __init__(self, deco_repo: str | Path | None = None, checkpoint: str | Path | None = None) -> None:
        super().__init__(name="DeCo", model_type="vpred")
        self.deco_repo = Path(deco_repo) if deco_repo is not None else None
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.model: Any | None = None

    def build(self, **kwargs: Any) -> Any:
        # TODO: Construct the official LightningModel from third_party/DeCo and
        # load the DeCo checkpoint. Keep this lazy for Stage 0 smoke tests.
        if self.deco_repo is None:
            raise ValueError("deco_repo must be provided to build the official DeCo model")
        try:
            import importlib

            return importlib.import_module("src.lightning_model")
        except ImportError as exc:
            raise ImportError(
                "Could not import DeCo official code. Add third_party/DeCo to PYTHONPATH "
                "or run through scripts/run_official_deco_baseline.sh."
            ) from exc

    def forward_raw(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any, **kwargs: Any) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DeCoAdapter.build() has not been wired to construct the official model yet")
        return self.model(x, t, cond, **kwargs)

