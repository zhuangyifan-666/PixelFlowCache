from __future__ import annotations

from pfc.adapters.deco_boundary_adapter import DeCoBoundaryAdapter
from pfc.adapters.jit_boundary_adapter import JiTBoundaryAdapter
from pfc.core.registry import register_adapter

register_adapter("deco", DeCoBoundaryAdapter)
register_adapter("jit", JiTBoundaryAdapter)

__all__ = ["DeCoBoundaryAdapter", "JiTBoundaryAdapter"]
