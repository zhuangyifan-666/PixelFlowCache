from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pfc.cache.wrap import wrap_jit_blocks
from pfc.core.boundary_spec import BoundaryGranularity, BoundaryRole, BoundarySet, BoundarySpec, PredictionType
from pfc.core.model_adapter import PixelDiffusionModelAdapter


class PixelGenBoundaryAdapter(PixelDiffusionModelAdapter):
    model_name = "pixelgen"
    prediction_type = PredictionType.XPRED
    time_direction = "noise_to_image"

    def list_boundary_candidates(self, model: Any) -> list[BoundarySpec]:
        denoiser = self._denoiser(model)
        blocks = self._blocks(denoiser)
        module_names = tuple(f"blocks.{idx}" for idx in range(len(blocks)))
        return [
            BoundarySpec(
                name="jit_style_blocks",
                module_names=module_names,
                role=BoundaryRole.BACKBONE,
                granularity=BoundaryGranularity.WHOLE_BACKBONE,
                description="All PixelGen JiT-style transformer backbone blocks.",
                is_quality_critical=True,
                is_speed_critical=True,
                prediction_role="xpred_backbone",
                extra={"cache_unit": "pixelgen_jit_blocks"},
            )
        ]

    def default_boundary_set(self, model: Any, preset_name: str | None = None) -> BoundarySet:
        spec = self._candidate_by_name(model, "jit_style_blocks")
        return BoundarySet(
            name="pixelgen_jit_style_blocks",
            boundaries=(spec,),
            description=f"Default PixelGen PixBFC boundary set for {preset_name or 'jit_style_blocks'}.",
        )

    def wrap_boundary_set(
        self,
        model: Any,
        boundary_set: BoundarySet,
        cache_state: Any,
        policy: Any,
    ) -> list[str]:
        denoiser = self._denoiser(model)
        layer_ids = self._layer_ids(boundary_set.module_names())
        return wrap_jit_blocks(denoiser, cache_state, policy, layer_ids)

    def output_to_velocity(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | float,
        eps: float = 5e-2,
    ) -> torch.Tensor:
        return super().output_to_velocity(output, x, t, eps=eps)

    def cache_proxy(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any | None = None) -> torch.Tensor:
        return x

    def branch_mode(self) -> str:
        return "cfg_cat"

    def _candidate_by_name(self, model: Any, name: str) -> BoundarySpec:
        for candidate in self.list_boundary_candidates(model):
            if candidate.name == name:
                return candidate
        raise ValueError(f"PixelGen boundary candidate not found: {name}")

    @classmethod
    def _denoiser(cls, model: Any) -> nn.Module:
        if hasattr(model, "blocks"):
            return model
        if getattr(model, "eval_original_model", False) and hasattr(model, "denoiser"):
            return getattr(model, "denoiser")
        if hasattr(model, "ema_denoiser"):
            return getattr(model, "ema_denoiser")
        if hasattr(model, "denoiser"):
            return getattr(model, "denoiser")
        raise ValueError("Expected PixelGen denoiser or wrapper with ema_denoiser/denoiser")

    @staticmethod
    def _blocks(denoiser: Any) -> Any:
        blocks = getattr(denoiser, "blocks", None)
        if blocks is None:
            raise ValueError("Expected PixelGen denoiser with a blocks attribute")
        return blocks

    @staticmethod
    def _layer_ids(module_names: tuple[str, ...]) -> list[int]:
        layer_ids: list[int] = []
        for module_name in module_names:
            prefix, _, suffix = module_name.partition(".")
            if prefix != "blocks" or not suffix.isdigit():
                raise ValueError(f"Unsupported PixelGen boundary module name: {module_name}")
            layer_ids.append(int(suffix))
        return sorted(dict.fromkeys(layer_ids))
