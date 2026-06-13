from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pfc.cache.deco_wrap import deco_cache_unit_category, wrap_deco_modules
from pfc.core.boundary_spec import BoundaryGranularity, BoundaryRole, BoundarySet, BoundarySpec, PredictionType
from pfc.core.model_adapter import PixelDiffusionModelAdapter


class DeCoBoundaryAdapter(PixelDiffusionModelAdapter):
    model_name = "deco"
    prediction_type = PredictionType.VPRED
    time_direction = "noise_to_image"

    def list_boundary_candidates(self, model: nn.Module) -> list[BoundarySpec]:
        groups = self._group_candidates(model)
        specs: list[BoundarySpec] = []
        if groups["backbone"]:
            specs.append(
                BoundarySpec(
                    name="backbone_blocks",
                    module_names=tuple(groups["backbone"]),
                    role=BoundaryRole.BACKBONE,
                    granularity=BoundaryGranularity.MODULE_GROUP,
                    description="DeCo transformer backbone blocks.",
                    is_speed_critical=True,
                    prediction_role="vpred_backbone",
                )
            )
        if groups["decoder"]:
            specs.append(
                BoundarySpec(
                    name="decoder_blocks",
                    module_names=tuple(groups["decoder"]),
                    role=BoundaryRole.DECODER,
                    granularity=BoundaryGranularity.MODULE_GROUP,
                    description="DeCo decoder residual blocks.",
                    is_speed_critical=True,
                    prediction_role="vpred_decoder",
                )
            )
        if groups["final"]:
            specs.append(
                BoundarySpec(
                    name="final_output",
                    module_names=tuple(groups["final"]),
                    role=BoundaryRole.FINAL_OUTPUT,
                    granularity=BoundaryGranularity.OUTPUT_BOUNDARY,
                    description="DeCo final velocity/output boundary.",
                    is_quality_critical=True,
                    prediction_role="vpred_output",
                )
            )
        backbone_plus_final = tuple(groups["backbone"] + groups["final"])
        if backbone_plus_final:
            specs.append(
                BoundarySpec(
                    name="backbone_plus_final",
                    module_names=backbone_plus_final,
                    role=BoundaryRole.CUSTOM,
                    granularity=BoundaryGranularity.MODULE_GROUP,
                    description="DeCo backbone blocks plus final output boundary.",
                    is_quality_critical=True,
                    is_speed_critical=True,
                )
            )
        all_candidates = tuple(groups["backbone"] + groups["decoder"] + groups["final"])
        if all_candidates:
            specs.append(
                BoundarySpec(
                    name="all_candidates",
                    module_names=all_candidates,
                    role=BoundaryRole.CUSTOM,
                    granularity=BoundaryGranularity.MODULE_GROUP,
                    description="All DeCo cache candidates used by the final Stage 4A all-candidates preset.",
                    is_quality_critical=True,
                    is_speed_critical=True,
                )
            )
        return specs

    def default_boundary_set(self, model: nn.Module, preset_name: str | None = None) -> BoundarySet:
        name = "all_candidates"
        if preset_name in {"bfc_backbone_plus_final_t02_10", "backbone_plus_final"}:
            name = "backbone_plus_final"
        elif preset_name in {"bfc_all_candidates_t02_10", "all_candidates", None, "teacache_style", "seacache_style"}:
            name = "all_candidates"
        spec = self._candidate_by_name(model, name)
        return BoundarySet(
            name=f"deco_{name}",
            boundaries=(spec,),
            description=f"Default DeCo PixBFC boundary set for {preset_name or name}.",
        )

    def wrap_boundary_set(
        self,
        model: nn.Module,
        boundary_set: BoundarySet,
        cache_state: Any,
        policy: Any,
    ) -> list[str]:
        return wrap_deco_modules(model, cache_state, policy, list(boundary_set.module_names()))

    def cache_proxy(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any | None = None) -> torch.Tensor:
        return x

    def branch_mode(self) -> str:
        return "cfg_cat"

    def _candidate_by_name(self, model: nn.Module, name: str) -> BoundarySpec:
        for candidate in self.list_boundary_candidates(model):
            if candidate.name == name:
                return candidate
        raise ValueError(f"DeCo boundary candidate not found: {name}")

    @staticmethod
    def _group_candidates(model: nn.Module) -> dict[str, list[str]]:
        groups = {"backbone": [], "decoder": [], "final": []}
        for name, module in model.named_modules():
            if not name:
                continue
            category = deco_cache_unit_category(name, module)
            lower = name.lower()
            if category == "backbone_block" or lower.startswith("blocks."):
                groups["backbone"].append(name)
            elif category == "decoder_block" or lower.startswith("dec_net.res_blocks."):
                groups["decoder"].append(name)
            elif category == "final_head" or lower in {"final_layer", "dec_net.final_layer"} or lower.endswith(".final_layer"):
                groups["final"].append(name)
        return {key: list(dict.fromkeys(value)) for key, value in groups.items()}
