from __future__ import annotations

from typing import Any

import torch

from pfc.cache.wrap import wrap_jit_blocks
from pfc.core.boundary_spec import BoundaryGranularity, BoundaryRole, BoundarySet, BoundarySpec, PredictionType
from pfc.core.model_adapter import PixelDiffusionModelAdapter


class JiTBoundaryAdapter(PixelDiffusionModelAdapter):
    model_name = "jit"
    prediction_type = PredictionType.XPRED
    time_direction = "noise_to_image"

    def list_boundary_candidates(self, model: Any) -> list[BoundarySpec]:
        blocks = self._blocks(model)
        module_names = tuple(f"blocks.{idx}" for idx in range(len(blocks)))
        candidates = [
            BoundarySpec(
                name="whole_backbone",
                module_names=module_names,
                role=BoundaryRole.BACKBONE,
                granularity=BoundaryGranularity.WHOLE_BACKBONE,
                description="All JiT transformer backbone blocks.",
                is_quality_critical=True,
                is_speed_critical=True,
                prediction_role="xpred_backbone",
            )
        ]
        if module_names:
            split_a = max(1, len(module_names) // 3)
            split_b = max(split_a, (2 * len(module_names)) // 3)
            groups = (
                ("early_blocks", module_names[:split_a]),
                ("middle_blocks", module_names[split_a:split_b]),
                ("late_blocks", module_names[split_b:]),
            )
            for name, names in groups:
                if names:
                    candidates.append(
                        BoundarySpec(
                            name=name,
                            module_names=tuple(names),
                            role=BoundaryRole.BACKBONE,
                            granularity=BoundaryGranularity.MODULE_GROUP,
                            description="Diagnostic JiT block group; not a default Stage 4A preset.",
                            extra={"diagnostic": True},
                        )
                    )
        return candidates

    def default_boundary_set(self, model: Any, preset_name: str | None = None) -> BoundarySet:
        whole = self._candidate_by_name(model, "whole_backbone")
        return BoundarySet(
            name="jit_whole_backbone",
            boundaries=(whole,),
            description="Default JiT PixBFC boundary set: whole transformer backbone.",
        )

    def wrap_boundary_set(
        self,
        model: Any,
        boundary_set: BoundarySet,
        cache_state: Any,
        policy: Any,
    ) -> list[str]:
        layer_ids = self._layer_ids(boundary_set.module_names())
        return wrap_jit_blocks(model, cache_state, policy, layer_ids)

    def cache_proxy(self, x: torch.Tensor, t: torch.Tensor | float, cond: Any | None = None) -> torch.Tensor:
        return x

    def branch_mode(self) -> str:
        return "cond_uncond"

    @staticmethod
    def _blocks(model: Any) -> Any:
        net = getattr(model, "net", model)
        blocks = getattr(net, "blocks", None)
        if blocks is None:
            raise ValueError("Expected JiT model or net with a blocks attribute")
        return blocks

    def _candidate_by_name(self, model: Any, name: str) -> BoundarySpec:
        for candidate in self.list_boundary_candidates(model):
            if candidate.name == name:
                return candidate
        raise ValueError(f"JiT boundary candidate not found: {name}")

    @staticmethod
    def _layer_ids(module_names: tuple[str, ...]) -> list[int]:
        layer_ids: list[int] = []
        for module_name in module_names:
            prefix, _, suffix = module_name.partition(".")
            if prefix != "blocks" or not suffix.isdigit():
                raise ValueError(f"Unsupported JiT boundary module name: {module_name}")
            layer_ids.append(int(suffix))
        return sorted(dict.fromkeys(layer_ids))
