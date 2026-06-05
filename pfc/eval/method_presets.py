from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


MethodType = Literal["reference", "cache", "reduced_steps"]


@dataclass(frozen=True)
class GenerationMethodPreset:
    model_name: str
    method_name: str
    method_type: MethodType
    reference_steps: int
    eval_steps: int
    cache_preset: dict[str, Any] | None
    deco_cache_units: str | None
    active_t_min: float | None
    active_t_max: float | None
    cache_interval: int | None
    active_window_warmup_refreshes: int = 0
    description: str = ""


def get_jit_stage4a_methods() -> dict[str, GenerationMethodPreset]:
    methods = [
        GenerationMethodPreset(
            model_name="JiT",
            method_name="no_cache_50",
            method_type="reference",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="50-step no-cache JiT reference.",
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="bfc_quality_t02_08",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all"},
            deco_cache_units=None,
            active_t_min=0.2,
            active_t_max=0.8,
            cache_interval=2,
            description="BoundaryFlowCache quality preset: all JiT blocks, interval 2, active t [0.2,0.8).",
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="bfc_speed_t02_10",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all"},
            deco_cache_units=None,
            active_t_min=0.2,
            active_t_max=1.0,
            cache_interval=2,
            description="BoundaryFlowCache speed preset: all JiT blocks, interval 2, active t [0.2,1.0).",
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="reduced_steps_35",
            method_type="reduced_steps",
            reference_steps=50,
            eval_steps=35,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="35-step no-cache JiT reduced-step baseline.",
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="reduced_steps_30",
            method_type="reduced_steps",
            reference_steps=50,
            eval_steps=30,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="30-step no-cache JiT reduced-step baseline.",
        ),
    ]
    return {method.method_name: method for method in methods}


def get_deco_stage4a_methods() -> dict[str, GenerationMethodPreset]:
    methods = [
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="no_cache_50",
            method_type="reference",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="50-step no-cache DeCo reference.",
        ),
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="bfc_all_candidates_t02_10",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units="all_candidates",
            active_t_min=0.2,
            active_t_max=1.0,
            cache_interval=2,
            description="BoundaryFlowCache DeCo all-candidates preset, interval 2, active t [0.2,1.0).",
        ),
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="bfc_backbone_plus_final_t02_10",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units="backbone_plus_final",
            active_t_min=0.2,
            active_t_max=1.0,
            cache_interval=2,
            description="BoundaryFlowCache DeCo backbone-plus-final preset, interval 2, active t [0.2,1.0).",
        ),
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="reduced_steps_35",
            method_type="reduced_steps",
            reference_steps=50,
            eval_steps=35,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="35-step no-cache DeCo reduced-step baseline.",
        ),
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="reduced_steps_30",
            method_type="reduced_steps",
            reference_steps=50,
            eval_steps=30,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="30-step no-cache DeCo reduced-step baseline.",
        ),
    ]
    return {method.method_name: method for method in methods}


def preset_to_json_dict(preset: GenerationMethodPreset) -> dict[str, Any]:
    return asdict(preset)


def list_jit_stage4a_method_names() -> list[str]:
    return list(get_jit_stage4a_methods())


def list_deco_stage4a_method_names() -> list[str]:
    return list(get_deco_stage4a_methods())

