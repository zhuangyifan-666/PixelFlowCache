from __future__ import annotations

import json

from pfc.eval.method_presets import (
    get_deco_stage4a_methods,
    get_jit_stage4a_methods,
    preset_to_json_dict,
)


def test_stage4a_jit_presets_required_methods() -> None:
    methods = get_jit_stage4a_methods()
    assert set(methods) == {
        "no_cache_50",
        "bfc_quality_t02_08",
        "bfc_speed_t02_10",
        "safe_bfc_quality",
        "safe_bfc_speed",
        "teacache_style",
        "seacache_style",
        "taylorseer_style",
        "taylorseer_quality_i3_o3",
        "reduced_steps_35",
        "reduced_steps_30",
    }
    assert methods["reduced_steps_35"].eval_steps == 35
    assert methods["reduced_steps_30"].eval_steps == 30
    quality = methods["bfc_quality_t02_08"]
    assert quality.cache_interval == 2
    assert quality.active_t_min == 0.2
    assert quality.active_t_max == 0.8
    json.dumps({name: preset_to_json_dict(preset) for name, preset in methods.items()})


def test_stage4a_deco_presets_required_methods() -> None:
    methods = get_deco_stage4a_methods()
    assert set(methods) == {
        "no_cache_50",
        "bfc_all_candidates_t02_10",
        "bfc_backbone_plus_final_t02_10",
        "teacache_style",
        "seacache_style",
        "reduced_steps_35",
        "reduced_steps_30",
    }
    assert methods["bfc_all_candidates_t02_10"].deco_cache_units == "all_candidates"
    assert methods["bfc_backbone_plus_final_t02_10"].deco_cache_units == "backbone_plus_final"
    assert methods["bfc_all_candidates_t02_10"].active_t_min == 0.2
    assert methods["bfc_all_candidates_t02_10"].active_t_max == 1.0
    json.dumps({name: preset_to_json_dict(preset) for name, preset in methods.items()})
