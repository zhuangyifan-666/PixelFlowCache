from __future__ import annotations

import json

from pfc.eval.method_presets import (
    get_pixelgen_stage4a_methods,
    list_pixelgen_stage4a_method_names,
    preset_to_json_dict,
)


def test_pixelgen_stage4a_presets_required_methods() -> None:
    methods = get_pixelgen_stage4a_methods()
    assert {
        "no_cache_50",
        "bfc_quality_t02_08",
        "bfc_speed_t02_10",
        "reduced_steps_35",
        "reduced_steps_30",
    }.issubset(methods)
    assert "bfc_speed_t02_09" in methods
    assert list_pixelgen_stage4a_method_names() == list(methods)
    assert methods["reduced_steps_35"].eval_steps == 35
    assert methods["reduced_steps_30"].eval_steps == 30
    json.dumps({name: preset_to_json_dict(preset) for name, preset in methods.items()})


def test_pixelgen_bfc_preset_windows_and_stages() -> None:
    methods = get_pixelgen_stage4a_methods()
    quality = methods["bfc_quality_t02_08"]
    speed = methods["bfc_speed_t02_10"]
    safety = methods["bfc_speed_t02_09"]
    assert quality.cache_interval == 2
    assert quality.active_t_min == 0.2
    assert quality.active_t_max == 0.8
    assert speed.cache_interval == 2
    assert speed.active_t_min == 0.2
    assert speed.active_t_max == 1.0
    assert safety.active_t_max == 0.9
    assert quality.solver_stages == ("heun_predictor", "heun_corrector")
    assert speed.solver_stages == ("heun_predictor", "heun_corrector")
    assert "all PixelGen JiT-style blocks" in quality.description
    assert "active t [0.2,1.0)" in speed.description
