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
        "seacache_style",
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


def test_pixelgen_seacache_style_preset() -> None:
    methods = get_pixelgen_stage4a_methods()
    seacache = methods["seacache_style"]
    assert seacache.model_name == "PixelGen"
    assert seacache.method_type == "dynamic_cache"
    assert seacache.reference_steps == 50
    assert seacache.eval_steps == 50
    assert seacache.cache_preset == {"cache_layers": "all", "cache_units": "pixelgen_jit_blocks"}
    assert seacache.deco_cache_units is None
    assert seacache.active_t_min is None
    assert seacache.active_t_max is None
    assert seacache.cache_interval is None
    assert seacache.solver_stages == ("heun_predictor", "heun_corrector")
    assert seacache.dynamic_cache_type == "sea"
    assert seacache.dynamic_cache_threshold == 0.06
    assert seacache.sea_beta == 2.0
    assert seacache.sea_proxy_downsample == 64
    assert "PixelGen x_t proxy" in seacache.description
