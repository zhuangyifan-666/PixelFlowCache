from __future__ import annotations

from pfc.eval.method_presets import get_jit_stage4a_methods


def test_jit_taylorseer_presets_exist() -> None:
    methods = get_jit_stage4a_methods()
    preset = methods["taylorseer_style"]
    assert preset.method_type == "forecast_cache"
    assert preset.taylorseer_interval == 4
    assert preset.taylorseer_max_order == 4
    assert preset.cache_preset == {"cache_layers": "all", "cache_units": "jit_blocks"}

    quality = methods["taylorseer_quality_i3_o3"]
    assert quality.method_type == "forecast_cache"
    assert quality.taylorseer_interval == 3
    assert quality.taylorseer_max_order == 3
