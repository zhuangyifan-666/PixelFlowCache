from __future__ import annotations

from pfc.eval.method_presets import get_jit_stage4a_methods


def test_jit_safe_bfc_presets_are_safe_cache_methods() -> None:
    methods = get_jit_stage4a_methods()
    for name in ("safe_bfc_quality", "safe_bfc_speed"):
        preset = methods[name]
        assert preset.method_type == "safe_cache"
        assert preset.cache_interval is None
        assert preset.active_t_min is None
        assert preset.active_t_max is None
        assert preset.eval_steps == 50
        assert preset.cache_preset == {"cache_layers": "all", "cache_units": "jit_safe_whole_backbone"}
