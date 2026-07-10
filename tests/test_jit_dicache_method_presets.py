from __future__ import annotations

from pfc.eval.method_presets import get_jit_stage4a_methods, preset_to_json_dict


def test_dicache_style_preset_is_adapted_probe_cache() -> None:
    preset = get_jit_stage4a_methods()["dicache_style"]
    assert preset.method_type == "probe_cache"
    assert preset.eval_steps == 50
    assert preset.cache_preset == {"probe_depth": 1, "cache_unit": "jit_block_stack_residual"}
    assert preset.dicache_probe_depth == 1
    assert preset.dicache_reuse_threshold == 0.4
    assert preset.dicache_error_choice == "delta_y"
    assert preset.dicache_branch_aggregation == "mean"
    assert preset.dicache_ret_ratio == 0.2
    assert preset.dicache_force_last_step_full is True
    assert preset.dicache_dcta_enabled is True
    assert preset.dicache_gamma_min == 1.0
    assert preset.dicache_gamma_max == 1.5
    assert preset.dicache_share_cfg_prefix is False
    assert preset.dicache_schedule_variant == "released_flux_compat"
    assert not hasattr(preset, "dicache_" + "released_" + "code_" + "compat")
    assert "not yet validated for JiT" in preset.description
    assert preset_to_json_dict(preset)["method_type"] == "probe_cache"
