from __future__ import annotations

from pfc.eval.method_presets import get_jit_stage4a_methods


def test_jit_speca_method_preset_matches_adapted_baseline_defaults() -> None:
    preset = get_jit_stage4a_methods()["speca_style"]
    assert preset.model_name == "JiT"
    assert preset.method_type == "speculative_cache"
    assert preset.reference_steps == 50
    assert preset.eval_steps == 50
    assert preset.cache_preset == {"cache_layers": "all", "cache_units": "jit_blocks"}
    assert preset.speca_max_order == 4
    assert preset.speca_first_full_steps == 3
    assert preset.speca_base_threshold == 0.1
    assert preset.speca_decay_rate == 0.01
    assert preset.speca_min_threshold == 0.01
    assert preset.speca_min_forecast_steps == 2
    assert preset.speca_max_forecast_steps == 5
    assert preset.speca_error_metric == "relative_l1"
    assert preset.speca_branch_aggregation == "mean"
    assert preset.speca_verifier_module == "auto"
    assert preset.speca_min_history == 2
    assert "Adapted SpeCa-style" in preset.description
