from __future__ import annotations

import json

from pfc.cache.backbone_cache_presets import (
    get_jit_backbone_cache_presets,
    preset_to_config_dict,
    preset_to_policy_kwargs,
)


def test_required_backbone_cache_presets_exist() -> None:
    presets = get_jit_backbone_cache_presets()
    assert {
        "no_cache",
        "quality_t02_08",
        "speed_t02_10",
        "quality_t01_08_w1",
        "quality_t01_08_w2",
        "aggressive_i3_t02_08",
    }.issubset(presets)


def test_backbone_cache_presets_are_json_serializable() -> None:
    presets = get_jit_backbone_cache_presets()
    json.dumps({name: preset_to_config_dict(preset) for name, preset in presets.items()})


def test_no_cache_preset_disables_policy() -> None:
    preset = get_jit_backbone_cache_presets()["no_cache"]
    kwargs = preset_to_policy_kwargs(preset)
    assert kwargs["enabled"] is False
    assert preset.cache_layers == "none"
    assert preset.cache_interval == 1


def test_stage3a_required_preset_values() -> None:
    presets = get_jit_backbone_cache_presets()
    assert presets["speed_t02_10"].active_t_max == 1.0
    assert presets["quality_t01_08_w2"].active_window_warmup_refreshes == 2
