from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BackboneCachePreset:
    name: str
    cache_layers: str
    cache_interval: int
    active_t_min: float | None
    active_t_max: float | None
    active_window_warmup_refreshes: int = 0
    description: str = ""


def get_jit_backbone_cache_presets() -> dict[str, BackboneCachePreset]:
    presets = [
        BackboneCachePreset(
            name="no_cache",
            cache_layers="none",
            cache_interval=1,
            active_t_min=None,
            active_t_max=None,
            description="50-step no-cache reference.",
        ),
        BackboneCachePreset(
            name="quality_t02_08",
            cache_layers="all",
            cache_interval=2,
            active_t_min=0.2,
            active_t_max=0.8,
            description="Quality-first Stage 2D BackboneCache preset.",
        ),
        BackboneCachePreset(
            name="speed_t02_10",
            cache_layers="all",
            cache_interval=2,
            active_t_min=0.2,
            active_t_max=1.0,
            description="Speed-quality Stage 2D BackboneCache preset.",
        ),
        BackboneCachePreset(
            name="quality_t01_08_w1",
            cache_layers="all",
            cache_interval=2,
            active_t_min=0.1,
            active_t_max=0.8,
            active_window_warmup_refreshes=1,
            description="First-hit delay variant with one forced reuse-candidate refresh.",
        ),
        BackboneCachePreset(
            name="quality_t01_08_w2",
            cache_layers="all",
            cache_interval=2,
            active_t_min=0.1,
            active_t_max=0.8,
            active_window_warmup_refreshes=2,
            description="First-hit delay variant with two forced reuse-candidate refreshes.",
        ),
        BackboneCachePreset(
            name="aggressive_i3_t02_08",
            cache_layers="all",
            cache_interval=3,
            active_t_min=0.2,
            active_t_max=0.8,
            description="More aggressive interval-3 Stage 2D comparison preset.",
        ),
    ]
    return {preset.name: preset for preset in presets}


def preset_to_policy_kwargs(preset: BackboneCachePreset) -> dict[str, Any]:
    return {
        "enabled": preset.name != "no_cache" and preset.cache_layers != "none",
        "interval": preset.cache_interval,
        "active_t_min": preset.active_t_min,
        "active_t_max": preset.active_t_max,
        "active_window_warmup_refreshes": preset.active_window_warmup_refreshes,
    }


def preset_to_config_dict(preset: BackboneCachePreset) -> dict[str, Any]:
    return asdict(preset)
