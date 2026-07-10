from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal


MethodType = Literal[
    "reference",
    "cache",
    "safe_cache",
    "dynamic_cache",
    "forecast_cache",
    "speculative_cache",
    "probe_cache",
    "reduced_steps",
]


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
    solver_stages: tuple[str, ...] | None = None
    active_window_warmup_refreshes: int = 0
    dynamic_cache_type: str | None = None
    dynamic_cache_threshold: float | None = None
    sea_beta: float = 2.0
    sea_proxy_downsample: int = 64
    taylorseer_interval: int | None = None
    taylorseer_max_order: int | None = None
    taylorseer_refresh_first_n_steps: int | None = None
    taylorseer_refresh_last_n_steps: int | None = None
    speca_max_order: int | None = None
    speca_first_full_steps: int | None = None
    speca_base_threshold: float | None = None
    speca_decay_rate: float | None = None
    speca_min_threshold: float | None = None
    speca_min_forecast_steps: int | None = None
    speca_max_forecast_steps: int | None = None
    speca_error_metric: str | None = None
    speca_branch_aggregation: str | None = None
    speca_verifier_module: str | None = None
    speca_min_history: int | None = None
    dicache_probe_depth: int | None = None
    dicache_reuse_threshold: float | None = None
    dicache_error_choice: str | None = None
    dicache_branch_aggregation: str | None = None
    dicache_ret_ratio: float | None = None
    dicache_force_last_step_full: bool | None = None
    dicache_dcta_enabled: bool | None = None
    dicache_gamma_min: float | None = None
    dicache_gamma_max: float | None = None
    dicache_eps: float | None = None
    dicache_max_stat_samples: int | None = None
    dicache_share_cfg_prefix: bool | None = None
    dicache_schedule_variant: str | None = None
    tags: tuple[str, ...] = ()
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
            method_name="safe_bfc_quality",
            method_type="safe_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "jit_safe_whole_backbone"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description=(
                "Safe-BFC quality preset: calibrated solver-perturbation safe map, "
                "no manual timestep window, no fixed interval."
            ),
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="safe_bfc_speed",
            method_type="safe_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "jit_safe_whole_backbone"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description=(
                "Safe-BFC speed preset: calibrated solver-perturbation safe map, "
                "no manual timestep window, no fixed interval."
            ),
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="taylorseer_style",
            method_type="forecast_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "jit_blocks"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            taylorseer_interval=4,
            taylorseer_max_order=4,
            taylorseer_refresh_first_n_steps=1,
            taylorseer_refresh_last_n_steps=0,
            description=(
                "Adapted TaylorSeer-style feature forecasting baseline using polynomial "
                "extrapolation over cached JiT block outputs."
            ),
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="taylorseer_quality_i3_o3",
            method_type="forecast_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "jit_blocks"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            taylorseer_interval=3,
            taylorseer_max_order=3,
            taylorseer_refresh_first_n_steps=1,
            taylorseer_refresh_last_n_steps=0,
            description=(
                "Adapted high-quality TaylorSeer-DiT-style configuration using interval 3 "
                "and max order 3."
            ),
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="speca_style",
            method_type="speculative_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "jit_blocks"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            speca_max_order=4,
            speca_first_full_steps=3,
            speca_base_threshold=0.1,
            speca_decay_rate=0.01,
            speca_min_threshold=0.01,
            speca_min_forecast_steps=2,
            speca_max_forecast_steps=5,
            speca_error_metric="relative_l1",
            speca_branch_aggregation="mean",
            speca_verifier_module="auto",
            speca_min_history=2,
            description=(
                "Adapted SpeCa-style forecast-then-verify baseline using the existing "
                "TaylorSeer JiT block-output predictor and lightweight last-block "
                "verification. Released-code-aligned default uses relative-L1 verification."
            ),
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="dicache_style",
            method_type="probe_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={
                "probe_depth": 1,
                "cache_unit": "jit_block_stack_residual",
            },
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            dicache_probe_depth=1,
            dicache_reuse_threshold=0.4,
            dicache_error_choice="delta_y",
            dicache_branch_aggregation="mean",
            dicache_ret_ratio=0.2,
            dicache_force_last_step_full=True,
            dicache_dcta_enabled=True,
            dicache_gamma_min=1.0,
            dicache_gamma_max=1.5,
            dicache_eps=1e-10,
            dicache_max_stat_samples=4096,
            dicache_share_cfg_prefix=False,
            dicache_schedule_variant="released_flux_compat",
            description=(
                "Adapted DiCache-style baseline for JiT using online shallow-block "
                "probing and first-order dynamic cache trajectory alignment over the "
                "JiT image-token block-stack residual. Defaults are inspired by the "
                "released FLUX example and are not yet validated for JiT."
            ),
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
        GenerationMethodPreset(
            model_name="JiT",
            method_name="teacache_style",
            method_type="dynamic_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            dynamic_cache_type="tea",
            dynamic_cache_threshold=0.10,
            sea_beta=2.0,
            sea_proxy_downsample=64,
            description="Adapted TeaCache-style raw accumulated-distance baseline using x_t proxy.",
        ),
        GenerationMethodPreset(
            model_name="JiT",
            method_name="seacache_style",
            method_type="dynamic_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            dynamic_cache_type="sea",
            dynamic_cache_threshold=0.06,
            sea_beta=2.0,
            sea_proxy_downsample=64,
            description="Adapted SeaCache-style SEA-filtered accumulated-distance baseline using x_t proxy.",
        ),
    ]
    return _tagged_method_map(methods)


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
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="teacache_style",
            method_type="dynamic_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units="all_candidates",
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            dynamic_cache_type="tea",
            dynamic_cache_threshold=0.10,
            sea_beta=2.0,
            sea_proxy_downsample=64,
            description="Adapted TeaCache-style raw accumulated-distance baseline using x_t proxy.",
        ),
        GenerationMethodPreset(
            model_name="DeCo",
            method_name="seacache_style",
            method_type="dynamic_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units="all_candidates",
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            dynamic_cache_type="sea",
            dynamic_cache_threshold=0.06,
            sea_beta=2.0,
            sea_proxy_downsample=64,
            description="Adapted SeaCache-style SEA-filtered accumulated-distance baseline using x_t proxy.",
        ),
    ]
    return _tagged_method_map(methods)


def get_pixelgen_stage4a_methods() -> dict[str, GenerationMethodPreset]:
    solver_stages = ("heun_predictor", "heun_corrector")
    methods = [
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="no_cache_50",
            method_type="reference",
            reference_steps=50,
            eval_steps=50,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="50-step no-cache PixelGen reference.",
        ),
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="bfc_quality_t02_08",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "pixelgen_jit_blocks"},
            deco_cache_units=None,
            active_t_min=0.2,
            active_t_max=0.8,
            cache_interval=2,
            solver_stages=solver_stages,
            description="BoundaryFlowCache quality preset: all PixelGen JiT-style blocks, interval 2, active t [0.2,0.8).",
        ),
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="bfc_speed_t02_10",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "pixelgen_jit_blocks"},
            deco_cache_units=None,
            active_t_min=0.2,
            active_t_max=1.0,
            cache_interval=2,
            solver_stages=solver_stages,
            description="BoundaryFlowCache speed preset: all PixelGen JiT-style blocks, interval 2, active t [0.2,1.0).",
        ),
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="bfc_speed_t02_09",
            method_type="cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "pixelgen_jit_blocks"},
            deco_cache_units=None,
            active_t_min=0.2,
            active_t_max=0.9,
            cache_interval=2,
            solver_stages=solver_stages,
            description="PixelGen safety ablation: all JiT-style blocks, interval 2, active t [0.2,0.9).",
        ),
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="reduced_steps_35",
            method_type="reduced_steps",
            reference_steps=50,
            eval_steps=35,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="35-step no-cache PixelGen reduced-step baseline.",
        ),
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="reduced_steps_30",
            method_type="reduced_steps",
            reference_steps=50,
            eval_steps=30,
            cache_preset=None,
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            description="30-step no-cache PixelGen reduced-step baseline.",
        ),
        GenerationMethodPreset(
            model_name="PixelGen",
            method_name="seacache_style",
            method_type="dynamic_cache",
            reference_steps=50,
            eval_steps=50,
            cache_preset={"cache_layers": "all", "cache_units": "pixelgen_jit_blocks"},
            deco_cache_units=None,
            active_t_min=None,
            active_t_max=None,
            cache_interval=None,
            solver_stages=solver_stages,
            dynamic_cache_type="sea",
            dynamic_cache_threshold=0.06,
            sea_beta=2.0,
            sea_proxy_downsample=64,
            description="Adapted SeaCache-style SEA-filtered accumulated-distance baseline using PixelGen x_t proxy.",
        ),
    ]
    return _tagged_method_map(methods)


def preset_to_json_dict(preset: GenerationMethodPreset) -> dict[str, Any]:
    return asdict(preset)


def _default_tags(preset: GenerationMethodPreset) -> tuple[str, ...]:
    if preset.method_name == "no_cache_50":
        return ("reference", "proxy_default", "final_50k")
    if preset.method_name == "teacache_style":
        return ("diagnostic", "legacy")
    if preset.method_name == "taylorseer_quality_i3_o3":
        return ("diagnostic",)
    if preset.method_type in {
        "safe_cache",
        "dynamic_cache",
        "forecast_cache",
        "speculative_cache",
        "probe_cache",
    }:
        return ("main_baseline", "proxy_default")
    if preset.method_type == "reduced_steps":
        return ("diagnostic", "proxy_default", "final_50k")
    if preset.method_type == "cache":
        return ("diagnostic", "legacy", "final_50k")
    return ("diagnostic",)


def _tagged_method_map(
    methods: list[GenerationMethodPreset],
) -> dict[str, GenerationMethodPreset]:
    tagged = [
        method if method.tags else replace(method, tags=_default_tags(method))
        for method in methods
    ]
    names = [method.method_name for method in tagged]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate method names for {tagged[0].model_name}: {names}")
    return {method.method_name: method for method in tagged}


def _methods_for_model(model_name: str) -> dict[str, GenerationMethodPreset]:
    normalized = model_name.strip().lower()
    if normalized == "jit":
        return get_jit_stage4a_methods()
    if normalized == "deco":
        return get_deco_stage4a_methods()
    if normalized == "pixelgen":
        return get_pixelgen_stage4a_methods()
    raise KeyError(f"unsupported model: {model_name}")


def list_methods_for_model(
    model_name: str,
    *,
    tags: set[str] | None = None,
) -> list[str]:
    methods = _methods_for_model(model_name)
    if not tags:
        return list(methods)
    return [
        name
        for name, preset in methods.items()
        if set(preset.tags) & set(tags)
    ]


def get_method_metadata(model_name: str, method_name: str) -> dict[str, Any]:
    return preset_to_json_dict(_methods_for_model(model_name)[method_name])


def method_supports_model(model_name: str, method_name: str) -> bool:
    try:
        return method_name in _methods_for_model(model_name)
    except KeyError:
        return False


def method_cli_overrides(model_name: str, method_name: str) -> list[str]:
    preset = _methods_for_model(model_name)[method_name]
    if preset.method_type == "probe_cache":
        return [
            "--dicache-probe-depth", str(preset.dicache_probe_depth),
            "--dicache-reuse-threshold", str(preset.dicache_reuse_threshold),
            "--dicache-error-choice", str(preset.dicache_error_choice),
            "--dicache-branch-aggregation", str(preset.dicache_branch_aggregation),
            "--dicache-ret-ratio", str(preset.dicache_ret_ratio),
            "--dicache-force-last-step-full" if preset.dicache_force_last_step_full else "--no-dicache-force-last-step-full",
            "--dicache-dcta" if preset.dicache_dcta_enabled else "--no-dicache-dcta",
            "--dicache-gamma-min", str(preset.dicache_gamma_min),
            "--dicache-gamma-max", str(preset.dicache_gamma_max),
            "--dicache-eps", str(preset.dicache_eps),
            "--dicache-max-stat-samples", str(preset.dicache_max_stat_samples),
            "--dicache-share-cfg-prefix" if preset.dicache_share_cfg_prefix else "--no-dicache-share-cfg-prefix",
            "--dicache-schedule-variant", preset.dicache_schedule_variant or "released_flux_compat",
        ]
    if preset.method_type == "dynamic_cache" and preset.dynamic_cache_threshold is not None:
        return [
            "--dynamic-cache-threshold", str(preset.dynamic_cache_threshold),
            "--sea-beta", str(preset.sea_beta),
            "--sea-proxy-downsample", str(preset.sea_proxy_downsample),
        ]
    if preset.method_type == "forecast_cache":
        return [
            "--taylorseer-interval", str(preset.taylorseer_interval),
            "--taylorseer-max-order", str(preset.taylorseer_max_order),
            "--taylorseer-refresh-first-n-steps", str(preset.taylorseer_refresh_first_n_steps),
            "--taylorseer-refresh-last-n-steps", str(preset.taylorseer_refresh_last_n_steps),
        ]
    if preset.method_type == "speculative_cache":
        return [
            "--speca-max-order", str(preset.speca_max_order),
            "--speca-first-full-steps", str(preset.speca_first_full_steps),
            "--speca-base-threshold", str(preset.speca_base_threshold),
            "--speca-decay-rate", str(preset.speca_decay_rate),
            "--speca-min-threshold", str(preset.speca_min_threshold),
            "--speca-min-forecast-steps", str(preset.speca_min_forecast_steps),
            "--speca-max-forecast-steps", str(preset.speca_max_forecast_steps),
            "--speca-error-metric", str(preset.speca_error_metric),
            "--speca-branch-aggregation", str(preset.speca_branch_aggregation),
            "--speca-verifier-module", str(preset.speca_verifier_module),
            "--speca-min-history", str(preset.speca_min_history),
        ]
    return []


def list_jit_stage4a_method_names() -> list[str]:
    return list(get_jit_stage4a_methods())


def list_deco_stage4a_method_names() -> list[str]:
    return list(get_deco_stage4a_methods())


def list_pixelgen_stage4a_method_names() -> list[str]:
    return list(get_pixelgen_stage4a_methods())
