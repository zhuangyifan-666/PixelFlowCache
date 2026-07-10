from __future__ import annotations

import pytest

from scripts.run_stage4a_full_eval_plan import build_parser, build_plan


def _plan(*extra: str) -> str:
    args = build_parser().parse_args(["--models", "jit", "--num-images", "1000", *extra])
    return "\n".join(build_plan(args))


def test_quality_and_speed_commands_receive_distinct_safe_maps() -> None:
    plan = _plan(
        "--methods", "safe_bfc_quality,safe_bfc_speed",
        "--safe-map-quality", "quality.json",
        "--safe-map-speed", "speed.json",
    )
    quality = next(line for line in plan.splitlines() if "--method safe_bfc_quality" in line)
    speed = next(line for line in plan.splitlines() if "--method safe_bfc_speed" in line)
    assert "--safe-map quality.json" in quality
    assert "--safe-map speed.json" in speed


@pytest.mark.parametrize("method", ["safe_bfc_quality", "safe_bfc_speed"])
def test_selected_safe_method_requires_matching_path(method: str) -> None:
    with pytest.raises(ValueError, match="require safe maps"):
        _plan("--methods", method)


def test_non_safe_method_needs_no_map_and_filter_excludes_unselected_methods() -> None:
    plan = _plan("--methods", "no_cache_50,seacache_style")
    assert "--method no_cache_50" in plan
    assert "--method seacache_style" in plan
    assert "--method safe_bfc_quality" not in plan
    assert "--safe-map" not in plan
    assert "--resume" not in plan
