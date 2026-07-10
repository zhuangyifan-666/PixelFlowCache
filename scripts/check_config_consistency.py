#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import get_method_metadata, method_supports_model  # noqa: E402


CANONICAL_CHECKPOINTS = {
    "jit": "ckpts/JiT/JiT-B-16-256",
    "deco": "ckpts/DeCo/DeCo_XL.ckpt",
    "pixelgen": "ckpts/PixelGen/PixelGen_XL_160ep.ckpt",
}
TYPE_ALIASES = {
    "reference": "reference",
    "boundary_flow_cache": "cache",
    "reduced_steps": "reduced_steps",
    "dynamic_cache": "dynamic_cache",
}


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _model_name(payload: dict[str, Any], path: Path) -> str | None:
    model = payload.get("model")
    if isinstance(model, str):
        value = model.lower()
    elif isinstance(model, dict):
        value = str(model.get("family", model.get("name", ""))).lower()
    else:
        value = path.stem.lower()
    for candidate in ("pixelgen", "deco", "jit"):
        if candidate in value or candidate in path.stem.lower():
            return candidate
    return None


def _method_items(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    methods = payload.get("methods")
    if isinstance(methods, dict):
        return [(str(name), config if isinstance(config, dict) else {}) for name, config in methods.items()]
    if isinstance(methods, list):
        return [(str(name), {}) for name in methods]
    if isinstance(payload.get("method"), str):
        return [(str(payload["method"]), payload)]
    return []


def check_configs(config_dir: Path = ROOT / "configs") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        except Exception as exc:
            errors.append(f"{path.name}: YAML parse/duplicate-key failure: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.name}: top-level YAML value must be a mapping")
            continue
        checked.append(path.name)
        model = _model_name(payload, path)
        if model is None:
            errors.append(f"{path.name}: cannot infer model")
            continue
        if path.name in {"jit_boundaryflowcache.yaml", "deco_boundaryflowcache.yaml"}:
            canonical = payload.get("canonical_config")
            if payload.get("deprecated") is not True or not canonical:
                errors.append(f"{path.name}: obsolete config must be deprecated and point to canonical_config")
            elif not (ROOT / str(canonical)).is_file():
                errors.append(f"{path.name}: canonical_config does not exist: {canonical}")
        for method, config in _method_items(payload):
            if not method_supports_model(model, method):
                errors.append(f"{path.name}: method {method!r} is absent from the {model} registry")
                continue
            preset = get_method_metadata(model, method)
            configured_steps = config.get("steps")
            if configured_steps is not None and int(configured_steps) != int(preset["eval_steps"]):
                errors.append(
                    f"{path.name}:{method}: steps={configured_steps} differs from registry={preset['eval_steps']}"
                )
            configured_type = config.get("type")
            if configured_type in TYPE_ALIASES:
                registry_type = str(preset["method_type"])
                if TYPE_ALIASES[configured_type] != registry_type:
                    errors.append(
                        f"{path.name}:{method}: type={configured_type} differs from registry={registry_type}"
                    )
            if method == "seacache_style":
                threshold = config.get(
                    "default_threshold",
                    payload.get("dynamic_cache", {}).get("dynamic_cache_threshold")
                    if isinstance(payload.get("dynamic_cache"), dict)
                    else None,
                )
                if threshold is not None and float(threshold) != 0.06:
                    errors.append(f"{path.name}:{method}: threshold must be 0.06, got {threshold}")
        path_config = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
        configured_checkpoint = (
            path_config.get("checkpoint_dir")
            or path_config.get("checkpoint")
            or payload.get("checkpoint")
        )
        if configured_checkpoint is not None and str(configured_checkpoint) != CANONICAL_CHECKPOINTS[model]:
            errors.append(
                f"{path.name}: checkpoint={configured_checkpoint!r} differs from canonical={CANONICAL_CHECKPOINTS[model]!r}"
            )
    return {
        "valid": not errors,
        "checked_files": checked,
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check YAML configs against the canonical method registry.")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = check_configs(args.config_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        for path in report["checked_files"]:
            print(f"OK parsed: {path}")
        for warning in report["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        for error in report["errors"]:
            print(f"error: {error}", file=sys.stderr)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
