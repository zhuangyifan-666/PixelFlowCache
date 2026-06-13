from __future__ import annotations

from typing import TypeVar

from pfc.core.model_adapter import PixelDiffusionModelAdapter


AdapterT = TypeVar("AdapterT", bound=type[PixelDiffusionModelAdapter])

_ADAPTERS: dict[str, type[PixelDiffusionModelAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[PixelDiffusionModelAdapter]) -> type[PixelDiffusionModelAdapter]:
    key = name.strip().lower()
    if not key:
        raise ValueError("adapter name must not be empty")
    _ADAPTERS[key] = adapter_cls
    return adapter_cls


def get_adapter(name: str) -> type[PixelDiffusionModelAdapter]:
    key = name.strip().lower()
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown PixBFC adapter: {name}. Available: {available_adapters()}") from exc


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)
