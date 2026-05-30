from __future__ import annotations

from typing import Any


class CachePolicy:
    name = "base"

    def begin_step(self, *args: Any, **kwargs: Any) -> None:
        return None

    def should_compute(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def reuse(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class NoCachePolicy(CachePolicy):
    name = "none"

    def should_compute(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def reuse(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("NoCachePolicy never reuses cached values")

