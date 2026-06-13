from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PredictionType(str, Enum):
    XPRED = "xpred"
    VPRED = "vpred"
    EPSPRED = "epspred"
    SCORE = "score"
    UNKNOWN = "unknown"


class BoundaryRole(str, Enum):
    BACKBONE = "backbone"
    DECODER = "decoder"
    FINAL_OUTPUT = "final_output"
    PATCH_PATHWAY = "patch_pathway"
    PIXEL_PATHWAY = "pixel_pathway"
    WHOLE_MODEL = "whole_model"
    CUSTOM = "custom"


class BoundaryGranularity(str, Enum):
    MODULE = "module"
    MODULE_GROUP = "module_group"
    WHOLE_BACKBONE = "whole_backbone"
    OUTPUT_BOUNDARY = "output_boundary"


@dataclass(frozen=True)
class BoundarySpec:
    name: str
    module_names: tuple[str, ...]
    role: BoundaryRole
    granularity: BoundaryGranularity
    description: str = ""
    is_quality_critical: bool = False
    is_speed_critical: bool = False
    prediction_role: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module_names": list(self.module_names),
            "role": self.role.value,
            "granularity": self.granularity.value,
            "description": self.description,
            "is_quality_critical": self.is_quality_critical,
            "is_speed_critical": self.is_speed_critical,
            "prediction_role": self.prediction_role,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundarySpec":
        return cls(
            name=str(data["name"]),
            module_names=tuple(str(item) for item in data.get("module_names", ())),
            role=BoundaryRole(data["role"]),
            granularity=BoundaryGranularity(data["granularity"]),
            description=str(data.get("description", "")),
            is_quality_critical=bool(data.get("is_quality_critical", False)),
            is_speed_critical=bool(data.get("is_speed_critical", False)),
            prediction_role=data.get("prediction_role"),
            extra=dict(data.get("extra") or {}),
        )

    def short_name(self) -> str:
        return self.name.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class BoundarySet:
    name: str
    boundaries: tuple[BoundarySpec, ...]
    description: str = ""

    def module_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for boundary in self.boundaries:
            for module_name in boundary.module_names:
                if module_name not in seen:
                    names.append(module_name)
                    seen.add(module_name)
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "module_names": list(self.module_names()),
        }

    def get(self, name: str) -> BoundarySpec | None:
        for boundary in self.boundaries:
            if boundary.name == name:
                return boundary
        return None

    def contains_module(self, module_name: str) -> bool:
        return module_name in set(self.module_names())
