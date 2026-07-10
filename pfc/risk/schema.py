from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


PIXARC_STAGE1_SCHEMA_VERSION = 1


def ensure_strict_json_value(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_strict_json_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_strict_json_value(item, f"{path}[{index}]")


@dataclass
class JiTRiskRecord:
    schema_version: int
    record_type: str
    run_id: str
    shard_index: int
    num_shards: int
    sample_id: int
    global_index: int
    class_label: int
    noise_seed: int
    step_idx: int
    t: float
    t_next: float
    dt: float
    solver_stage: str
    cfg_enabled: bool
    cfg_scale_effective: float
    boundary_plan: str
    boundary_start: int | None
    boundary_end: int | None
    action: str
    cache_age: int | None
    action_ready: bool
    skip_reason: str | None
    risk_scaled_rms: float | None
    risk_rel_l2: float | None
    risk_low: float | None
    risk_high: float | None
    velocity_rel_l2: float | None
    cond_xpred_rel_l2: float | None
    uncond_xpred_rel_l2: float | None
    diagnostic_action_latency_ms: float | None
    signals: dict[str, Any]
    provenance_ref: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        ensure_strict_json_value(payload)
        json.dumps(payload, allow_nan=False)
        return payload


@dataclass
class JiTCorrectnessRecord:
    schema_version: int
    record_type: str
    run_id: str
    shard_index: int
    num_shards: int
    sample_id: int
    global_index: int
    step_idx: int
    branch: str
    boundary_plan: str | None
    check_name: str
    max_abs: float
    mean_abs: float
    relative_l2: float
    allclose: bool
    shape_match: bool
    dtype_match: bool
    future_leakage_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    provenance_ref: str = "run_meta.json"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        ensure_strict_json_value(payload)
        json.dumps(payload, allow_nan=False)
        return payload
