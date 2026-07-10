import json

import pytest

from pfc.risk.io import strict_json_dumps
from pfc.risk.schema import JiTRiskRecord


def test_unavailable_action_serializes_null_not_nan():
    record = JiTRiskRecord(
        schema_version=1,
        record_type="jit_risk",
        run_id="test",
        shard_index=0,
        num_shards=1,
        sample_id=0,
        global_index=0,
        class_label=0,
        noise_seed=21,
        step_idx=0,
        t=0.0,
        t_next=0.02,
        dt=0.02,
        solver_stage="euler",
        cfg_enabled=False,
        cfg_scale_effective=1.0,
        boundary_plan="early",
        boundary_start=0,
        boundary_end=4,
        action="reuse_age_1",
        cache_age=1,
        action_ready=False,
        skip_reason="insufficient_history",
        risk_scaled_rms=None,
        risk_rel_l2=None,
        risk_low=None,
        risk_high=None,
        velocity_rel_l2=None,
        cond_xpred_rel_l2=None,
        uncond_xpred_rel_l2=None,
        diagnostic_action_latency_ms=None,
        signals={"state_delta_rel_l2": None},
        provenance_ref="shard_0/shard_meta.json",
    ).to_dict()
    encoded = strict_json_dumps(record)
    assert json.loads(encoded)["risk_scaled_rms"] is None
    assert "NaN" not in encoded and "Infinity" not in encoded


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_strict_json_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_dumps({"value": value})
