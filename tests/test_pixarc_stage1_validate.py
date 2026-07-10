import json

import pytest

from pfc.risk.io import load_jsonl, write_json_atomic, write_jsonl
from pixarc_stage1_test_utils import make_stage1_run
from scripts.validate_jit_pixarc_stage1 import validate_stage1


def test_valid_synthetic_stage1_run_passes(tmp_path):
    run_dir = make_stage1_run(tmp_path / "valid")
    report = validate_stage1(run_dir, expected_images=2)
    assert report["status"] == "PASS"
    assert report["checks"]["positive_reuse_risk_found"]


def test_validator_blocks_age0_error(tmp_path):
    run_dir = make_stage1_run(tmp_path / "age0")
    path = run_dir / "samples" / "sample_000000" / "correctness_records.jsonl"
    rows = load_jsonl(path)
    replay = next(row for row in rows if row["check_name"] == "replay_age_0")
    replay.update({"allclose": False, "max_abs": 0.1, "relative_l2": 0.1})
    write_jsonl(path, rows)
    report = validate_stage1(run_dir, expected_images=2)
    assert report["status"] == "BLOCK"
    assert "replay_age0" in {issue["code"] for issue in report["issues"]}


def test_validator_blocks_future_leakage(tmp_path):
    run_dir = make_stage1_run(tmp_path / "future")
    path = run_dir / "samples" / "sample_000000" / "sample_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["future_leakage_count"] = 1
    write_json_atomic(path, summary)
    report = validate_stage1(run_dir, expected_images=2)
    assert report["status"] == "BLOCK"
    assert "future_leakage" in {issue["code"] for issue in report["issues"]}


def test_validator_blocks_missing_sample_and_degenerate_candidates(tmp_path):
    missing = make_stage1_run(tmp_path / "missing", num_images=2, sample_indices=[0])
    assert validate_stage1(missing, expected_images=2)["status"] == "BLOCK"
    degenerate = make_stage1_run(tmp_path / "degenerate", positive_reuse=False)
    report = validate_stage1(degenerate, expected_images=2)
    assert report["status"] == "BLOCK"
    assert "candidate_sanity" in {issue["code"] for issue in report["issues"]}


def test_validator_blocks_nonfinite_json(tmp_path):
    run_dir = make_stage1_run(tmp_path / "nonfinite")
    path = run_dir / "samples" / "sample_000000" / "risk_records.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["risk_scaled_rms"] = float("nan")
    rows[0] = json.dumps(payload, allow_nan=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = validate_stage1(run_dir, expected_images=2)
    assert report["status"] == "BLOCK"


@pytest.mark.parametrize("field", ["candidate_shape_match", "candidate_dtype_match"])
def test_validator_blocks_candidate_shape_or_dtype_mismatch(tmp_path, field):
    run_dir = make_stage1_run(tmp_path / field)
    path = run_dir / "samples" / "sample_000000" / "risk_records.jsonl"
    rows = load_jsonl(path)
    ready = next(row for row in rows if row["action"] == "reuse_age_1" and row["action_ready"])
    ready["signals"][field] = False
    write_jsonl(path, rows)
    report = validate_stage1(run_dir, expected_images=2)
    assert report["status"] == "BLOCK"
    assert "shape_dtype" in {issue["code"] for issue in report["issues"]}
