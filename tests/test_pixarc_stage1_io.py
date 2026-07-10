import json

import pytest

from pfc.risk.io import AtomicSampleWriter, reconcile_sample_output, write_json_atomic


def test_atomic_sample_commit_and_resume_skip(tmp_path):
    run_dir = tmp_path / "run"
    with AtomicSampleWriter(run_dir, 3, "signature") as writer:
        target = writer.commit(
            risk_records=[{"value": 1.0}],
            correctness_records=[{"value": 0.0}],
            sample_summary={"history_items_after_clear": 0},
        )
    assert target.name == "sample_000003"
    done = json.loads((target / "DONE.json").read_text(encoding="utf-8"))
    assert done["risk_record_count"] == 1
    assert reconcile_sample_output(run_dir, 3, "signature", resume=True) == "skip"
    with pytest.raises(FileExistsError):
        reconcile_sample_output(run_dir, 3, "signature", resume=False)
    with pytest.raises(ValueError, match="different run configuration"):
        reconcile_sample_output(run_dir, 3, "other", resume=True)


def test_incomplete_sample_and_temporary_directory_are_reconciled(tmp_path):
    run_dir = tmp_path / "run"
    incomplete = run_dir / "samples" / "sample_000001"
    incomplete.mkdir(parents=True)
    (incomplete / "partial.json").write_text("{}", encoding="utf-8")
    temporary = incomplete.parent / ".sample_000001.interrupted.tmp"
    temporary.mkdir()
    assert reconcile_sample_output(run_dir, 1, "signature", resume=True) == "run"
    assert not incomplete.exists()
    assert not temporary.exists()


def test_atomic_json_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "nested" / "payload.json"
    write_json_atomic(target, {"finite": 1.0})
    assert json.loads(target.read_text(encoding="utf-8")) == {"finite": 1.0}
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []
