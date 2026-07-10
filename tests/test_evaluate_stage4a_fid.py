from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scripts.evaluate_stage4a_fid as fid_eval


def test_auto_backend_never_selects_unimplemented_torchmetrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fid_eval.importlib.util,
        "find_spec",
        lambda name: object() if name == "torchmetrics" else None,
    )
    backend, message = fid_eval.select_backend("auto")
    assert backend is None
    assert "implemented" in message


def test_explicit_torchmetrics_is_clear_not_implemented() -> None:
    backend, _ = fid_eval.select_backend("torchmetrics")
    with pytest.raises(NotImplementedError, match="not implemented"):
        fid_eval.validate_backend_capabilities(
            backend, ["fid"], fid_stats=None, real_dir=None
        )


def test_backend_capability_checks_fail_before_compute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only FID"):
        fid_eval.validate_backend_capabilities(
            "cleanfid", ["fid", "is"], fid_stats=None, real_dir=tmp_path
        )
    with pytest.raises(ValueError, match="requires"):
        fid_eval.validate_backend_capabilities(
            "torch_fidelity", ["fid"], fid_stats=None, real_dir=None
        )


def test_fake_png_index_validation_is_strict(tmp_path: Path) -> None:
    (tmp_path / "000000.png").write_bytes(b"png")
    (tmp_path / "000002.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="missing_indices"):
        fid_eval.validate_fake_images(tmp_path, expected_images=3)
    (tmp_path / "000001.png").write_bytes(b"png")
    report = fid_eval.validate_fake_images(tmp_path, expected_images=3)
    assert report["valid"] is True


def test_fid_stats_validation_records_hash_and_size(tmp_path: Path) -> None:
    path = tmp_path / "stats.npz"
    np.savez(path, mu=np.zeros(2), sigma=np.eye(2))
    report = fid_eval.validate_fid_statistics(path)
    assert len(report["stats_sha256"]) == 64
    assert report["stats_size_bytes"] == path.stat().st_size
    assert report["stats_feature_dimension"] == 2
