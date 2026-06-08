from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path

import scripts.evaluate_stage4a_fid as eval_fid


ROOT = Path(__file__).resolve().parents[1]


def test_fid_eval_dry_run_fake_image_dir(tmp_path: Path) -> None:
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    (fake_dir / "000000.png").write_bytes(b"not decoded during dry run")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_stage4a_fid.py",
            "--fake-dir",
            str(fake_dir),
            "--out",
            str(tmp_path / "fid_results.json"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "dry_run" in result.stdout
    assert "num_fake_images" in result.stdout


def test_fid_stats_path_does_not_require_torch_fidelity_input2(monkeypatch, tmp_path: Path) -> None:
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    stats_path = tmp_path / "stats.npz"
    stats_path.write_bytes(b"mocked")
    seen: dict[str, object] = {}

    def create_feature_extractor(name, feature_layers, **kwargs):
        seen["feature_extractor"] = name
        seen["feature_layers"] = tuple(feature_layers)
        seen["kwargs"] = kwargs
        return object()

    def extract_featuresdict_from_input_id_cached(input_id, feat_extractor, **kwargs):
        assert input_id == 1
        assert "input2" not in kwargs
        return {"2048": "fid_features", "logits_unbiased": "isc_features"}

    def fid_featuresdict_to_statistics(featuresdict, feat_layer_name):
        assert feat_layer_name == "2048"
        return {"mu": "fake_mu", "sigma": "fake_sigma"}

    def fid_statistics_to_metric(fake_stats, ref_stats, verbose):
        assert ref_stats == {"mu": "ref_mu", "sigma": "ref_sigma"}
        return {"frechet_inception_distance": 12.5}

    def isc_featuresdict_to_metric(featuresdict, feat_layer_name, **kwargs):
        assert feat_layer_name == "logits_unbiased"
        return {"inception_score_mean": 2.0, "inception_score_std": 0.1}

    utils_module = types.ModuleType("torch_fidelity.utils")
    utils_module.create_feature_extractor = create_feature_extractor
    utils_module.extract_featuresdict_from_input_id_cached = extract_featuresdict_from_input_id_cached
    fid_module = types.ModuleType("torch_fidelity.metric_fid")
    fid_module.fid_featuresdict_to_statistics = fid_featuresdict_to_statistics
    fid_module.fid_statistics_to_metric = fid_statistics_to_metric
    isc_module = types.ModuleType("torch_fidelity.metric_isc")
    isc_module.isc_featuresdict_to_metric = isc_featuresdict_to_metric

    monkeypatch.setitem(sys.modules, "torch_fidelity.utils", utils_module)
    monkeypatch.setitem(sys.modules, "torch_fidelity.metric_fid", fid_module)
    monkeypatch.setitem(sys.modules, "torch_fidelity.metric_isc", isc_module)
    monkeypatch.setattr(eval_fid, "_load_fid_statistics", lambda path: {"mu": "ref_mu", "sigma": "ref_sigma"})

    metrics = eval_fid._compute_with_torch_fidelity(
        argparse.Namespace(
            fake_dir=fake_dir,
            fid_stats=stats_path,
            real_dir=None,
            device="cpu",
            batch_size=4,
            metrics=["fid", "is"],
        )
    )

    assert metrics == {
        "frechet_inception_distance": 12.5,
        "inception_score_mean": 2.0,
        "inception_score_std": 0.1,
    }
    assert seen["feature_layers"] == ("logits_unbiased", "2048")
