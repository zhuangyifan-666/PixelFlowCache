from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import scripts.preflight_experiments as preflight
from scripts.preflight_experiments import BLOCK, PASS, WARN, _safe_map_summary, _write_reports


def test_safe_map_all_false_is_blocker(tmp_path: Path) -> None:
    path = tmp_path / "safe.json"
    path.write_text(json.dumps({"safe_map": {"0": {"1": False}}}), encoding="utf-8")
    status, message, details = _safe_map_summary(path)
    assert status == BLOCK
    assert details["safe_true"] == 0
    assert "no reusable" in message


def test_safe_map_nonempty_passes(tmp_path: Path) -> None:
    path = tmp_path / "safe.json"
    path.write_text(json.dumps({"safe_map": {"0": {"1": True, "2": False}}}), encoding="utf-8")
    status, _, details = _safe_map_summary(path)
    assert status == PASS
    assert details["safe_total"] == 2
    assert details["density"] == 0.5


def test_preflight_writes_json_and_markdown(tmp_path: Path) -> None:
    out = tmp_path / "preflight_report.json"
    payload = {
        "overall_status": "WARN",
        "checks": [{"category": "gpu", "name": "count", "status": "WARN", "message": "offline"}],
    }
    _write_reports(out, payload)
    assert json.loads(out.read_text(encoding="utf-8"))["overall_status"] == "WARN"
    assert out.with_suffix(".md").is_file()


def test_required_gpus_zero_never_blocks_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "collect_gpu_provenance",
        lambda: {"cuda_available": False, "gpu_count": 0, "gpu_names": [], "gpus": []},
    )
    monkeypatch.setattr(preflight, "_nvidia_smi_inventory", lambda: {"available": False})
    checks = []
    preflight.check_gpu(checks, 0)
    assert checks[0].status == PASS
    assert checks[0].message == "GPU validation disabled by required_gpus=0"


def test_third_party_checks_only_requested_models_and_pixeldit_is_strict_when_requested() -> None:
    checks = []
    preflight.check_third_party(checks, ["jit"])
    assert [check.name for check in checks] == ["jit"]

    pixeldit = []
    preflight.check_third_party(pixeldit, ["pixeldit"])
    assert [check.name for check in pixeldit] == ["pixeldit"]
    assert pixeldit[0].status in {PASS, BLOCK}


def test_conda_environment_probe_success_and_failure(monkeypatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "conda" if name == "conda" else None)

    success_result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {"python": "3.11", "torch": "2.5", "cuda_available": False, "gpu_count": 0, "missing": []}
        ),
        stderr="",
    )
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **kwargs: success_result)
    checks = []
    preflight.check_conda_environments(
        checks, Namespace(strict=True, env_jit="jit"), ["jit"]
    )
    assert checks[0].status == PASS
    command_text = " ".join(checks[0].details["command"])
    assert "load_jit_model" not in command_text
    assert "download" not in command_text

    failure_result = SimpleNamespace(returncode=1, stdout="", stderr="missing env")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **kwargs: failure_result)
    strict_checks = []
    preflight.check_conda_environments(
        strict_checks, Namespace(strict=True, env_jit="missing"), ["jit"]
    )
    assert strict_checks[0].status == BLOCK
    relaxed_checks = []
    preflight.check_conda_environments(
        relaxed_checks, Namespace(strict=False, env_jit="missing"), ["jit"]
    )
    assert relaxed_checks[0].status == WARN


def test_safe_map_density_uses_only_explicit_safe_tree(tmp_path: Path) -> None:
    path = tmp_path / "safe.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "JiT",
                "steps": 50,
                "selected_modules": ["blocks.0"],
                "max_age": 1,
                "unrelated": {"enabled": True},
                "safe": {"euler": {"global": {"backbone": {"1": {"1": False}}}}},
            }
        ),
        encoding="utf-8",
    )
    status, _, details = _safe_map_summary(path)
    assert status == BLOCK
    assert details["safe_total"] == 1
    assert details["safe_true"] == 0
    assert details["schema_key"] == "safe"


def test_config_consistency_report_is_integrated(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "check_config_files",
        lambda path: {"valid": False, "checked_files": ["x.yaml"], "errors": ["drift"], "warnings": []},
    )
    args = Namespace(
        jit_ckpt_dir=Path("ckpts/JiT/JiT-B-16-256"),
        deco_ckpt=Path("ckpts/DeCo/DeCo_XL.ckpt"),
        pixelgen_ckpt=Path("ckpts/PixelGen/PixelGen_XL_160ep.ckpt"),
    )
    checks = []
    preflight.check_configs(checks, ["jit"], ["no_cache_50"], args)
    consistency = next(check for check in checks if check.name == "config_consistency")
    assert consistency.status == BLOCK
    assert consistency.details["errors"] == ["drift"]


def test_torch_fidelity_capability_accepts_fake_compatible_modules(monkeypatch) -> None:
    modules = {
        "torch_fidelity.metric_fid": SimpleNamespace(
            fid_featuresdict_to_statistics=lambda: None,
            fid_statistics_to_metric=lambda: None,
        ),
        "torch_fidelity.metric_isc": SimpleNamespace(isc_featuresdict_to_metric=lambda: None),
        "torch_fidelity.utils": SimpleNamespace(
            create_feature_extractor=lambda: None,
            extract_featuresdict_from_input_id_cached=lambda: None,
        ),
    }
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda name: "0.3.0")
    status, message, details = preflight._torch_fidelity_capability()
    assert status == PASS
    assert "0.3.0" in message
    assert details["missing_internal_apis"] == []
