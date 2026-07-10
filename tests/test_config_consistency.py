from __future__ import annotations

from pathlib import Path

from scripts.check_config_consistency import ROOT, check_configs


def test_repository_configs_match_canonical_registry() -> None:
    report = check_configs(ROOT / "configs")
    assert report["valid"], report["errors"]


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("model: jit\nmodel: deco\n", encoding="utf-8")
    report = check_configs(tmp_path)
    assert not report["valid"]
    assert "duplicate YAML key" in report["errors"][0]
