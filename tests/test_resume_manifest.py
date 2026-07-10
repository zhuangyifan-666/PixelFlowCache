from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pfc.eval.generation_io import (
    load_manifest_index_map,
    reconcile_resume_state,
    upsert_manifest_records,
    validate_output_indices,
    write_manifest_atomic,
)
from pfc.eval.label_schedule import ensure_label_schedule
import pfc.eval.label_schedule as label_schedule_module
from pfc.eval.sharding import compute_shard_indices, pending_indices


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_upsert_is_unique_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest_atomic(path, [{"index": 0, "label": 1}, {"index": 1, "label": 2}])
    upsert_manifest_records(path, [{"index": 1, "label": 9}, {"index": 2, "label": 3}])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert rows[1]["label"] == 9
    assert not list(tmp_path.glob(".manifest.jsonl.*.tmp"))


def test_duplicate_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"index": 1}\n{"index": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate manifest indices"):
        load_manifest_index_map(path)


def test_partial_and_zero_pending_resume(tmp_path: Path) -> None:
    (tmp_path / "000001.png").write_bytes(b"png")
    assert pending_indices([0, 1, 2], tmp_path, resume=True, save_png=True) == [0, 2]
    (tmp_path / "000000.png").write_bytes(b"png")
    (tmp_path / "000002.png").write_bytes(b"png")
    assert pending_indices([0, 1, 2], tmp_path, resume=True, save_png=True) == []


def test_sharded_resume_uses_global_indices(tmp_path: Path) -> None:
    indices = compute_shard_indices(10, 3, 1, "strided")
    assert indices == [1, 4, 7]
    (tmp_path / "000004.png").write_bytes(b"png")
    assert pending_indices(indices, tmp_path, resume=True, save_png=True) == [1, 7]


def test_orphan_png_is_reconstructed_without_regeneration(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "000001.png").write_bytes(b"png")
    manifest = tmp_path / "manifest.jsonl"

    result = reconcile_resume_state(
        [0, 1], [7, 8], image_dir, manifest, resume=True, save_png=True
    )

    assert result.complete_indices == [1]
    assert result.pending_indices == [0]
    assert result.orphan_png_indices == [1]
    row = load_manifest_index_map(manifest)[1]
    assert row["label"] == 8
    assert row["resume_status"] == "orphan_png_reconciled"


def test_stale_manifest_is_pending_and_replaced_after_generation(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_atomic(manifest, [{"index": 2, "label": 12, "path": str(image_dir / "000002.png")}])

    result = reconcile_resume_state(
        [2], [0, 0, 12], image_dir, manifest, resume=True, save_png=True
    )
    assert result.pending_indices == [2]
    assert result.stale_manifest_indices == [2]

    image_dir.mkdir()
    canonical = image_dir / "000002.png"
    canonical.write_bytes(b"new")
    upsert_manifest_records(manifest, [{"index": 2, "label": 12, "path": str(canonical)}])
    assert load_manifest_index_map(manifest)[2]["path"] == str(canonical)


def test_resume_rejects_wrong_label(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_atomic(manifest, [{"index": 0, "label": 99, "path": "wrong.png"}])
    with pytest.raises(ValueError, match="manifest label mismatch"):
        reconcile_resume_state([0], [3], tmp_path, manifest, resume=True, save_png=True)


def test_resume_canonicalizes_wrong_manifest_path_with_warning(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    expected = image_dir / "000000.png"
    expected.write_bytes(b"png")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_atomic(manifest, [{"index": 0, "label": 3, "path": "old/location.png"}])

    result = reconcile_resume_state([0], [3], image_dir, manifest, resume=True, save_png=True)

    assert result.complete_indices == [0]
    assert "canonicalized" in result.warnings[0]
    row = load_manifest_index_map(manifest)[0]
    assert row["path"] == str(expected)
    assert row["resume_path_reconciled"] is True


def test_zero_and_partial_pending_are_exact(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    for index in (0, 2):
        path = image_dir / f"{index:06d}.png"
        path.write_bytes(b"png")
    write_manifest_atomic(
        manifest,
        [
            {"index": 0, "label": 4, "path": str(image_dir / "000000.png")},
            {"index": 2, "label": 6, "path": str(image_dir / "000002.png")},
        ],
    )
    partial = reconcile_resume_state(
        [0, 1, 2], [4, 5, 6], image_dir, manifest, resume=True, save_png=True
    )
    assert partial.pending_indices == [1]

    (image_dir / "000001.png").write_bytes(b"png")
    upsert_manifest_records(
        manifest, [{"index": 1, "label": 5, "path": str(image_dir / "000001.png")}]
    )
    complete = reconcile_resume_state(
        [0, 1, 2], [4, 5, 6], image_dir, manifest, resume=True, save_png=True
    )
    assert complete.pending_indices == []
    assert complete.complete_indices == [0, 1, 2]


def test_sharded_reconciliation_uses_shard_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    indices = compute_shard_indices(8, 2, 1, "strided")
    (image_dir / "000003.png").write_bytes(b"png")
    manifest = tmp_path / "manifest_shard1.jsonl"
    result = reconcile_resume_state(
        indices, list(range(8)), image_dir, manifest, resume=True, save_png=True
    )
    assert result.orphan_png_indices == [3]
    assert result.pending_indices == [1, 5, 7]
    assert list(load_manifest_index_map(manifest)) == [3]


def test_label_schedule_uses_atomic_replace_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path]] = []
    original_replace = label_schedule_module.os.replace

    def spy_replace(source, target) -> None:
        calls.append((Path(source), Path(target)))
        original_replace(source, target)

    monkeypatch.setattr(label_schedule_module.os, "replace", spy_replace)
    paths = ensure_label_schedule([1, 2, 3], tmp_path)
    assert calls and calls[0][1] == paths["json"]
    assert not list(tmp_path.glob(".labels.json.*.tmp"))

    paths["json"].write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing label schedule is unreadable"):
        ensure_label_schedule([1, 2, 3], tmp_path)
    assert paths["json"].read_text(encoding="utf-8") == "{broken"


def test_resume_without_png_markers_is_rejected() -> None:
    with pytest.raises(ValueError, match="Resume requires PNG completion markers"):
        reconcile_resume_state([], [], ".", "manifest.jsonl", resume=True, save_png=False)


def test_output_validation_rejects_duplicate_numeric_and_non_numeric(tmp_path: Path) -> None:
    for name in ("000001.png", "1.png", "bad.png"):
        (tmp_path / name).write_bytes(b"png")
    report = validate_output_indices(tmp_path, [0, 1], strict=False)
    assert report["duplicate_indices"] == [1]
    assert report["missing_indices"] == [0]
    assert report["invalid_filenames"] == ["bad.png"]
    with pytest.raises(ValueError, match="validation failed"):
        validate_output_indices(tmp_path, [0, 1], strict=True)


@pytest.mark.parametrize(
    ("script", "checkpoint_flag"),
    [
        ("scripts/run_jit_stage4a_generate.py", "--jit-ckpt-dir"),
        ("scripts/run_deco_stage4a_generate.py", "--deco-ckpt"),
        ("scripts/run_pixelgen_stage4a_generate.py", "--pixelgen-ckpt"),
    ],
)
def test_npz_resume_is_rejected_before_model_load(
    tmp_path: Path, script: str, checkpoint_flag: str
) -> None:
    result = subprocess.run(
        [
            sys.executable, script, "--method", "no_cache_50", "--num-images", "2",
            "--batch-size", "1", "--output-root", str(tmp_path / "out"),
            checkpoint_flag, str(tmp_path / "missing"), "--save-npz", "--resume",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "NPZ resume is not supported because the in-memory tensor set may be incomplete." in result.stderr


@pytest.mark.parametrize(
    "script",
    [
        "scripts/run_jit_stage4a_generate.py",
        "scripts/run_deco_stage4a_generate.py",
        "scripts/run_pixelgen_stage4a_generate.py",
    ],
)
def test_all_generators_share_resume_without_png_rejection(tmp_path: Path, script: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--method",
            "no_cache_50",
            "--output-root",
            str(tmp_path),
            "--resume",
            "--no-save-png",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Resume requires PNG completion markers in the current implementation." in result.stderr
