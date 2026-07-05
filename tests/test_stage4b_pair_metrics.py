from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from scripts.evaluate_stage4b_pair_metrics import evaluate_pair_metrics


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_rgb_png(path: Path, rgb: tuple[int, int, int], width: int = 8, height: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + bytes(rgb) * width
    payload = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(payload))
        + _png_chunk(b"IEND", b"")
    )


def test_pair_metrics_identical_images(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    method = tmp_path / "method"
    for idx in range(3):
        name = f"{idx:06d}.png"
        _write_rgb_png(ref / name, (32, 64, 96))
        _write_rgb_png(method / name, (32, 64, 96))

    out = tmp_path / "metrics.json"
    payload = evaluate_pair_metrics(
        reference_dir=ref,
        method_dir=method,
        out=out,
        metrics=["psnr", "ssim", "rel_l2"],
        batch_size=2,
        device_name="cpu",
        save_per_image=True,
    )

    assert payload["num_pairs"] == 3
    assert payload["summary"]["ssim"]["mean"] == 1.0
    assert payload["summary"]["rel_l2"]["mean"] == 0.0
    assert json.loads(out.read_text())["num_pairs"] == 3
    assert out.with_suffix(".per_image.csv").exists()


def test_pair_metrics_missing_filename_fails_strict(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    method = tmp_path / "method"
    _write_rgb_png(ref / "000000.png", (0, 0, 0))
    _write_rgb_png(ref / "000001.png", (0, 0, 0))
    _write_rgb_png(method / "000000.png", (0, 0, 0))

    out = tmp_path / "metrics.json"
    try:
        evaluate_pair_metrics(
            reference_dir=ref,
            method_dir=method,
            out=out,
            metrics=["psnr"],
            batch_size=1,
            device_name="cpu",
        )
    except RuntimeError as exc:
        assert "000001.png" in str(exc)
    else:
        raise AssertionError("strict filename mismatch should fail")


def test_pair_metrics_rejects_same_realpath(tmp_path: Path) -> None:
    ref = tmp_path / "same"
    _write_rgb_png(ref / "000000.png", (0, 0, 0))
    out = tmp_path / "metrics.json"
    try:
        evaluate_pair_metrics(
            reference_dir=ref,
            method_dir=ref,
            out=out,
            metrics=["psnr"],
            batch_size=1,
            device_name="cpu",
        )
    except RuntimeError as exc:
        assert "must be different" in str(exc)
    else:
        raise AssertionError("same realpath should fail")
