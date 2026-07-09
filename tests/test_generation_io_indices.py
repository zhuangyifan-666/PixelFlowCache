from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pfc.eval.generation_io import save_image_batch_png


def _batch(size: int = 2) -> torch.Tensor:
    return torch.zeros(size, 3, 2, 2, dtype=torch.float32)


def test_save_image_batch_png_start_idx_saves_consecutive_filenames(tmp_path: Path) -> None:
    records = save_image_batch_png(_batch(2), [10, 11], 5, tmp_path)

    assert [record["index"] for record in records] == [5, 6]
    assert [record["label"] for record in records] == [10, 11]
    assert (tmp_path / "000005.png").is_file()
    assert (tmp_path / "000006.png").is_file()


def test_save_image_batch_png_indices_saves_global_filenames_and_records(tmp_path: Path) -> None:
    records = save_image_batch_png(_batch(3), [7, 8, 9], [0, 4, 8], tmp_path)

    assert [(record["index"], record["label"]) for record in records] == [(0, 7), (4, 8), (8, 9)]
    assert (tmp_path / "000000.png").is_file()
    assert (tmp_path / "000004.png").is_file()
    assert (tmp_path / "000008.png").is_file()


def test_save_image_batch_png_indices_length_must_match_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="indices must match batch size"):
        save_image_batch_png(_batch(2), [1, 2], [0], tmp_path)


def test_save_image_batch_png_labels_length_must_match_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labels must match batch size"):
        save_image_batch_png(_batch(2), [1], 0, tmp_path)
