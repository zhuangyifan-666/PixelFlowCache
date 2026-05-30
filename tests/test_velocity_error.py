from __future__ import annotations

import json
import math

import torch

from pfc.diagnostics.velocity_error import image_error_stats, tensor_error_stats


def test_tensor_error_stats_identical_tensors() -> None:
    x = torch.ones(2, 3)
    stats = tensor_error_stats(x, x, name="same")
    assert stats["rel_l2"] == 0.0
    assert stats["mse"] == 0.0
    assert math.isclose(stats["cosine"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    json.dumps(stats)


def test_tensor_error_stats_nonidentical_tensors() -> None:
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 0.0])
    stats = tensor_error_stats(a, b)
    assert stats["mae"] > 0
    assert stats["l2_diff"] > 0
    assert -1.0 <= stats["cosine"] <= 1.0


def test_image_error_stats_includes_psnr() -> None:
    x = torch.zeros(1, 3, 4, 4)
    stats = image_error_stats(x, x)
    assert "psnr" in stats
    assert math.isinf(stats["psnr"])
