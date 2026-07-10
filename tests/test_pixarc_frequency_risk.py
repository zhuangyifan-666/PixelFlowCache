import math

import torch

from pfc.risk.frequency import RadialFrequencyRisk


def test_frequency_risk_is_finite_and_reuses_masks():
    metric = RadialFrequencyRisk(0.15, 0.45)
    delta = torch.randn(1, 2, 8, 8)
    fresh = torch.randn(1, 2, 8, 8)
    first = metric.risks(delta, fresh)
    second = metric.risks(delta * 2, fresh)
    assert all(math.isfinite(value) for value in first.values())
    assert all(math.isfinite(value) for value in second.values())
    assert metric.cache_size == 1
    low, middle, high = metric.split(delta)
    assert torch.allclose(low + middle + high, delta, atol=1e-5, rtol=1e-5)
