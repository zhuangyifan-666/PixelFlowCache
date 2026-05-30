from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from pfc.profiling.feature_recorder import FeatureRecorder
from pfc.profiling.jsonl import JsonlWriter


def test_feature_recorder_writes_delta_and_removes_hooks(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    path = tmp_path / "feature_stats.jsonl"
    writer = JsonlWriter(path)
    recorder = FeatureRecorder(
        module_filter=lambda name, module: name == "0",
        writer=writer,
        model_name="tiny",
    )
    recorder.attach(model)
    recorder.set_context(0, 0.0)
    model(torch.ones(1, 4))
    recorder.set_context(1, 0.1)
    model(torch.ones(1, 4) * 2)
    recorder.remove()
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["record_type"] == "feature"
    assert "temporal_delta" not in records[0]
    assert "temporal_delta" in records[1]
    assert len(recorder.handles) == 0

