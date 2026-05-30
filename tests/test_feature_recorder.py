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


def test_feature_recorder_tracks_cond_uncond_previous_separately(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Identity())
    path = tmp_path / "feature_stats.jsonl"
    writer = JsonlWriter(path)
    recorder = FeatureRecorder(
        module_filter=lambda name, module: name == "0",
        writer=writer,
        model_name="tiny",
    )
    recorder.attach(model)
    recorder.set_context(0, 0.0, cfg_branch="cond")
    model(torch.ones(1, 4))
    recorder.set_context(0, 0.0, cfg_branch="uncond")
    model(torch.zeros(1, 4))
    recorder.set_context(1, 0.1, cfg_branch="cond")
    model(torch.ones(1, 4) * 2)
    recorder.remove()
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert "temporal_delta" not in records[1]
    assert records[2]["cfg_branch"] == "cond"
    assert records[2]["temporal_delta"]["rel_l2_delta"] == 1.0


def test_feature_recorder_split_cfg_cat_batch(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Identity())
    path = tmp_path / "feature_stats.jsonl"
    writer = JsonlWriter(path)
    recorder = FeatureRecorder(
        module_filter=lambda name, module: name == "0",
        writer=writer,
        model_name="tiny",
        split_batch_dim0=True,
    )
    recorder.attach(model)
    recorder.set_context(0, 0.0, cfg_branch="cfg_cat", extra={"cfg_cat_batch_size": 2})
    model(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    recorder.remove()
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    split = record["split_batch_dim0"]
    assert split["batch_size"] == 2
    assert split["uncond"]["shape"] == [2, 4]
    assert split["cond"]["shape"] == [2, 4]
    assert split["cond_minus_uncond"]["shape"] == [2, 4]
