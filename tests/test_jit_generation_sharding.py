from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_jit_stage4a_generate import _apply_shard_paths, compute_shard_indices


def test_strided_shards_cover_indices_without_overlap() -> None:
    shards = [compute_shard_indices(10, 4, idx, "strided") for idx in range(4)]
    merged = sorted(index for shard in shards for index in shard)
    assert merged == list(range(10))
    assert len(set(merged)) == 10


def test_contiguous_shards_cover_indices_without_overlap() -> None:
    shards = [compute_shard_indices(10, 4, idx, "contiguous") for idx in range(4)]
    merged = sorted(index for shard in shards for index in shard)
    assert merged == list(range(10))
    assert shards[0] == [0, 1, 2]
    assert shards[-1] == [8, 9]


def test_shard_manifest_names_are_separate(tmp_path: Path) -> None:
    base = tmp_path / "run"
    paths = {"base_dir": base, "manifest": base / "manifest.jsonl", "latency": base / "latency.json", "cache_stats": base / "cache_stats.json", "generation_meta": base / "generation_meta.json"}
    args = Namespace(num_shards=4, shard_index=2, manifest_suffix=None)
    sharded = _apply_shard_paths(paths, args)
    assert sharded["manifest"].name == "manifest_shard2.jsonl"
    assert sharded["latency"].name == "latency_shard2.json"
