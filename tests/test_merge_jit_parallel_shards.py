from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_jit_parallel_shards_merges_manifest_cache_and_latency(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True)
    for idx in range(8):
        (image_dir / f"{idx:06d}.png").write_bytes(b"png")
    for shard in range(4):
        rows = [{"index": idx, "label": idx} for idx in range(shard, 8, 4)]
        with (run_dir / f"manifest_shard{shard}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        _write_json(run_dir / f"cache_stats_shard{shard}.json", {"total_calls": 2, "hits": 1, "misses": 1, "refreshes": 1, "disabled": 0, "by_module": {"blocks.0": {"calls": 2, "hits": 1, "misses": 1, "refreshes": 1, "disabled": 0}}})
        _write_json(run_dir / f"latency_shard{shard}.json", {"latency_sec": 10 + shard, "generated_images_this_run": 2, "total_shard_images": 2})
        _write_json(run_dir / f"generation_meta_shard{shard}.json", {"method_name": "no_cache_50", "shard_index": shard})

    subprocess.run(
        [
            sys.executable,
            "scripts/merge_jit_parallel_shards.py",
            "--run-dir",
            str(run_dir),
            "--num-shards",
            "4",
            "--expected-images",
            "8",
            "--method",
            "no_cache_50",
            "--strict",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    manifest_indices = [json.loads(line)["index"] for line in (run_dir / "manifest.jsonl").read_text().splitlines()]
    assert manifest_indices == list(range(8))
    cache = json.loads((run_dir / "cache_stats.json").read_text())
    assert cache["total_calls"] == 8
    assert cache["hits"] == 4
    latency = json.loads((run_dir / "latency.json").read_text())
    assert latency["parallel_latency_sec"] == 13
    assert latency["images_per_sec"] == 8 / 13
