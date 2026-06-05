from __future__ import annotations

import json
from pathlib import Path

from pfc.eval.generation_io import (
    append_generation_manifest,
    load_generation_manifest,
    prepare_generation_dir,
    write_generation_meta,
)


def test_generation_dir_manifest_and_meta(tmp_path: Path) -> None:
    paths = prepare_generation_dir(tmp_path, "JiT", "no_cache_50", "run0")
    assert paths["base_dir"].name == "no_cache_50"
    assert paths["image_dir"].exists()

    records = [{"index": 0, "label": 0, "path": "000000.png"}, {"index": 1, "label": 1, "path": "000001.png"}]
    append_generation_manifest(paths["manifest"], records)
    assert load_generation_manifest(paths["manifest"]) == records

    write_generation_meta(paths["generation_meta"], {"model": "JiT", "method": "no_cache_50"})
    meta = json.loads(paths["generation_meta"].read_text(encoding="utf-8"))
    assert meta["model"] == "JiT"
    assert "timestamp_utc" in meta

