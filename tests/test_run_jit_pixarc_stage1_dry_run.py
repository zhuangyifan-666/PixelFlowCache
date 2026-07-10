import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage1_dry_run_resolves_without_outputs_or_checkpoint(tmp_path):
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_pixarc_stage1_instrument.py",
            "--run-id",
            "dry",
            "--output-root",
            str(output_root),
            "--num-images",
            "32",
            "--steps",
            "50",
            "--num-shards",
            "4",
            "--shard-index",
            "2",
            "--jit-ckpt-dir",
            str(tmp_path / "missing-checkpoint"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["checkpoint_checked"] is False
    assert payload["model_loaded"] is False
    assert payload["cuda_used"] is False
    assert payload["indices"] == list(range(2, 32, 4))
    assert not output_root.exists()


def test_stage1_rejects_non_unit_batch_before_runtime(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_pixarc_stage1_instrument.py",
            "--run-id",
            "bad-batch",
            "--batch-size",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires batch_size=1" in result.stderr
