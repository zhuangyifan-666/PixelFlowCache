from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pixelgen_command_plan_prints_generation_and_fid_commands(tmp_path: Path) -> None:
    out_script = tmp_path / "pixelgen_plan.sh"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage4a_pixelgen_eval_plan.py",
            "--num-images",
            "100",
            "--methods",
            "no_cache_50,bfc_quality_t02_08,bfc_speed_t02_10,reduced_steps_30",
            "--pixelgen-ckpt",
            str(tmp_path / "PixelGen_XL.ckpt"),
            "--out-script",
            str(out_script),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "run_pixelgen_stage4a_generate.py" in output
    assert "evaluate_stage4a_fid.py" in output
    assert "CUDA_VISIBLE_DEVICES=0" in output
    assert "CUDA_VISIBLE_DEVICES=3" in output
    assert "--batch-size 4" in output
    assert "--pixelgen-ckpt" in output
    assert "/path/to/imagenet/val" in output
    assert "torchrun" not in output
    assert "nohup" not in output
    assert out_script.exists()
    assert "Review and run commands manually" in out_script.read_text(encoding="utf-8")


def test_pixelgen_command_plan_second_round_methods() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage4a_pixelgen_eval_plan.py",
            "--num-images",
            "100",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "# Second round" in output
    assert "CUDA_VISIBLE_DEVICES=0 conda run -n pixelgen python scripts/run_pixelgen_stage4a_generate.py --method reduced_steps_35" in output
    assert "CUDA_VISIBLE_DEVICES=1 conda run -n pixelgen python scripts/run_pixelgen_stage4a_generate.py --method bfc_speed_t02_09" in output
    assert "--method seacache_style" not in output


def test_pixelgen_command_plan_seacache_style_prints_dynamic_args() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage4a_pixelgen_eval_plan.py",
            "--num-images",
            "100",
            "--methods",
            "seacache_style",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "run_pixelgen_stage4a_generate.py --method seacache_style" in output
    assert "--dynamic-cache-threshold 0.06" in output
    assert "--sea-beta 2.0" in output
    assert "--sea-proxy-downsample 64" in output
    assert "evaluate_stage4a_fid.py" in output
    assert "torchrun" not in output
    assert "accelerate launch" not in output
    assert "nohup" not in output
