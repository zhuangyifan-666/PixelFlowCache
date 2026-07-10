from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.run_parallel_generate as launcher


ROOT = Path(__file__).resolve().parents[1]


def _args(*extra: str):
    args = launcher.build_parser().parse_args(
        ["--model", "jit", "--method", "no_cache_50", "--run-id", "test", *extra]
    )
    args.num_shards = args.num_shards or len(launcher.split_csv(args.gpus))
    launcher.resolve_launch_options(args)
    return args


@pytest.mark.parametrize(
    ("model", "environment", "batch_size"),
    [("jit", "jit", 8), ("deco", "deco", 4), ("pixelgen", "pixelgen", 4)],
)
def test_model_environment_and_default_batch(model: str, environment: str, batch_size: int) -> None:
    args = _args("--model", model)
    command = launcher.worker_command(args, 0)
    assert command[:5] == ["conda", "run", "-n", environment, "python"]
    assert command[command.index("--batch-size") + 1] == str(batch_size)


def test_method_model_registry_validation() -> None:
    args = launcher.build_parser().parse_args(
        ["--model", "deco", "--method", "taylorseer_style", "--run-id", "bad"]
    )
    with pytest.raises(ValueError, match="not supported"):
        launcher.resolve_launch_options(args)


@pytest.mark.parametrize(
    ("method", "specific_flag", "path"),
    [
        ("safe_bfc_quality", "--safe-map-quality", "quality.json"),
        ("safe_bfc_speed", "--safe-map-speed", "speed.json"),
    ],
)
def test_safe_methods_resolve_matching_map(method: str, specific_flag: str, path: str) -> None:
    args = _args("--method", method, specific_flag, path)
    command = launcher.worker_command(args, 0)
    assert command[command.index("--safe-map") + 1] == path


def test_safe_map_common_override_wins_and_missing_map_fails() -> None:
    args = _args(
        "--method", "safe_bfc_quality", "--safe-map", "common.json",
        "--safe-map-quality", "quality.json",
    )
    assert launcher.worker_command(args, 0)[-2:] == ["--safe-map", "common.json"]
    missing = launcher.build_parser().parse_args(
        ["--model", "jit", "--method", "safe_bfc_speed", "--run-id", "missing"]
    )
    with pytest.raises(ValueError, match="requires --safe-map"):
        launcher.resolve_launch_options(missing)


def test_non_safe_method_does_not_receive_safe_map_and_seacache_gets_registry_args() -> None:
    args = _args("--method", "seacache_style", "--safe-map", "ignored.json")
    command = launcher.worker_command(args, 0)
    assert "--safe-map" not in command
    assert command[command.index("--dynamic-cache-threshold") + 1] == "0.06"
    assert "--sea-beta" in command


@pytest.mark.parametrize(
    ("method", "debug_flag"),
    [
        ("safe_bfc_quality", "--safe-debug-jsonl"),
        ("seacache_style", "--dynamic-cache-debug-jsonl"),
        ("taylorseer_style", "--taylorseer-debug-jsonl"),
        ("speca_style", "--speca-debug-jsonl"),
        ("dicache_style", "--dicache-debug-jsonl"),
    ],
)
def test_debug_jsonl_routes_by_method_and_shard(method: str, debug_flag: str) -> None:
    extra = ["--method", method, "--debug-jsonl", "debug.jsonl", "--num-shards", "4"]
    if method == "safe_bfc_quality":
        extra.extend(["--safe-map", "safe.json"])
    args = _args(*extra)
    paths = []
    for shard in range(4):
        command = launcher.worker_command(args, shard)
        assert debug_flag in command
        paths.append(command[command.index(debug_flag) + 1])
    assert paths == [f"debug_shard{index}.jsonl" for index in range(4)]


def test_reference_debug_is_rejected() -> None:
    args = launcher.build_parser().parse_args(
        [
            "--model", "jit", "--method", "no_cache_50", "--run-id", "test",
            "--debug-jsonl", "debug.jsonl",
        ]
    )
    with pytest.raises(ValueError, match="unsupported"):
        launcher.resolve_launch_options(args)


def test_custom_environment_python_override_and_worker_json_tokens() -> None:
    custom = _args("--env-jit", "custom-jit")
    assert launcher.worker_command(custom, 0)[:5] == ["conda", "run", "-n", "custom-jit", "python"]

    override = _args(
        "--python-executable", "C:/Python/python.exe",
        "--worker-arg=--flag",
        "--worker-args-json", '["value with space", "--second"]',
    )
    command = launcher.worker_command(override, 0)
    assert command[0] == "C:/Python/python.exe"
    assert "conda" not in command
    assert command[-3:] == ["--flag", "value with space", "--second"]


def test_print_only_does_not_execute_and_contains_no_distributed_launcher(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_parallel_generate.py",
            "--model", "pixelgen",
            "--method", "no_cache_50",
            "--gpus", "0,1,2,3",
            "--num-shards", "4",
            "--run-id", "print_only",
            "--output-root", str(tmp_path),
            "--print-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.stdout.count("conda run -n pixelgen") == 4
    assert "--batch-size 4" in result.stdout
    assert "PixelGen_XL_160ep.ckpt" in result.stdout
    assert all(token not in result.stdout for token in ("torchrun", "accelerate", "nohup"))
    assert not list(tmp_path.iterdir())


def test_worker_failure_returns_before_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedProcess:
        pid = 123

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    monkeypatch.setattr(launcher.signal, "signal", lambda *args, **kwargs: None)

    def merge_must_not_run(*args, **kwargs):
        raise AssertionError("merge must not run after worker failure")

    monkeypatch.setattr(launcher.subprocess, "run", merge_must_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel_generate.py",
            "--model", "jit",
            "--method", "no_cache_50",
            "--gpus", "0",
            "--num-shards", "1",
            "--run-id", "failed",
            "--output-root", str(tmp_path),
            "--execute",
        ],
    )
    assert launcher.main() == 1
    run_dir = tmp_path / "jit" / "failed" / "no_cache_50"
    assert (run_dir / "parallel_launcher_meta.json").is_file()
    assert not (run_dir / "parallel_merge_report.json").exists()
