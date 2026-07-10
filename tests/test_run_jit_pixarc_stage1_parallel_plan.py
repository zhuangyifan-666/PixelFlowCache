import json
import subprocess
import sys
from pathlib import Path

import scripts.run_jit_pixarc_stage1_parallel as parallel
from scripts.run_jit_pixarc_stage1_parallel import build_parallel_plan, build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_parallel_plan_uses_independent_conda_workers_and_shards():
    args = build_parser().parse_args(
        ["--run-id", "plan", "--gpus", "0,1,2,3", "--num-shards", "4", "--print-only"]
    )
    plan = build_parallel_plan(args)
    assert len(plan["workers"]) == 4
    for index, worker in enumerate(plan["workers"]):
        assert worker["argv"][:5] == ["conda", "run", "-n", "jit", "python"]
        assert worker["environment"] == {"CUDA_VISIBLE_DEVICES": str(index)}
        shard_position = worker["argv"].index("--shard-index")
        assert worker["argv"][shard_position + 1] == str(index)
        command = " ".join(worker["argv"]).lower()
        assert all(forbidden not in command for forbidden in ("torchrun", "accelerate", "nohup"))


def test_parallel_plan_honors_python_override():
    args = build_parser().parse_args(
        [
            "--run-id",
            "override",
            "--gpus",
            "2,3",
            "--num-shards",
            "2",
            "--python-executable",
            "/opt/jit/bin/python",
        ]
    )
    plan = build_parallel_plan(args)
    assert all(worker["argv"][0] == "/opt/jit/bin/python" for worker in plan["workers"])
    assert all("conda" not in worker["argv"] for worker in plan["workers"])


def test_print_only_does_not_create_run_or_execute_worker(tmp_path):
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_jit_pixarc_stage1_parallel.py",
            "--run-id",
            "print-only",
            "--output-root",
            str(output_root),
            "--gpus",
            "0,1,2,3",
            "--num-shards",
            "4",
            "--print-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["print_only"] is True
    assert len(payload["workers"]) == 4
    assert not output_root.exists()


def test_worker_failure_terminates_peers_and_does_not_merge(tmp_path, monkeypatch):
    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    processes = [FakeProcess(2), FakeProcess(None)]
    monkeypatch.setattr(parallel, "_is_windows", lambda: False)
    monkeypatch.setattr(parallel.subprocess, "Popen", lambda *args, **kwargs: processes.pop(0))
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("merge must not run")),
    )
    plan = {
        "run_dir": str(tmp_path / "run"),
        "workers": [
            {"shard_index": 0, "gpu": "0", "environment": {}, "argv": ["worker0"]},
            {"shard_index": 1, "gpu": "1", "environment": {}, "argv": ["worker1"]},
        ],
        "merge_enabled": True,
        "merge_argv": ["merge"],
    }
    assert parallel.execute_parallel_plan(plan) == 1
