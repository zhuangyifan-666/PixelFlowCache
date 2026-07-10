#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.method_presets import get_method_metadata, method_supports_model  # noqa: E402
from pfc.eval.provenance import (  # noqa: E402
    collect_file_provenance,
    collect_git_provenance,
    collect_gpu_provenance,
    collect_runtime_provenance,
    collect_submodule_provenance,
)
PASS = "PASS"
WARN = "WARN"
BLOCK = "BLOCK"
MODEL_NAMES = {"jit", "deco", "pixelgen", "pixeldit"}
THIRD_PARTY = {
    "jit": "JiT",
    "deco": "DeCo",
    "pixelgen": "PixelGen",
    "pixeldit": "PixelDiT",
}
ENV_PACKAGES = {
    "jit": ("torch", "torchvision", "timm", "PIL", "numpy"),
    "deco": ("torch", "torchvision", "timm", "PIL", "numpy"),
    "pixelgen": ("torch", "torchvision", "timm", "PIL", "numpy"),
}
TORCH_FIDELITY_INTERNAL_APIS = {
    "torch_fidelity.metric_fid": (
        "fid_featuresdict_to_statistics",
        "fid_statistics_to_metric",
    ),
    "torch_fidelity.metric_isc": ("isc_featuresdict_to_metric",),
    "torch_fidelity.utils": (
        "create_feature_extractor",
        "extract_featuresdict_from_input_id_cached",
    ),
}


def check_config_files(config_dir: Path) -> dict[str, Any]:
    try:
        from scripts.check_config_consistency import check_configs
    except Exception as exc:
        return {
            "valid": False,
            "checked_files": [],
            "errors": [f"config consistency checker is unavailable: {exc}"],
            "warnings": [],
        }
    return check_configs(config_dir)
PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "torch_fidelity": "torch-fidelity",
    "cleanfid": "clean-fid",
    "PIL": "Pillow",
    "numpy": "numpy",
    "scipy": "scipy",
    "lpips": "lpips",
    "skimage": "scikit-image",
    "pytest": "pytest",
    "yaml": "PyYAML",
}


@dataclass
class Check:
    category: str
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _add(
    checks: list[Check], category: str, name: str, status: str, message: str, **details: Any
) -> None:
    checks.append(Check(category, name, status, message, details or None))


def check_git(checks: list[Check]) -> None:
    git = collect_git_provenance(ROOT, include_diff_stat=True)
    status = WARN if git.get("git_dirty") else PASS
    _add(checks, "git", "worktree", status, "worktree is dirty" if status == WARN else "worktree is clean", **git)
    large_untracked: list[dict[str, Any]] = []
    for relative in git.get("git_status_porcelain", []):
        if not str(relative).startswith("?? "):
            continue
        path = ROOT / str(relative)[3:]
        if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024:
            large_untracked.append({"path": str(path), "size": path.stat().st_size})
    _add(
        checks,
        "git",
        "untracked_large_files",
        WARN if large_untracked else PASS,
        f"found {len(large_untracked)} untracked files >=100 MiB",
        files=large_untracked,
    )


def check_third_party(checks: list[Check], models: list[str]) -> None:
    submodules = {row["path"]: row for row in collect_submodule_provenance(ROOT)}
    for key in models:
        directory = THIRD_PARTY[key]
        path = ROOT / "third_party" / directory
        row = submodules.get(f"third_party/{directory}", {})
        has_source = path.is_dir() and any(path.rglob("*.py"))
        status = PASS if row.get("initialized") and has_source else BLOCK
        _add(
            checks,
            "third_party",
            key,
            status,
            "initialized source checkout present" if status == PASS else "submodule is missing, uninitialized, or has no Python source",
            path=str(path),
            submodule=row or None,
        )


def _path_size(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return None


def check_checkpoints(checks: list[Check], args: argparse.Namespace, models: list[str]) -> None:
    mapping = {
        "jit": args.jit_ckpt_dir,
        "deco": args.deco_ckpt,
        "pixelgen": args.pixelgen_ckpt,
    }
    for model in models:
        if model not in mapping:
            continue
        path = mapping[model]
        artifact = path / "checkpoint-last.pth" if model == "jit" else path
        exists = artifact.is_file()
        size = _path_size(artifact)
        details = collect_file_provenance(artifact, hash_file=args.hash_files)
        details["configured_path"] = str(path.resolve())
        plausible = bool(exists and size and size >= 1024 * 1024)
        _add(
            checks,
            "checkpoint",
            model,
            PASS if plausible else BLOCK,
            "checkpoint path exists with plausible size" if plausible else "checkpoint path is missing or smaller than 1 MiB",
            **details,
        )


def check_fid_stats(checks: list[Check], path: Path, hash_files: bool) -> None:
    if not path.is_file():
        _add(checks, "fid", "reference_stats", BLOCK, "FID stats file is missing", path=str(path))
        return
    try:
        import numpy as np

        with np.load(path) as data:
            keys = sorted(data.files)
            valid = {"mu", "sigma"}.issubset(keys) or {"m", "s"}.issubset(keys)
    except Exception as exc:
        _add(checks, "fid", "reference_stats", BLOCK, f"FID stats are unreadable: {exc}", path=str(path))
        return
    provenance = collect_file_provenance(path, hash_file=hash_files)
    _add(
        checks, "fid", "reference_stats", PASS if valid else BLOCK,
        "FID stats are readable" if valid else f"FID stats lack required keys: {keys}",
        keys=keys, **provenance,
    )


def _safe_map_summary(path: Path) -> tuple[str, str, dict[str, Any]]:
    if not path.is_file():
        return BLOCK, "safe map is missing", {"path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return BLOCK, f"safe map is unreadable: {exc}", {"path": str(path)}
    safe_tree = payload.get("safe")
    schema_key = "safe"
    if safe_tree is None and "safe_map" in payload:
        safe_tree = payload["safe_map"]
        schema_key = "safe_map"
    values: list[bool] = []

    def visit(node: Any) -> None:
        if isinstance(node, bool):
            values.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(safe_tree)
    safe_true = sum(values)
    modules = payload.get(
        "selected_modules",
        payload.get(
            "modules",
            payload.get("module_names", payload.get("boundary_groups", payload.get("module_to_boundary"))),
        ),
    )
    details = {
        "path": str(path.resolve()), "safe_total": len(values), "safe_true": safe_true,
        "density": safe_true / len(values) if values else 0.0,
        "schema_key": schema_key if safe_tree is not None else None,
        "model": payload.get("model_name", payload.get("model")),
        "steps": payload.get("steps", payload.get("total_steps")),
        "selected_modules": modules,
        "branches": payload.get("branches"),
        "solver_stages": payload.get("solver_stages"),
        "max_age": payload.get("max_age"),
    }
    if not values or not safe_true:
        return BLOCK, "safe map is empty or contains no reusable entries", details
    if details["model"] is not None and "jit" not in str(details["model"]).lower():
        return BLOCK, "safe map model does not match JiT", details
    if details["steps"] is not None and int(details["steps"]) != 50:
        return BLOCK, "safe map step count does not match the 50-step runtime", details
    if details["selected_modules"] is not None and not details["selected_modules"]:
        return BLOCK, "safe map has no module/boundary mapping", details
    return PASS, "safe map contains reusable entries", details


def check_safe_maps(checks: list[Check], args: argparse.Namespace, methods: list[str]) -> None:
    requested = []
    if "safe_bfc_quality" in methods:
        requested.append(("quality", args.safe_map_quality))
    if "safe_bfc_speed" in methods:
        requested.append(("speed", args.safe_map_speed))
    for name, path in requested:
        status, message, details = _safe_map_summary(path)
        _add(checks, "safe_map", name, status, message, **details)


def check_environment(checks: list[Check]) -> None:
    required = {"torch", "torchvision", "timm", "PIL", "numpy", "scipy", "lpips", "skimage", "pytest", "yaml"}
    versions: dict[str, str | None] = {}
    for module, distribution in PACKAGE_DISTRIBUTIONS.items():
        found = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if found else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        versions[module] = version
        status = PASS if found else (BLOCK if module in required else WARN)
        _add(checks, "environment", module, status, f"version={version}" if found else "package not installed")
    backend_found = bool(versions["torch_fidelity"] or versions["cleanfid"])
    _add(
        checks, "environment", "fid_backend", PASS if backend_found else BLOCK,
        "implemented FID backend available" if backend_found else "install torch-fidelity or clean-fid",
    )
    try:
        import torch

        lpips_weights = any((Path(torch.hub.get_dir()) / "checkpoints").glob("alexnet-*.pth"))
    except Exception:
        lpips_weights = False
    _add(
        checks, "environment", "lpips_weights", PASS if lpips_weights else WARN,
        "LPIPS AlexNet weights are pre-staged" if lpips_weights else "LPIPS weights are not cached; pre-stage them or explicitly allow download",
    )
    runtime = collect_runtime_provenance()
    _add(checks, "environment", "runtime", PASS, "runtime provenance collected", **runtime)

    status, message, details = _torch_fidelity_capability()
    _add(checks, "fid", "torch_fidelity_internal_api", status, message, **details)


def _torch_fidelity_capability() -> tuple[str, str, dict[str, Any]]:
    try:
        version = importlib.metadata.version("torch-fidelity")
    except importlib.metadata.PackageNotFoundError:
        version = None
    missing: list[str] = []
    import_errors: list[str] = []
    for module_name, attributes in TORCH_FIDELITY_INTERNAL_APIS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
        for attribute in attributes:
            if not hasattr(module, attribute):
                missing.append(f"{module_name}.{attribute}")
    details = {
        "version": version,
        "required_internal_apis": TORCH_FIDELITY_INTERNAL_APIS,
        "missing_internal_apis": missing,
        "import_errors": import_errors,
    }
    if import_errors or missing:
        return BLOCK, f"torch-fidelity {version or 'unknown'} lacks required internal APIs", details
    return PASS, f"torch-fidelity {version or 'unknown'} internal APIs are compatible", details


def check_conda_environments(
    checks: list[Check],
    args: argparse.Namespace,
    models: list[str],
) -> None:
    requested = [model for model in models if model in ENV_PACKAGES]
    conda = shutil.which("conda")
    if conda is None:
        status = BLOCK if args.strict else WARN
        for model in requested:
            _add(
                checks,
                "conda_environment",
                model,
                status,
                "conda is unavailable; model environment probe was not run",
                environment=getattr(args, f"env_{model}"),
            )
        return

    for model in requested:
        environment = str(getattr(args, f"env_{model}"))
        modules = ENV_PACKAGES[model]
        probe = (
            "import importlib.util,json,sys; "
            f"mods={list(modules)!r}; "
            "missing=[name for name in mods if importlib.util.find_spec(name) is None]; "
            "import torch; "
            "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
            "'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),"
            "'missing':missing}))"
        )
        command = [conda, "run", "-n", environment, "python", "-c", probe]
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
                timeout=60,
            )
            payload = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {}
        except Exception as exc:
            result = None
            payload = {}
            error = str(exc)
        else:
            error = result.stderr.strip() if result.returncode else ""
        missing = payload.get("missing") if isinstance(payload, dict) else None
        success = bool(result is not None and result.returncode == 0 and not missing)
        status = PASS if success else (BLOCK if args.strict else WARN)
        _add(
            checks,
            "conda_environment",
            model,
            status,
            "environment probe passed" if success else "environment probe failed",
            environment=environment,
            command=command,
            payload=payload,
            error=error,
        )


def _nvidia_smi_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    command = [
        executable,
        "--query-gpu=driver_version,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, cwd=ROOT, check=False, text=True, capture_output=True, timeout=15
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "command": command}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "available": result.returncode == 0,
        "command": command,
        "rows": rows,
        "stderr": result.stderr.strip(),
    }


def check_gpu(checks: list[Check], required_gpus: int) -> None:
    gpu = collect_gpu_provenance()
    count = int(gpu.get("gpu_count", 0))
    inventory = _nvidia_smi_inventory()
    if required_gpus == 0:
        status = PASS
        message = "GPU validation disabled by required_gpus=0"
    else:
        status = PASS if gpu.get("cuda_available") and count >= required_gpus else BLOCK
        message = f"{count} CUDA GPU(s) visible; {required_gpus} required"
    _add(
        checks, "gpu", "cuda_inventory", status,
        message, **gpu, nvidia_smi=inventory,
    )


def check_disk(checks: list[Check], args: argparse.Namespace) -> None:
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / 1024**3
    _add(
        checks, "disk", "free_space", PASS if free_gb >= args.min_free_disk_gb else BLOCK,
        f"{free_gb:.2f} GiB free; {args.min_free_disk_gb:.2f} GiB required",
        free_bytes=usage.free,
    )
    for name, path in (("output", ROOT / "outputs"), ("logs", args.out.parent), ("cache", ROOT / ".cache")):
        probe = path if path.exists() else next((parent for parent in path.parents if parent.exists()), ROOT)
        writable = os.access(probe, os.W_OK)
        _add(checks, "disk", f"{name}_writable", PASS if writable else BLOCK, f"write probe parent: {probe}")


def _tracked_shells() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.sh"], cwd=ROOT, check=False, text=True, capture_output=True, timeout=15
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def check_scripts(checks: list[Check]) -> None:
    shells = _tracked_shells()
    crlf = [str(path.relative_to(ROOT)) for path in shells if b"\r\n" in path.read_bytes()]
    _add(checks, "scripts", "shell_lf", PASS if not crlf else BLOCK, f"{len(crlf)} tracked shell scripts contain CRLF", files=crlf)
    bash = shutil.which("bash")
    failures: list[str] = []
    if bash:
        for path in shells:
            result = subprocess.run([bash, "-n", str(path)], cwd=ROOT, check=False, text=True, capture_output=True, timeout=60)
            if result.returncode:
                failures.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")
        status = PASS if not failures else BLOCK
        message = f"bash -n checked {len(shells)} scripts"
    else:
        status = WARN
        message = "bash is unavailable; syntax check deferred to server"
    _add(checks, "scripts", "bash_syntax", status, message, failures=failures)
    hard_paths: list[str] = []
    stale_mount_prefix = "/" + "mnt/iset/"
    stale_repo_fragment = "zhuangyifan" + "/PixelFlowCache"
    for base in (ROOT / "scripts", ROOT / "configs"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".sh", ".yaml", ".yml"}:
                if path.resolve() == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if stale_mount_prefix in text or stale_repo_fragment in text:
                    hard_paths.append(str(path.relative_to(ROOT)))
    _add(checks, "scripts", "hard_coded_paths", PASS if not hard_paths else BLOCK, f"found {len(hard_paths)} stale absolute-path files", files=hard_paths)
    required = [
        "run_jit_stage4a_generate.py", "run_deco_stage4a_generate.py",
        "run_pixelgen_stage4a_generate.py", "run_server_readiness_plan.py",
    ]
    missing = [name for name in required if not (ROOT / "scripts" / name).is_file()]
    _add(checks, "scripts", "entrypoints", PASS if not missing else BLOCK, f"missing entrypoints: {missing}")
    planner_failures: list[str] = []
    planner_commands = [
        [
            sys.executable, "scripts/run_server_readiness_plan.py", "--models", "jit",
            "--methods", "no_cache_50", "--gpus", "0", "--skip-safe-calibration", "--print-only",
        ],
        [
            sys.executable, "scripts/run_jit_single_gpu_timing_plan.py", "--methods",
            "no_cache_50", "--repeats", "1", "--print-only",
        ],
    ]
    for command in planner_commands:
        result = subprocess.run(
            command, cwd=ROOT, check=False, text=True, capture_output=True, timeout=60
        )
        if result.returncode or "--resume" in result.stdout:
            planner_failures.append(
                f"{command}: rc={result.returncode}; stderr={result.stderr.strip()}; default_resume={'--resume' in result.stdout}"
            )
    _add(
        checks, "scripts", "planner_dry_runs", PASS if not planner_failures else BLOCK,
        "planner print-only checks passed" if not planner_failures else "planner print-only check failed",
        failures=planner_failures,
    )


def check_configs(checks: list[Check], models: list[str], methods: list[str], args: argparse.Namespace) -> None:
    unsupported: list[str] = []
    resolved: dict[str, Any] = {}
    for model in models:
        if model not in {"jit", "deco", "pixelgen"}:
            continue
        for method in methods:
            if method_supports_model(model, method):
                resolved[f"{model}:{method}"] = get_method_metadata(model, method)
            elif method in {"safe_bfc_quality", "safe_bfc_speed", "taylorseer_style", "speca_style", "dicache_style"} and model != "jit":
                continue
            else:
                unsupported.append(f"{model}:{method}")
    _add(checks, "config", "method_registry", PASS if not unsupported else BLOCK, f"unsupported model/method pairs: {unsupported}", resolved=resolved)
    consistency = check_config_files(ROOT / "configs")
    _add(
        checks,
        "config",
        "config_consistency",
        PASS if consistency["valid"] else BLOCK,
        "canonical config consistency passed" if consistency["valid"] else "canonical config consistency failed",
        **consistency,
    )
    expected_ckpts = {
        "jit": Path("ckpts/JiT/JiT-B-16-256"),
        "deco": Path("ckpts/DeCo/DeCo_XL.ckpt"),
        "pixelgen": Path("ckpts/PixelGen/PixelGen_XL_160ep.ckpt"),
    }
    actual = {"jit": args.jit_ckpt_dir, "deco": args.deco_ckpt, "pixelgen": args.pixelgen_ckpt}
    mismatches = [
        model for model in models
        if model in expected_ckpts and Path(actual[model]) != expected_ckpts[model]
    ]
    _add(checks, "config", "checkpoint_defaults", WARN if mismatches else PASS, f"noncanonical checkpoint arguments: {mismatches}")


def _write_reports(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md_path = path.with_suffix(".md")
    lines = ["# PixelFlowCache Experiment Preflight", "", f"Overall: **{payload['overall_status']}**", "", "| Category | Check | Status | Message |", "| --- | --- | --- | --- |"]
    for row in payload["checks"]:
        lines.append(f"| {row['category']} | {row['name']} | {row['status']} | {row['message'].replace('|', '/')} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static, no-download experiment preflight. Never loads a checkpoint or model.")
    parser.add_argument("--models", default="jit,deco,pixelgen")
    parser.add_argument("--methods", default="no_cache_50")
    parser.add_argument("--jit-ckpt-dir", type=Path, default=Path("ckpts/JiT/JiT-B-16-256"))
    parser.add_argument("--deco-ckpt", type=Path, default=Path("ckpts/DeCo/DeCo_XL.ckpt"))
    parser.add_argument("--pixelgen-ckpt", type=Path, default=Path("ckpts/PixelGen/PixelGen_XL_160ep.ckpt"))
    parser.add_argument("--fid-stats", type=Path, default=Path("third_party/JiT/fid_stats/jit_in256_stats.npz"))
    parser.add_argument("--safe-map-quality", type=Path, default=Path("calibrations/jit_safe/stage5a_jit_safe_calib128_seed123/safe_map_quality.json"))
    parser.add_argument("--safe-map-speed", type=Path, default=Path("calibrations/jit_safe/stage5a_jit_safe_calib128_seed123/safe_map_speed.json"))
    parser.add_argument("--env-jit", default="jit")
    parser.add_argument("--env-deco", default="deco")
    parser.add_argument("--env-pixelgen", default="pixelgen")
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--min-free-disk-gb", type=float, default=100.0)
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("logs/preflight/preflight_report.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = _csv(args.models)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown_models = sorted(set(models) - MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unsupported models: {unknown_models}")
    if args.required_gpus < 0:
        raise ValueError("--required-gpus must be non-negative")
    checks: list[Check] = []
    check_git(checks)
    check_third_party(checks, models)
    check_checkpoints(checks, args, models)
    check_fid_stats(checks, args.fid_stats, args.hash_files)
    check_safe_maps(checks, args, methods)
    check_environment(checks)
    check_conda_environments(checks, args, models)
    check_gpu(checks, args.required_gpus)
    check_disk(checks, args)
    check_scripts(checks)
    check_configs(checks, models, methods, args)
    overall = BLOCK if any(check.status == BLOCK for check in checks) else WARN if any(check.status == WARN for check in checks) else PASS
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "strict": args.strict,
        "models": models,
        "methods": methods,
        "overall_status": overall,
        "counts": {status: sum(check.status == status for check in checks) for status in (PASS, WARN, BLOCK)},
        "checks": [asdict(check) for check in checks],
    }
    _write_reports(args.out, payload)
    print(f"{overall}: {args.out}")
    return 2 if args.strict and overall == BLOCK else 0


if __name__ == "__main__":
    raise SystemExit(main())
