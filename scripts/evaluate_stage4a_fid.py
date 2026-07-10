#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc.eval.generation_io import count_images, validate_output_indices  # noqa: E402
from pfc.eval.provenance import (  # noqa: E402
    collect_command_provenance,
    collect_git_provenance,
    collect_runtime_provenance,
    sha256_file,
)


def select_backend(requested: str) -> tuple[str | None, str]:
    if requested == "torchmetrics":
        return "torchmetrics", (
            "torchmetrics was selected explicitly, but its folder-evaluation backend is not implemented."
        )
    candidates = ["torch_fidelity", "cleanfid"] if requested == "auto" else [requested]
    for name in candidates:
        if importlib.util.find_spec(name) is not None:
            return name, f"selected backend: {name}"
    return None, (
        "No implemented FID backend found. Install one of: "
        "`pip install torch-fidelity` or `pip install clean-fid`."
    )


def validate_backend_capabilities(
    backend: str | None,
    metrics: list[str],
    *,
    fid_stats: Path | None,
    real_dir: Path | None,
    require_reference: bool = True,
) -> None:
    unknown = sorted(set(metrics) - {"fid", "is", "kid"})
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}")
    if backend is None:
        return
    if backend == "torchmetrics":
        raise NotImplementedError(
            "The torchmetrics backend is not implemented for folder loading. "
            "Use --backend torch_fidelity or --backend cleanfid."
        )
    if backend == "cleanfid":
        unsupported = sorted(set(metrics) - {"fid"})
        if unsupported:
            raise ValueError(f"cleanfid backend currently supports only FID; unsupported: {unsupported}")
        if require_reference and real_dir is None:
            raise ValueError("cleanfid backend currently requires --real-dir")
    if backend == "torch_fidelity":
        if fid_stats is not None and "kid" in metrics:
            raise ValueError("--fid-stats contains only FID reference statistics; provide --real-dir for KID.")
        if require_reference and ("fid" in metrics or "kid" in metrics) and fid_stats is None and real_dir is None:
            raise ValueError("torch_fidelity FID/KID requires --fid-stats or --real-dir")


def _write_results(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)


def _load_fid_statistics(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path) as data:
        if {"mu", "sigma"}.issubset(data.files):
            mu_key, sigma_key = "mu", "sigma"
        elif {"m", "s"}.issubset(data.files):
            mu_key, sigma_key = "m", "s"
        else:
            raise ValueError(f"Unsupported FID stats file keys in {path}: {sorted(data.files)}")
        return {
            "mu": np.asarray(data[mu_key]),
            "sigma": np.asarray(data[sigma_key]),
        }


def validate_fid_statistics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing FID stats file: {path}")
    stats = _load_fid_statistics(path)
    mu = stats["mu"]
    sigma = stats["sigma"]
    if getattr(mu, "ndim", None) != 1 or getattr(sigma, "ndim", None) != 2:
        raise ValueError(f"Invalid FID statistics shapes: mu={getattr(mu, 'shape', None)}, sigma={getattr(sigma, 'shape', None)}")
    if sigma.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError(f"FID sigma shape {sigma.shape} does not match mu length {mu.shape[0]}")
    return {
        "reference_stats_path": str(path.resolve()),
        "stats_sha256": sha256_file(path),
        "stats_size_bytes": path.stat().st_size,
        "stats_feature_dimension": int(mu.shape[0]),
    }


def validate_fake_images(fake_dir: Path, expected_images: int | None = None) -> dict[str, Any]:
    if not fake_dir.is_dir():
        raise FileNotFoundError(f"Missing fake image directory: {fake_dir}")
    png_count = sum(1 for child in fake_dir.iterdir() if child.is_file() and child.suffix.lower() == ".png")
    if png_count <= 0:
        raise ValueError(f"No PNG images found under fake directory: {fake_dir}")
    expected_count = png_count if expected_images is None else expected_images
    if expected_count <= 0:
        raise ValueError("--expected-images must be positive")
    report = validate_output_indices(fake_dir, range(expected_count), strict=False)
    if png_count != expected_count:
        report["expected_image_count"] = expected_count
    if not report["valid"] or png_count != expected_count:
        raise ValueError(f"Fake image validation failed: {report}")
    return report


def _backend_version(backend: str) -> str | None:
    distribution = {"torch_fidelity": "torch-fidelity", "cleanfid": "clean-fid"}.get(backend, backend)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _align_fid_statistics_dtype(fake_stats: dict[str, Any], ref_stats: dict[str, Any]) -> dict[str, Any]:
    aligned = dict(ref_stats)
    for key in ("mu", "sigma"):
        fake_value = fake_stats.get(key)
        ref_value = aligned.get(key)
        fake_dtype = getattr(fake_value, "dtype", None)
        if fake_dtype is not None and hasattr(ref_value, "astype"):
            aligned[key] = ref_value.astype(fake_dtype, copy=False)
    return aligned


def _compute_with_torch_fidelity_stats_file(args: argparse.Namespace) -> dict[str, Any]:
    from torch_fidelity.metric_fid import fid_featuresdict_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_featuresdict_to_metric
    from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached

    if "kid" in args.metrics:
        raise ValueError("--fid-stats contains only FID reference statistics; provide --real-dir to compute KID.")

    feature_layer_isc = "logits_unbiased"
    feature_layer_fid = "2048"
    feature_layers = []
    if "is" in args.metrics:
        feature_layers.append(feature_layer_isc)
    if "fid" in args.metrics:
        feature_layers.append(feature_layer_fid)

    kwargs: dict[str, Any] = {
        "input1": str(args.fake_dir),
        "cuda": args.device.startswith("cuda"),
        "batch_size": args.batch_size,
        "feature_extractor": "inception-v3-compat",
        "feature_layer_isc": feature_layer_isc,
        "feature_layer_fid": feature_layer_fid,
        "verbose": True,
    }
    feat_extractor = create_feature_extractor(kwargs["feature_extractor"], feature_layers, **kwargs)
    featuresdict = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)

    metrics: dict[str, Any] = {}
    if "is" in args.metrics:
        metrics.update(isc_featuresdict_to_metric(featuresdict, feature_layer_isc, **kwargs))
    if "fid" in args.metrics:
        fake_stats = fid_featuresdict_to_statistics(featuresdict, feature_layer_fid)
        ref_stats = _align_fid_statistics_dtype(fake_stats, _load_fid_statistics(args.fid_stats))
        metrics.update(fid_statistics_to_metric(fake_stats, ref_stats, kwargs["verbose"]))
    return {str(key): float(value) for key, value in metrics.items()}


def _compute_with_torch_fidelity(args: argparse.Namespace) -> dict[str, Any]:
    if args.fid_stats and "fid" in args.metrics:
        return _compute_with_torch_fidelity_stats_file(args)

    from torch_fidelity import calculate_metrics

    kwargs: dict[str, Any] = {
        "input1": str(args.fake_dir),
        "cuda": args.device.startswith("cuda"),
        "batch_size": args.batch_size,
        "isc": "is" in args.metrics,
        "fid": "fid" in args.metrics,
        "kid": "kid" in args.metrics,
        "verbose": True,
    }
    if args.real_dir:
        kwargs["input2"] = str(args.real_dir)
    elif "fid" in args.metrics or "kid" in args.metrics:
        raise ValueError("Real reference is required for torch_fidelity FID")
    metrics = calculate_metrics(**kwargs)
    return {str(key): float(value) for key, value in metrics.items()}


def _compute_with_cleanfid(args: argparse.Namespace) -> dict[str, Any]:
    from cleanfid import fid

    if not args.real_dir:
        raise ValueError("cleanfid backend currently requires --real-dir")
    return {"FID": float(fid.compute_fid(str(args.fake_dir), str(args.real_dir), num_workers=args.num_workers))}


def _compute_with_torchmetrics(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError(
        "torchmetrics backend is detected but not wired for folder loading yet; use torch_fidelity or cleanfid for Stage 4A."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 4A generated image folders with FID/IS/KID backends.")
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--fid-stats", type=Path)
    parser.add_argument("--backend", choices=["auto", "torch_fidelity", "cleanfid", "torchmetrics"], default="auto")
    parser.add_argument("--metrics", default="fid,is")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--proxy-result", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.metrics = [item.strip().lower() for item in args.metrics.split(",") if item.strip()]
    backend, message = select_backend(args.backend)
    validate_backend_capabilities(
        backend,
        args.metrics,
        fid_stats=args.fid_stats,
        real_dir=args.real_dir,
        require_reference=not args.dry_run,
    )
    image_validation = validate_fake_images(args.fake_dir, args.expected_images)
    fake_count = int(image_validation["image_count"])
    if args.real_dir and not args.real_dir.exists():
        raise FileNotFoundError(f"Missing real image directory: {args.real_dir}")
    stats_provenance = validate_fid_statistics(args.fid_stats) if args.fid_stats else {
        "reference_stats_path": None,
        "stats_sha256": None,
        "stats_size_bytes": None,
        "stats_feature_dimension": None,
    }
    proxy_only = bool(args.proxy_result or fake_count < 50_000)
    final_distribution_metric = bool(fake_count == 50_000 and not args.proxy_result)
    if args.dry_run:
        print(message)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "backend": backend,
                    "fake_dir": str(args.fake_dir.resolve()),
                    "num_fake_images": fake_count,
                    "real_dir": str(args.real_dir.resolve()) if args.real_dir else None,
                    "fid_stats": str(args.fid_stats.resolve()) if args.fid_stats else None,
                    "metrics": args.metrics,
                    "expected_images": args.expected_images,
                    "image_validation": image_validation,
                    "proxy_only": proxy_only,
                    "final_distribution_metric": final_distribution_metric,
                    **stats_provenance,
                    "out": str(args.out.resolve()),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if backend is None:
        print(message)
        return 2
    if backend == "torch_fidelity":
        metrics = _compute_with_torch_fidelity(args)
    elif backend == "cleanfid":
        metrics = _compute_with_cleanfid(args)
    elif backend == "torchmetrics":
        metrics = _compute_with_torchmetrics(args)
    else:
        raise AssertionError(f"Unhandled backend: {backend}")
    runtime_provenance = collect_runtime_provenance()
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "backend_version": _backend_version(backend),
        "feature_extractor": "inception-v3-compat" if backend == "torch_fidelity" else "cleanfid-default",
        "fake_dir": str(args.fake_dir.resolve()),
        "real_dir": str(args.real_dir.resolve()) if args.real_dir else None,
        "fid_stats": str(args.fid_stats.resolve()) if args.fid_stats else None,
        "metrics_requested": ",".join(args.metrics),
        "num_fake_images": fake_count,
        "image_count": fake_count,
        "expected_images": args.expected_images,
        "image_validation": image_validation,
        "batch_size": args.batch_size,
        "device": args.device,
        "torch_version": runtime_provenance.get("torch_version"),
        "proxy_only": proxy_only,
        "final_distribution_metric": final_distribution_metric,
        **stats_provenance,
        "command": collect_command_provenance(),
        "command_argv": list(sys.argv),
        "runtime": runtime_provenance,
        "git": collect_git_provenance(ROOT),
        **metrics,
    }
    _write_results(args.out, result)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
