#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


@dataclass
class MetricSummary:
    count: int
    mean: float
    std: float
    min: float
    max: float


def _split_metrics(value: str) -> list[str]:
    metrics = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"psnr", "ssim", "lpips", "rel_l2"}
    unknown = sorted(set(metrics) - valid)
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    return metrics


def _png_names(path: Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing image directory: {path}")
    return sorted(child.name for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".png")


def _paired_names(reference_dir: Path, method_dir: Path, strict: bool, limit: int | None) -> list[str]:
    ref_names = _png_names(reference_dir)
    method_names = _png_names(method_dir)
    ref_set = set(ref_names)
    method_set = set(method_names)
    missing = sorted(ref_set - method_set)
    extra = sorted(method_set - ref_set)
    if strict and (missing or extra):
        examples = {
            "missing_in_method": missing[:20],
            "extra_in_method": extra[:20],
            "missing_count": len(missing),
            "extra_count": len(extra),
        }
        raise RuntimeError(f"Reference and method filenames do not match: {examples}")
    names = sorted(ref_set & method_set)
    if limit is not None:
        names = names[:limit]
    return names


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((ref - pred) ** 2))
    if mse <= EPS:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(structural_similarity(ref, pred, channel_axis=-1, data_range=1.0))


def _rel_l2(ref: np.ndarray, pred: np.ndarray) -> float:
    numerator = float(np.linalg.norm((pred - ref).reshape(-1), ord=2))
    denominator = float(np.linalg.norm(ref.reshape(-1), ord=2))
    return numerator / max(denominator, EPS)


def _summarize(values: list[float]) -> MetricSummary:
    if not values:
        return MetricSummary(count=0, mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    return MetricSummary(
        count=int(arr.size),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")
    return device


def _make_lpips_model(device: torch.device):
    import lpips

    model = lpips.LPIPS(net="alex")
    model.eval()
    model.to(device)
    return model


def _to_lpips_tensor(batch: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(batch, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.mul(2.0).sub(1.0)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _write_csv_header(path: Path, metrics: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", *metrics])
        writer.writeheader()


def _append_csv_rows(path: Path, rows: list[dict[str, float | str]], metrics: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", *metrics])
        writer.writerows(rows)


def evaluate_pair_metrics(
    reference_dir: Path,
    method_dir: Path,
    out: Path,
    metrics: list[str],
    batch_size: int,
    device_name: str,
    limit: int | None = None,
    strict: bool = True,
    save_per_image: bool = True,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    names = _paired_names(reference_dir, method_dir, strict=strict, limit=limit)
    if not names:
        raise RuntimeError("No paired PNG filenames found.")
    device = _device(device_name)
    lpips_model = _make_lpips_model(device) if "lpips" in metrics else None
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    per_image_path = out.with_suffix(".per_image.csv")
    if save_per_image:
        _write_csv_header(per_image_path, metrics)

    start = time.perf_counter()
    for start_idx in range(0, len(names), batch_size):
        chunk = names[start_idx : start_idx + batch_size]
        ref_batch = [_read_rgb(reference_dir / name) for name in chunk]
        pred_batch = [_read_rgb(method_dir / name) for name in chunk]
        rows: list[dict[str, float | str]] = []
        for name, ref, pred in zip(chunk, ref_batch, pred_batch):
            if ref.shape != pred.shape:
                raise RuntimeError(f"Image shape mismatch for {name}: {ref.shape} vs {pred.shape}")
            row: dict[str, float | str] = {"filename": name}
            if "psnr" in metrics:
                row["psnr"] = _psnr(ref, pred)
                values["psnr"].append(float(row["psnr"]))
            if "ssim" in metrics:
                row["ssim"] = _ssim(ref, pred)
                values["ssim"].append(float(row["ssim"]))
            if "rel_l2" in metrics:
                row["rel_l2"] = _rel_l2(ref, pred)
                values["rel_l2"].append(float(row["rel_l2"]))
            rows.append(row)
        if lpips_model is not None:
            with torch.no_grad():
                ref_tensor = _to_lpips_tensor(ref_batch, device)
                pred_tensor = _to_lpips_tensor(pred_batch, device)
                lpips_values = lpips_model(ref_tensor, pred_tensor).detach().flatten().cpu().numpy().astype(float)
            for row, value in zip(rows, lpips_values):
                row["lpips"] = float(value)
                values["lpips"].append(float(value))
        if save_per_image:
            _append_csv_rows(per_image_path, rows, metrics)
    elapsed = time.perf_counter() - start
    summary = {metric: asdict(_summarize(vals)) for metric, vals in values.items()}
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dir": str(reference_dir.resolve()),
        "method_dir": str(method_dir.resolve()),
        "num_pairs": len(names),
        "metrics": metrics,
        "device": str(device),
        "batch_size": batch_size,
        "limit": limit,
        "elapsed_sec": elapsed,
        "pairs_per_sec": len(names) / elapsed if elapsed > 0 else float("inf"),
        "summary": summary,
        "per_image_csv": str(per_image_path.resolve()) if save_per_image else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate full-reference same-seed pair metrics for Stage 4B.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", default="psnr,ssim,lpips,rel_l2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-strict", dest="strict", action="store_false", default=True)
    parser.add_argument("--no-per-image", dest="save_per_image", action="store_false", default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = _split_metrics(args.metrics)
    payload = evaluate_pair_metrics(
        reference_dir=args.reference_dir,
        method_dir=args.method_dir,
        out=args.out,
        metrics=metrics,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
        strict=args.strict,
        save_per_image=args.save_per_image,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
