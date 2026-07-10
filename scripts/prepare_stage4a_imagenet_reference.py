#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_files(path: Path) -> list[Path]:
    return sorted(child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)


def _is_imagefolder(path: Path) -> bool:
    return any(child.is_dir() and _image_files(child) for child in path.iterdir()) if path.exists() else False


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or validate ImageNet reference images for Stage 4A FID.")
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/stage4a/fid_reference")
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--image-size", type=int, default=256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--symlink", action="store_true")
    mode.add_argument("--copy", action="store_true")
    mode.add_argument("--resize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split_dir = args.imagenet_root / args.split
    if not split_dir.exists():
        print(f"Missing ImageNet split directory: {split_dir}")
        return 2
    images = _image_files(split_dir)
    if args.num_images is not None:
        images = images[: args.num_images]
    mode = "copy" if args.copy else "resize" if args.resize else "symlink"
    out_dir = args.out_root / f"{args.split}_{args.image_size}_{len(images)}"
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "imagenet_root": str(args.imagenet_root.resolve()),
        "split_dir": str(split_dir.resolve()),
        "split": args.split,
        "imagefolder_style": _is_imagefolder(split_dir),
        "image_count": len(images),
        "out_dir": str(out_dir.resolve()),
        "mode": mode,
        "image_size": args.image_size,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, source in enumerate(images):
        target = out_dir / f"{idx:06d}{source.suffix.lower()}"
        if mode == "symlink":
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(source.resolve())
        elif mode == "copy":
            if not target.exists():
                shutil.copy2(source, target)
        else:
            from PIL import Image

            with Image.open(source) as image:
                image = image.convert("RGB").resize((args.image_size, args.image_size))
                image.save(target.with_suffix(".png"))
    _write_meta(out_dir / "reference_meta.json", meta)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
