#!/usr/bin/env python3
"""
Crop full face/body iBUG images to ear crops for easier piercing annotation.

For each image + .pts:
  1) YOLO tip-centered ear crop
  2) Remap landmarks into crop pixel space
  3) Save cropped image + remapped .pts (+ crop_meta.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.annotations import IMAGE_EXTS, parse_pts, pts_path_for, write_pts
from train.config import IBUG_ROOT, INPUT_SIZE
from train.crop import EarCropper, extract_crop, remap_points_to_crop


def process_one(
    img_path: Path,
    out_img: Path,
    cropper: EarCropper,
    out_size: int,
) -> dict:
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"unreadable: {img_path}")

    # Get crop meta without relying on 256-only path: call crop then rescale pts
    crop256, meta = cropper.crop(image)
    # Rebuild crop at desired out_size from meta (consistent remap)
    raw = extract_crop(image, meta.x0, meta.y0, meta.side, flip=meta.flipped)
    crop = cv2.resize(raw, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    pts_src = pts_path_for(img_path)
    pts = parse_pts(pts_src)
    remapped = []
    if pts:
        arr = np.asarray(pts, dtype=np.float32)
        remapped_arr = remap_points_to_crop(
            arr, meta.x0, meta.y0, meta.side, out_size=out_size, flipped=meta.flipped
        )
        remapped = remapped_arr.tolist()

    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), crop)
    if remapped:
        write_pts(out_img.with_suffix(".pts"), remapped)

    info = {
        "source_image": str(img_path),
        "source_pts": str(pts_src) if pts_src.is_file() else None,
        "crop": {
            "x0": meta.x0,
            "y0": meta.y0,
            "side": meta.side,
            "flipped": meta.flipped,
            "out_size": out_size,
        },
        "n_landmarks": len(remapped),
    }
    out_img.with_suffix(".crop_meta.json").write_text(json.dumps(info, indent=2))
    return info


def collect_images(src: Path) -> list[Path]:
    if not src.is_dir():
        return []
    # CollectionB: celebrity subfolders
    if src.name == "CollectionB":
        files: list[Path] = []
        for sub in sorted(src.iterdir()):
            if sub.is_dir():
                files.extend(
                    sorted(
                        p
                        for p in sub.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS and pts_path_for(p).is_file()
                    )
                )
        return files
    return sorted(
        p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS and pts_path_for(p).is_file()
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Batch ear-crop iBUG images for annotation")
    p.add_argument(
        "--out-root",
        default=str(ROOT / "data" / "data" / "ibug_crops"),
        help="Output root for cropped images + remapped .pts",
    )
    p.add_argument("--out-size", type=int, default=512, help="Saved crop size (square)")
    p.add_argument(
        "--sets",
        nargs="+",
        default=["collectiona_train", "collectiona_test"],
        choices=["collectiona_train", "collectiona_test", "collectionb", "all"],
        help="Which splits to process",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional max images (debug)")
    args = p.parse_args()

    sets = args.sets
    if "all" in sets:
        sets = ["collectiona_train", "collectiona_test", "collectionb"]

    mapping = {
        "collectiona_train": (
            IBUG_ROOT / "collectiona_1" / "CollectionA" / "train",
            Path(args.out_root) / "collectiona_train",
        ),
        "collectiona_test": (
            IBUG_ROOT / "collectiona_1" / "CollectionA" / "test",
            Path(args.out_root) / "collectiona_test",
        ),
        "collectionb": (
            IBUG_ROOT / "collectionb_1" / "CollectionB",
            Path(args.out_root) / "collectionb",
        ),
    }

    cropper = EarCropper()
    total_ok = total_fail = 0
    failures = []

    for key in sets:
        src, dst = mapping[key]
        images = collect_images(src)
        if args.limit:
            images = images[: args.limit]
        print(f"\n=== {key}: {len(images)} images → {dst} ===")
        dst.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(images, desc=key):
            # preserve relative name; for CollectionB prefix with person folder
            if key == "collectionb":
                out_name = f"{img_path.parent.name}__{img_path.name}"
            else:
                out_name = img_path.name
            out_img = dst / out_name
            try:
                process_one(img_path, out_img, cropper, args.out_size)
                total_ok += 1
            except Exception as exc:  # noqa: BLE001
                total_fail += 1
                failures.append({"image": str(img_path), "error": str(exc)})

    summary = {
        "ok": total_ok,
        "fail": total_fail,
        "out_size": args.out_size,
        "out_root": args.out_root,
        "failures": failures[:50],
    }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "crop_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. ok={total_ok} fail={total_fail}")
    print(f"Crops → {out_root}")
    print("Point Gradio at e.g. data/data/ibug_crops/collectiona_train")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
