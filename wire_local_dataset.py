#!/usr/bin/env python3
"""Wire local Windows assets into the canonical repo layout (hardlinks when possible).

Default sources (override with flags):
  images/labels: <ROOT>/dataset annotated/datasetr annotated/{images,labels}
  checkpoint:    <ROOT>/SHGNet-56_final.pth

Targets:
  data/data/ear_pose/images/{train,val}
  data/data/ear_pose/labels/{train,val}
  models/shgnet/SHGNet-56_final.pth
  outputs/checkpoints/SHGNet-56_final.pth

Example (PowerShell, from repo root):
  .\\.venv\\Scripts\\python.exe scripts\\wire_local_dataset.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "dataset annotated" / "datasetr annotated"
DEFAULT_CKPT = ROOT / "SHGNet-56_final.pth"
EAR_POSE = ROOT / "data" / "data" / "ear_pose"
MODELS_SHG = ROOT / "models" / "shgnet"
CKPT_DIR = ROOT / "outputs" / "checkpoints"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _split_for_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("ear_pose_val__") or name.startswith("ear_pose_val__"):
        return "val"
    return "train"


def wire_checkpoint(src: Path) -> None:
    if not src.is_file():
        print(f"[skip] checkpoint not found: {src}")
        return
    size = src.stat().st_size
    if size < 1_000_000:
        print(f"[warn] checkpoint looks like a stub ({size} bytes): {src}")
    for dst in (MODELS_SHG / "SHGNet-56_final.pth", CKPT_DIR / "SHGNet-56_final.pth"):
        how = _link_or_copy(src, dst)
        print(f"[ckpt] {how} → {dst} ({dst.stat().st_size} bytes)")


def wire_pack(pack: Path) -> dict:
    img_dir = pack / "images"
    lab_dir = pack / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"Missing images folder: {img_dir}")
    if not lab_dir.is_dir():
        raise SystemExit(f"Missing labels folder: {lab_dir}")

    counts = {
        "train_img": 0,
        "val_img": 0,
        "train_lab": 0,
        "val_lab": 0,
        "missing_lab": 0,
    }
    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        split = _split_for_name(img.name)
        dst_img = EAR_POSE / "images" / split / img.name
        _link_or_copy(img, dst_img)
        counts[f"{split}_img"] += 1

        lab = lab_dir / f"{img.stem}.txt"
        if lab.is_file():
            dst_lab = EAR_POSE / "labels" / split / f"{img.stem}.txt"
            _link_or_copy(lab, dst_lab)
            counts[f"{split}_lab"] += 1
        else:
            counts["missing_lab"] += 1
            if counts["missing_lab"] <= 5:
                print(f"[warn] no label for {img.name}")
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description="Wire local dataset + SHGNet-56.pth")
    p.add_argument(
        "--pack",
        default=str(DEFAULT_PACK),
        help="Flat pack with images/ + labels/",
    )
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to SHGNet-56_final.pth",
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        help="Only wire the checkpoint",
    )
    args = p.parse_args()

    print(f"ROOT={ROOT}")
    wire_checkpoint(Path(args.checkpoint))

    if not args.skip_images:
        counts = wire_pack(Path(args.pack))
        print("[data]", counts)

    # Quick readiness summary
    train_imgs = list((EAR_POSE / "images" / "train").glob("*.*"))
    train_imgs = [p for p in train_imgs if p.suffix.lower() in IMAGE_EXTS]
    ckpt = MODELS_SHG / "SHGNet-56_final.pth"
    print("--- readiness ---")
    print(f"train images: {len(train_imgs)}")
    print(
        f"checkpoint:   {ckpt} "
        f"({'ok' if ckpt.is_file() and ckpt.stat().st_size > 1_000_000 else 'MISSING'})"
    )
    print("Next: python -m train.train --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
