#!/usr/bin/env python3
"""Wire local assets into the canonical repo layout (hardlinks when possible).

Sources (auto-detected; override with flags):
  images:  <ROOT>/dataset annotated/datasetr annotated/images
           or --images / --pack
  labels:  pack/labels  OR  local_assets/labels.zip  (train/ + val/)
  ckpt:    models/shgnet/SHGNet-56_final.pth  OR  local_assets/SHGNet-56_pretrained_init.pth
  yolo:    real .onnx next to repo / under models/ (skips broken Mac symlinks)

Targets:
  data/data/ear_pose/images/{train,val}
  data/data/ear_pose/labels/{train,val}
  models/shgnet/SHGNet-56_final.pth
  outputs/checkpoints/SHGNet-56_final.pth
  models/yolo26n-pose.onnx  (and models/yolo/) when a real file is found

Examples:
  python scripts/wire_local_dataset.py
  python scripts/wire_local_dataset.py --images ".../images" --labels-zip local_assets/labels.zip
  python scripts/wire_local_dataset.py --skip-images   # ckpt + labels.zip only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "dataset annotated" / "datasetr annotated"
_LOCAL = ROOT / "local_assets"
DEFAULT_CKPT = next(
    (
        p
        for p in (
            ROOT / "models" / "shgnet" / "SHGNet-56_final.pth",
            _LOCAL / "SHGNet-56_pretrained_init.pth",
            ROOT / "SHGNet-56_final.pth",
        )
        if p.is_file() and p.stat().st_size > 1_000_000
    ),
    ROOT / "models" / "shgnet" / "SHGNet-56_final.pth",
)
DEFAULT_LABELS_ZIP = (
    _LOCAL / "labels.zip"
    if (_LOCAL / "labels.zip").is_file()
    else ROOT / "labels.zip"
)
EAR_POSE = ROOT / "data" / "data" / "ear_pose"
MODELS_SHG = ROOT / "models" / "shgnet"
MODELS_YOLO = ROOT / "models" / "yolo"
CKPT_DIR = ROOT / "outputs" / "checkpoints"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_WEIGHT_BYTES = 1_000_000


def _is_real_file(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _is_usable_weight(path: Path) -> bool:
    """True for a real weight file (not a broken / Mac absolute symlink stub)."""
    try:
        if path.is_symlink():
            return path.exists() and path.resolve().is_file() and path.stat().st_size >= MIN_WEIGHT_BYTES
        return path.is_file() and path.stat().st_size >= MIN_WEIGHT_BYTES
    except OSError:
        return False


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


def _label_index(labels_root: Path) -> dict[str, tuple[str, Path]]:
    """stem -> (split, path) from labels/{train,val}/*.txt or flat labels/*.txt."""
    index: dict[str, tuple[str, Path]] = {}
    if not labels_root.is_dir():
        return index
    for split in ("train", "val"):
        d = labels_root / split
        if d.is_dir():
            for lab in d.glob("*.txt"):
                index[lab.stem] = (split, lab)
    # Flat pack: labels/*.txt → split by name heuristic
    for lab in labels_root.glob("*.txt"):
        if lab.stem not in index:
            index[lab.stem] = (_split_for_name(lab.name), lab)
    return index


def wire_checkpoint(src: Path) -> bool:
    if not src.is_file():
        print(f"[skip] checkpoint not found: {src}")
        return False
    size = src.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        print(f"[warn] checkpoint looks like a stub ({size} bytes): {src}")
    ok = True
    for dst in (MODELS_SHG / "SHGNet-56_final.pth", CKPT_DIR / "SHGNet-56_final.pth"):
        how = _link_or_copy(src, dst)
        print(f"[ckpt] {how} → {dst} ({dst.stat().st_size} bytes)")
        if dst.stat().st_size < MIN_WEIGHT_BYTES:
            ok = False
    return ok


def wire_labels_zip(zip_path: Path) -> dict[str, int]:
    """Extract labels.zip → data/data/ear_pose/labels/{train,val}."""
    counts = {"train_lab": 0, "val_lab": 0}
    if not zip_path.is_file():
        print(f"[skip] labels zip not found: {zip_path}")
        return counts

    with tempfile.TemporaryDirectory(prefix="labels_zip_") as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        tmp_p = Path(tmp)
        # Find labels/ root inside the archive
        candidates = [tmp_p / "labels", tmp_p]
        labels_root = next((c for c in candidates if (c / "train").is_dir()), None)
        if labels_root is None:
            # maybe archive is already train/val at top
            if (tmp_p / "train").is_dir():
                labels_root = tmp_p
            else:
                found = list(tmp_p.rglob("train"))
                found = [p for p in found if p.is_dir() and "MACOSX" not in str(p)]
                labels_root = found[0].parent if found else None
        if labels_root is None:
            print(f"[err] could not find labels/train inside {zip_path}")
            return counts

        for split in ("train", "val"):
            src_dir = labels_root / split
            if not src_dir.is_dir():
                continue
            dst_dir = EAR_POSE / "labels" / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for lab in sorted(src_dir.glob("*.txt")):
                shutil.copy2(lab, dst_dir / lab.name)
                counts[f"{split}_lab"] += 1
    print(f"[labels.zip] → {EAR_POSE / 'labels'} {counts}")
    return counts


def wire_images(img_dir: Path, prefer_label_split: bool = True) -> dict[str, int]:
    """Link/copy images into ear_pose/images/{train,val} matching label split when possible."""
    counts = {"train_img": 0, "val_img": 0, "missing_lab": 0, "unmatched": 0}
    if not img_dir.is_dir():
        print(f"[skip] images folder not found: {img_dir}")
        return counts

    index = _label_index(EAR_POSE / "labels") if prefer_label_split else {}

    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.stem in index:
            split, _ = index[img.stem]
        else:
            split = _split_for_name(img.name)
            counts["missing_lab"] += 1
            if counts["missing_lab"] <= 5:
                print(f"[warn] no label yet for {img.name} → {split}/")

        dst_img = EAR_POSE / "images" / split / img.name
        _link_or_copy(img, dst_img)
        counts[f"{split}_img"] += 1

    print(f"[images] {img_dir} → {EAR_POSE / 'images'} {counts}")
    return counts


def wire_pack_labels(lab_dir: Path) -> dict[str, int]:
    """Wire flat or split labels from a pack folder."""
    counts = {"train_lab": 0, "val_lab": 0}
    if not lab_dir.is_dir():
        return counts
    index = _label_index(lab_dir)
    # Also handle nested labels/train|val under pack
    if not index and (lab_dir / "train").is_dir():
        index = _label_index(lab_dir)
    for stem, (split, lab) in index.items():
        dst = EAR_POSE / "labels" / split / f"{stem}.txt"
        _link_or_copy(lab, dst)
        counts[f"{split}_lab"] += 1
    if sum(counts.values()):
        print(f"[labels] {lab_dir} → {EAR_POSE / 'labels'} {counts}")
    return counts


def wire_yolo_candidates(extra: list[Path] | None = None) -> bool:
    """Find a real YOLO ONNX and place it under models/."""
    candidates = [
        ROOT / "yolo26n-pose.onnx",
        ROOT / "models" / "yolo26n-pose.onnx",
        ROOT / "models" / "yolo" / "yolo26n-pose.onnx",
        *(extra or []),
    ]
    src = next((p for p in candidates if _is_usable_weight(p)), None)
    if src is None:
        print("[skip] no real yolo26n-pose.onnx found (symlink stubs ignored)")
        return False
    for dst in (
        ROOT / "models" / "yolo26n-pose.onnx",
        MODELS_YOLO / "yolo26n-pose.onnx",
    ):
        if dst.resolve() == src.resolve():
            print(f"[yolo] already at {dst} ({dst.stat().st_size} bytes)")
            continue
        how = _link_or_copy(src, dst)
        print(f"[yolo] {how} → {dst} ({dst.stat().st_size} bytes)")
    return True


def readiness() -> dict:
    def count_imgs(split: str) -> int:
        d = EAR_POSE / "images" / split
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    def count_labs(split: str) -> int:
        d = EAR_POSE / "labels" / split
        if not d.is_dir():
            return 0
        return sum(1 for p in d.glob("*.txt"))

    ckpt = MODELS_SHG / "SHGNet-56_final.pth"
    yolo = ROOT / "models" / "yolo26n-pose.onnx"
    yolo_alt = MODELS_YOLO / "yolo26n-pose.onnx"
    report = {
        "train_images": count_imgs("train"),
        "val_images": count_imgs("val"),
        "train_labels": count_labs("train"),
        "val_labels": count_labs("val"),
        "checkpoint_ok": _is_usable_weight(ckpt),
        "checkpoint": str(ckpt),
        "yolo_ok": _is_usable_weight(yolo) or _is_usable_weight(yolo_alt),
    }
    # paired train stems
    img_stems = {
        p.stem
        for p in (EAR_POSE / "images" / "train").glob("*.*")
        if p.suffix.lower() in IMAGE_EXTS
    } if (EAR_POSE / "images" / "train").is_dir() else set()
    lab_stems = {
        p.stem for p in (EAR_POSE / "labels" / "train").glob("*.txt")
    } if (EAR_POSE / "labels" / "train").is_dir() else set()
    report["train_paired"] = len(img_stems & lab_stems)
    report["train_imgs_missing_lab"] = len(img_stems - lab_stems)
    report["train_labs_missing_img"] = len(lab_stems - img_stems)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Wire local dataset + SHGNet-56 weights")
    p.add_argument("--pack", default=str(DEFAULT_PACK), help="Flat pack with images/ + labels/")
    p.add_argument("--images", default=None, help="Images folder (overrides pack/images)")
    p.add_argument(
        "--labels-zip",
        default=str(DEFAULT_LABELS_ZIP),
        help="YOLO labels archive with train/ + val/ (default: labels.zip)",
    )
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to SHGNet-56_final.pth",
    )
    p.add_argument("--skip-images", action="store_true", help="Do not wire images")
    p.add_argument("--skip-labels-zip", action="store_true", help="Do not extract labels.zip")
    p.add_argument("--skip-checkpoint", action="store_true")
    args = p.parse_args()

    pack = Path(args.pack)
    img_dir = Path(args.images) if args.images else (pack / "images")
    pack_labels = pack / "labels"

    print(f"ROOT={ROOT}")

    # 1) Labels first so image split can follow label train/val
    zip_counts = {"train_lab": 0, "val_lab": 0}
    if not args.skip_labels_zip:
        zip_counts = wire_labels_zip(Path(args.labels_zip))
    if pack_labels.is_dir():
        # Pack labels override / supplement zip for matching stems
        wire_pack_labels(pack_labels)

    # 2) Checkpoint
    if not args.skip_checkpoint:
        wire_checkpoint(Path(args.checkpoint))

    # 3) Images
    if not args.skip_images:
        wire_images(img_dir)

    # 4) YOLO if present as a real file
    wire_yolo_candidates()

    report = readiness()
    print("--- readiness ---")
    print(
        f"train images/labels: {report['train_images']}/{report['train_labels']} "
        f"(paired={report['train_paired']})"
    )
    print(f"val   images/labels: {report['val_images']}/{report['val_labels']}")
    if report["train_imgs_missing_lab"]:
        print(f"[warn] train images without labels: {report['train_imgs_missing_lab']}")
    if report["train_labs_missing_img"]:
        print(f"[warn] train labels without images: {report['train_labs_missing_img']}")
    print(
        f"checkpoint: {'ok' if report['checkpoint_ok'] else 'MISSING'}  ({report['checkpoint']})"
    )
    print(f"yolo onnx:  {'ok' if report['yolo_ok'] else 'MISSING (live only)'}")

    ready_train = (
        report["train_paired"] > 0 and report["checkpoint_ok"]
    )
    if ready_train:
        print("READY to train →  python -m train.train --device cuda")
        return 0

    print("NOT ready to train yet. Still need:")
    if report["train_paired"] == 0:
        if report["train_labels"] == 0:
            print("  - labels under data/data/ear_pose/labels/train/  (or labels.zip)")
        if report["train_images"] == 0:
            print(
                "  - images under data/data/ear_pose/images/train/\n"
                "    e.g. --images \"D:\\\\try on proj\\\\2shgnet-try-on\\\\"
                "dataset annotated\\\\datasetr annotated\\\\images\""
            )
        elif report["train_labs_missing_img"] or report["train_imgs_missing_lab"]:
            print("  - matching image/label stems (check filenames)")
    if not report["checkpoint_ok"]:
        print("  - SHGNet-56_final.pth  (place at repo root or pass --checkpoint)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
