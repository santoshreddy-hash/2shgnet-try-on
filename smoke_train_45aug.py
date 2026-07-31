#!/usr/bin/env python3
"""Smoke test: on-the-fly 45-aug dataset + short train path (no full training).

Default: --validate-only (build dataset, apply all 45 variants on 1 image,
run 1 forward+backward step). Does NOT run multi-epoch training.

Usage:
  python scripts/smoke_train_45aug.py
  python scripts/smoke_train_45aug.py --limit 2 --steps 1
  python scripts/smoke_train_45aug.py --run-train   # tiny 1-epoch smoke (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.config import DATA_IMAGES, resolve_pretrained_56
from train.dataset import Piercing56Dataset, discover_annotated
from train.metrics import heatmap_mse
from train.model import unfreeze_last_hourglass
from train.online_variants import VARIANTS_PER_IMAGE, apply_variant, variant_tags
from train.shgnet_base import select_device
from train.train import collate, load_checkpoint_into_56
from train.yolo_pose_labels import labels_dir_for_images


def validate_variants_math() -> None:
    tags = variant_tags()
    assert len(tags) == VARIANTS_PER_IMAGE == 45, (len(tags), VARIANTS_PER_IMAGE)
    assert len(set(tags)) == 45, "duplicate variant tags"


def validate_one_image_45(ds: Piercing56Dataset) -> None:
    assert len(ds.names) >= 1
    assert ds.variants_per_image == 45
    assert len(ds) == len(ds.names) * 45

    # Materialize all 45 variants for image 0
    shapes = []
    for v in range(45):
        sample = ds[v]
        assert sample["image"].ndim == 3 and sample["image"].shape[0] == 3
        assert sample["heatmaps"].shape[0] == 56
        assert sample["landmarks"].shape == (56, 2)
        shapes.append(tuple(sample["image"].shape))
        assert "__" in sample["name"] or sample["name"] == ds.names[0]
    assert len(set(shapes)) >= 1
    print(f"[ok] 45 variants for {ds.names[0]} image_tensor={shapes[0]}")


def validate_apply_variant_unit() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[80:180, 100:160] = 200
    pts = np.stack([np.linspace(110, 150, 56), np.linspace(90, 170, 56)], axis=1).astype(
        np.float32
    )
    tags_seen = []
    for vid in range(45):
        tag, out, aug_pts = apply_variant(img, pts, vid, seed=42)
        assert out.shape == img.shape
        assert aug_pts is not None and aug_pts.shape == (56, 2)
        tags_seen.append(tag)
    assert len(tags_seen) == 45
    print("[ok] apply_variant unit (45) on synthetic crop")


def one_train_step(ds: Piercing56Dataset, device: torch.device, batch_size: int) -> None:
    ckpt = resolve_pretrained_56()
    if ckpt is None:
        raise SystemExit("Missing SHGNet-56_final.pth — cannot smoke train step")
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    model, _ = load_checkpoint_into_56(ckpt, device)
    unfreeze_last_hourglass(model)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    x = batch["image"].to(device)
    y = batch["heatmaps"].to(device)
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    loss = heatmap_mse(pred, y)
    loss.backward()
    opt.step()
    print(f"[ok] 1 train step loss={float(loss.detach()):.5f} batch={x.shape} device={device}")


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke: 45-aug online + short train path")
    p.add_argument("--images-dir", default=str(DATA_IMAGES))
    p.add_argument("--limit", type=int, default=2, help="Max annotated images to use")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--validate-only",
        action="store_true",
        default=True,
        help="Only validate dataset/variants + 1 step (default)",
    )
    p.add_argument(
        "--run-train",
        action="store_true",
        help="Also run a tiny 1-epoch train via train.train (still short)",
    )
    p.add_argument("--steps", type=int, default=1, help="Unused reserved; use 1 step")
    args = p.parse_args()

    print(f"VARIANTS_PER_IMAGE={VARIANTS_PER_IMAGE}")
    validate_variants_math()
    validate_apply_variant_unit()

    img_dir = Path(args.images_dir)
    try:
        ann_dir = labels_dir_for_images(img_dir)
    except Exception:
        ann_dir = img_dir
    names = discover_annotated(img_dir, ann_dir)
    if not names:
        print(f"No annotated images under {img_dir}", file=sys.stderr)
        return 1
    names = names[: max(1, args.limit)]
    print(f"Using {len(names)} images × 45 = {len(names) * 45} smoke samples")

    ds = Piercing56Dataset(
        names,
        img_dir=img_dir,
        ann_dir=ann_dir,
        augment=False,
        fill_55_with_pretrained=False,
        variants_per_image=45,
    )
    validate_one_image_45(ds)

    device = select_device(args.device)
    one_train_step(ds, device, args.batch_size)

    if args.run_train:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "train.train",
            "--images-dir",
            str(img_dir),
            "--variants-per-image",
            "45",
            "--stage2-epochs",
            "1",
            "--stage3-epochs",
            "0",
            "--skip-stage3",
            "--batch-size",
            str(args.batch_size),
            "--device",
            str(device),
            "--run-name",
            "smoke_45aug",
        ]
        print("Running tiny train:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            return rc

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
