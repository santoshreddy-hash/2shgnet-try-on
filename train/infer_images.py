#!/usr/bin/env python3
"""Run SHGNet-56 inference on arbitrary images (no GT required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CKPT_DIR, INPUT_SIZE, PIERCING_INDEX
from train.crop import EarCropper, remap_points_to_full
from train.model import build_ldnet56
from train.shgnet_base import heatmaps_to_points, preprocess_ear_bgr, select_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch") or {}
    model = build_ldnet56(
        nstack=int(arch.get("nstack", 2)),
        layer=int(arch.get("layer", 4)),
        in_channel=int(arch.get("in_channel", 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def draw_on_full(img: np.ndarray, pts_full: np.ndarray) -> np.ndarray:
    vis = img.copy()
    for i in range(55):
        x, y = int(round(pts_full[i, 0])), int(round(pts_full[i, 1]))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 2, (0, 220, 255), -1)
    x, y = int(round(pts_full[PIERCING_INDEX, 0])), int(round(pts_full[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)
    cv2.putText(
        vis,
        f"piercing #56 ({x},{y})",
        (max(0, x + 12), max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )
    return vis


def draw_on_crop(crop: np.ndarray, pts: np.ndarray) -> np.ndarray:
    vis = crop.copy()
    for i in range(55):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        cv2.circle(vis, (x, y), 2, (0, 220, 255), -1)
    x, y = int(round(pts[PIERCING_INDEX, 0])), int(round(pts[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    return vis


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=str(ROOT / "data" / "test_images"))
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "SHGNet-56_final.pth"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "test_new"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    img_dir = Path(args.images)
    files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if img_dir.is_dir()
        else [img_dir]
    )
    if not files:
        print(f"No images in {img_dir}", file=sys.stderr)
        return 1

    device = select_device(args.device)
    model = load_ckpt(Path(args.checkpoint), device)
    print(f"Loaded {args.checkpoint} on {device}")
    print(f"Testing {len(files)} images")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cropper = EarCropper()
    results = []

    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"SKIP unreadable: {fp.name}")
            continue
        crop, meta = cropper.crop(img)
        tensor = preprocess_ear_bgr(crop, INPUT_SIZE).to(device)
        with torch.inference_mode():
            hm = model(tensor)
        pred = heatmaps_to_points(hm, INPUT_SIZE)
        if pred.ndim == 3:
            pred = pred[0]

        pts_full = remap_points_to_full(
            pred, meta.x0, meta.y0, meta.side, INPUT_SIZE, flipped=meta.flipped
        )
        pierce = pts_full[PIERCING_INDEX]

        stem = fp.name.split("-")[0] if "-" in fp.name else fp.stem
        # keep short id like train_0010
        short = fp.stem[:10] if fp.stem.startswith("train_") else fp.stem

        full_vis = draw_on_full(img, pts_full)
        crop_vis = draw_on_crop(crop, pred)
        cv2.imwrite(str(out_dir / f"{short}_full.png"), full_vis)
        cv2.imwrite(str(out_dir / f"{short}_crop.png"), crop_vis)

        entry = {
            "image": fp.name,
            "crop": {"x0": meta.x0, "y0": meta.y0, "side": meta.side, "flipped": meta.flipped},
            "piercing_full": [float(pierce[0]), float(pierce[1])],
            "piercing_crop": [float(pred[PIERCING_INDEX, 0]), float(pred[PIERCING_INDEX, 1])],
            "landmarks_56_full": pts_full.tolist(),
        }
        results.append(entry)
        (out_dir / f"{short}_pred.json").write_text(json.dumps(entry, indent=2))
        print(
            f"{fp.name[:40]:40s}  piercing=({pierce[0]:.1f},{pierce[1]:.1f})  "
            f"crop=({meta.x0},{meta.y0},{meta.side}) flip={meta.flipped}"
        )

    (out_dir / "predictions.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved visuals + JSON → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
