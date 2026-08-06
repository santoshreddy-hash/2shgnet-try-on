#!/usr/bin/env python3
"""Evaluate SHGNet-56 on pre-cropped ear images (e.g. ibug_crops/collectiona_test)."""

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

from train.annotations import parse_pts, write_pts
from train.config import CKPT_DIR, INPUT_SIZE, NUM_LANDMARKS_55, PIERCING_INDEX
from train.metrics import landmark_nme_55, nme, pck
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


def scale_pts(pts: np.ndarray, src_wh: tuple[int, int], dst: int) -> np.ndarray:
    """Scale landmark coords from (w,h) image space → square dst×dst."""
    w, h = src_wh
    out = pts.astype(np.float32).copy()
    out[:, 0] *= dst / float(w)
    out[:, 1] *= dst / float(h)
    return out


def draw_vis(
    crop: np.ndarray,
    pred_native: np.ndarray,
    gt_native: np.ndarray | None,
) -> np.ndarray:
    vis = crop.copy()
    for i in range(min(55, len(pred_native))):
        px, py = int(round(pred_native[i, 0])), int(round(pred_native[i, 1]))
        cv2.circle(vis, (px, py), 2, (0, 180, 255), -1)  # pred orange
        if gt_native is not None and i < len(gt_native):
            gx, gy = int(round(gt_native[i, 0])), int(round(gt_native[i, 1]))
            cv2.circle(vis, (gx, gy), 2, (0, 255, 0), -1)  # GT green
    # piercing prediction (no GT expected on test)
    x, y = int(round(pred_native[PIERCING_INDEX, 0])), int(round(pred_native[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(vis, (x, y), 8, (0, 0, 255), 2)
    label = f"#56 ({x},{y})"
    if gt_native is not None and len(gt_native) >= NUM_LANDMARKS_55:
        nme55 = landmark_nme_55(
            scale_pts(pred_native[:55], (crop.shape[1], crop.shape[0]), INPUT_SIZE),
            scale_pts(gt_native[:55], (crop.shape[1], crop.shape[0]), INPUT_SIZE),
        )
        label = f"nme55={nme55:.3f}  " + label
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def main() -> int:
    p = argparse.ArgumentParser(description="Eval SHGNet-56 on pre-cropped ears")
    p.add_argument(
        "--images",
        default=str(ROOT / "data" / "data" / "ibug_crops" / "collectiona_test"),
    )
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "SHGNet-56_final.pth"))
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "collectiona_test_results"),
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    img_dir = Path(args.images)
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images in {img_dir}", file=sys.stderr)
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        for name in ("best_stage3.pth", "best_stage2.pth", "best_stage1.pth"):
            alt = CKPT_DIR / name
            if alt.is_file():
                ckpt_path = alt
                break
        else:
            print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1

    device = select_device(args.device)
    model = load_ckpt(ckpt_path, device)
    print(f"Loaded {ckpt_path} on {device}")
    print(f"Testing {len(files)} crops from {img_dir}")

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    pts_dir = out_dir / "pts"
    json_dir = out_dir / "json"
    for d in (vis_dir, pts_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"SKIP unreadable: {fp.name}")
            continue
        h, w = img.shape[:2]

        tensor = preprocess_ear_bgr(img, INPUT_SIZE).to(device)
        with torch.inference_mode():
            hm = model(tensor)
        pred256 = heatmaps_to_points(hm, INPUT_SIZE)
        if pred256.ndim == 3:
            pred256 = pred256[0]
        pred_native = pred256.astype(np.float32).copy()
        pred_native[:, 0] *= w / float(INPUT_SIZE)
        pred_native[:, 1] *= h / float(INPUT_SIZE)

        gt_native = None
        pts_path = fp.with_suffix(".pts")
        metrics: dict = {}
        if pts_path.is_file():
            gt_list = parse_pts(pts_path)
            if len(gt_list) >= NUM_LANDMARKS_55:
                gt_native = np.asarray(gt_list[:NUM_LANDMARKS_55], dtype=np.float32)
                gt256 = gt_native.copy()
                gt256[:, 0] *= INPUT_SIZE / float(w)
                gt256[:, 1] *= INPUT_SIZE / float(h)
                metrics["nme55"] = landmark_nme_55(pred256[:55], gt256)
                metrics["nme55_diag"] = nme(pred256[:55], gt256)
                metrics.update(pck(pred256[:55], gt256))

        stem = fp.stem
        vis = draw_vis(img, pred_native, gt_native)
        cv2.imwrite(str(vis_dir / f"{stem}_pred.png"), vis)
        write_pts(pts_dir / f"{stem}.pts", pred_native.tolist())

        entry = {
            "image": fp.name,
            "size": [int(w), int(h)],
            "piercing_crop": [
                float(pred_native[PIERCING_INDEX, 0]),
                float(pred_native[PIERCING_INDEX, 1]),
            ],
            "landmarks_56": pred_native.tolist(),
            "metrics": metrics,
        }
        (json_dir / f"{stem}.json").write_text(json.dumps(entry, indent=2))
        rows.append(entry)

        m = metrics
        extra = (
            f" nme55={m['nme55']:.4f} pck@0.05={m.get('pck@0.05', float('nan')):.3f}"
            if m
            else " (no GT)"
        )
        print(
            f"{fp.name}: pierce=({pred_native[PIERCING_INDEX,0]:.1f},"
            f"{pred_native[PIERCING_INDEX,1]:.1f}){extra}"
        )

    summary = {
        "checkpoint": str(ckpt_path),
        "images_dir": str(img_dir),
        "n_images": len(rows),
        "out_dir": str(out_dir),
    }
    with_gt = [r for r in rows if r["metrics"]]
    if with_gt:
        summary["mean_nme55"] = float(np.mean([r["metrics"]["nme55"] for r in with_gt]))
        summary["mean_pck@0.05"] = float(
            np.mean([r["metrics"].get("pck@0.05", float("nan")) for r in with_gt])
        )
        summary["mean_pck@0.1"] = float(
            np.mean([r["metrics"].get("pck@0.1", float("nan")) for r in with_gt])
        )
        summary["n_with_gt55"] = len(with_gt)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Summary ===")
    print(f"Images: {len(rows)}")
    if with_gt:
        print(f"Mean NME (55):  {summary['mean_nme55']:.4f}")
        print(f"Mean PCK@0.05:  {summary['mean_pck@0.05']:.3f}")
        print(f"Mean PCK@0.1:   {summary['mean_pck@0.1']:.3f}")
    print(f"Results → {out_dir}")
    print("  vis/   overlays (green=GT 1-55, orange=pred, red=#56 piercing)")
    print("  pts/   predicted 56-point .pts in crop coords")
    print("  json/  per-image predictions + metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
