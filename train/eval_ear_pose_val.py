#!/usr/bin/env python3
"""Eval SHGNet-56 on ear_pose val images; write predicted #56 into YOLO labels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CKPT_DIR, INPUT_SIZE, NUM_LANDMARKS_55, PIERCING_INDEX
from train.metrics import landmark_nme_55, nme, pck
from train.model import build_ldnet56
from train.shgnet_base import heatmaps_to_points, preprocess_ear_bgr, select_device
from train.yolo_pose_labels import (
    IMAGE_EXTS,
    label_path_for,
    labels_dir_for_images,
    list_yolo_pose_images,
    read_yolo_pose,
    save_piercing_px,
)


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


def draw_vis(crop: np.ndarray, pred: np.ndarray, gt55: np.ndarray | None) -> np.ndarray:
    vis = crop.copy()
    for i in range(min(55, len(pred))):
        px, py = int(round(pred[i, 0])), int(round(pred[i, 1]))
        cv2.circle(vis, (px, py), 2, (0, 180, 255), -1)
        if gt55 is not None and i < len(gt55):
            gx, gy = int(round(gt55[i, 0])), int(round(gt55[i, 1]))
            cv2.circle(vis, (gx, gy), 2, (0, 255, 0), -1)
    x, y = int(round(pred[PIERCING_INDEX, 0])), int(round(pred[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(vis, (x, y), 8, (0, 0, 255), 2)
    label = f"#56 ({x},{y})"
    if gt55 is not None:
        # metrics in 256 space
        h, w = crop.shape[:2]
        pred256 = pred.copy()
        pred256[:, 0] *= INPUT_SIZE / float(w)
        pred256[:, 1] *= INPUT_SIZE / float(h)
        gt256 = gt55.copy()
        gt256[:, 0] *= INPUT_SIZE / float(w)
        gt256[:, 1] *= INPUT_SIZE / float(h)
        label = f"nme55={landmark_nme_55(pred256[:55], gt256):.3f}  " + label
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def main() -> int:
    p = argparse.ArgumentParser(description="Eval + write #56 on ear_pose val")
    p.add_argument(
        "--images",
        default=str(ROOT / "data" / "data" / "ear_pose" / "images" / "val"),
    )
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "SHGNet-56_final.pth"))
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "ear_pose_val_results"),
    )
    p.add_argument(
        "--write-labels",
        action="store_true",
        default=True,
        help="Write predicted piercing #56 into matching labels/*.txt",
    )
    p.add_argument("--no-write-labels", action="store_true")
    p.add_argument(
        "--mirror-labels",
        action="append",
        default=[],
        help="Also write #56 into this labels/val dir (repeatable)",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()
    write_labels = args.write_labels and not args.no_write_labels

    img_dir = Path(args.images)
    labels_dir = labels_dir_for_images(img_dir)
    files = list_yolo_pose_images(img_dir)
    if not files:
        files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images in {img_dir}", file=sys.stderr)
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    device = select_device(args.device)
    model = load_ckpt(ckpt_path, device)
    print(f"Loaded {ckpt_path} on {device}")
    print(f"Testing {len(files)} images from {img_dir}")
    print(f"Labels dir: {labels_dir}")

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    json_dir = out_dir / "json"
    for d in (vis_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    mirror_dirs = [Path(x) for x in args.mirror_labels]
    rows = []
    nme55s, pck05s = [], []

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
        pred = pred256.astype(np.float32).copy()
        pred[:, 0] *= w / float(INPUT_SIZE)
        pred[:, 1] *= h / float(INPUT_SIZE)

        gt55 = None
        metrics: dict = {}
        lp = label_path_for(fp, labels_dir)
        if lp.is_file():
            try:
                _, kpts = read_yolo_pose(lp)
                if len(kpts) >= NUM_LANDMARKS_55:
                    gt55 = np.zeros((NUM_LANDMARKS_55, 2), dtype=np.float32)
                    gt55[:, 0] = kpts[:55, 0] * w
                    gt55[:, 1] = kpts[:55, 1] * h
                    gt256 = gt55.copy()
                    gt256[:, 0] *= INPUT_SIZE / float(w)
                    gt256[:, 1] *= INPUT_SIZE / float(h)
                    metrics["nme55"] = float(landmark_nme_55(pred256[:55], gt256))
                    metrics["nme55_diag"] = float(nme(pred256[:55], gt256))
                    metrics.update({k: float(v) for k, v in pck(pred256[:55], gt256).items()})
                    nme55s.append(metrics["nme55"])
                    pck05s.append(metrics.get("pck@0.05", float("nan")))
            except ValueError:
                pass

        pierce = (float(pred[PIERCING_INDEX, 0]), float(pred[PIERCING_INDEX, 1]))
        written = []
        if write_labels:
            out_lp = save_piercing_px(fp, pierce[0], pierce[1], labels_dir)
            written.append(str(out_lp))
            for md in mirror_dirs:
                md.mkdir(parents=True, exist_ok=True)
                # copy base label if missing, then write piercing
                dst = md / f"{fp.stem}.txt"
                if not dst.is_file() and lp.is_file():
                    shutil.copy2(lp, dst)
                save_piercing_px(fp, pierce[0], pierce[1], md)
                written.append(str(dst))

        vis = draw_vis(img, pred, gt55)
        cv2.imwrite(str(vis_dir / f"{fp.stem}_pred.png"), vis)
        entry = {
            "image": fp.name,
            "size": [int(w), int(h)],
            "piercing_px": list(pierce),
            "landmarks_56_px": pred.tolist(),
            "metrics": metrics,
            "labels_written": written,
        }
        (json_dir / f"{fp.stem}.json").write_text(json.dumps(entry, indent=2))
        rows.append(entry)
        extra = f" nme55={metrics['nme55']:.4f}" if metrics else ""
        print(f"{fp.name}: pierce=({pierce[0]:.1f},{pierce[1]:.1f}){extra}")

    summary = {
        "checkpoint": str(ckpt_path),
        "images_dir": str(img_dir),
        "labels_dir": str(labels_dir),
        "n_images": len(rows),
        "wrote_landmark_56": write_labels,
        "out_dir": str(out_dir),
    }
    if nme55s:
        summary["mean_nme55"] = float(np.mean(nme55s))
        summary["mean_pck@0.05"] = float(np.nanmean(pck05s))
        summary["n_with_gt55"] = len(nme55s)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
