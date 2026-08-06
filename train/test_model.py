#!/usr/bin/env python3
"""Test SHGNet-56 .pth on annotated ear_pose / iBUG images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CKPT_DIR, DATA_IMAGES, INPUT_SIZE, PIERCING_INDEX
from train.dataset import Piercing56Dataset, discover_annotated, train_val_split
from train.metrics import landmark_nme_55, nme, pck, piercing_point_error
from train.model import build_ldnet56
from train.shgnet_base import heatmaps_to_points, select_device
from train.config import SEED, VAL_SPLIT


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


def main() -> int:
    p = argparse.ArgumentParser(description="Test SHGNet-56 PyTorch checkpoint")
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "SHGNet-56_final.pth"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "test_vis"))
    p.add_argument("--device", default=None)
    p.add_argument("--val-only", action="store_true", help="Eval on held-out 15% split")
    p.add_argument("--limit", type=int, default=0, help="Max images (0=all)")
    args = p.parse_args()

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

    names = discover_annotated()
    if not names:
        print("No annotated images.", file=sys.stderr)
        return 1
    if args.val_only:
        _, names = train_val_split(names, VAL_SPLIT, SEED)
        print(f"Val split: {len(names)} images")
    if args.limit and args.limit > 0:
        names = names[: args.limit]

    ds = Piercing56Dataset(
        names,
        img_dir=DATA_IMAGES,
        augment=False,
        fill_55_with_pretrained=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pck_sums = {}
    for batch in loader:
        name = batch["name"][0]
        img_t = batch["image"].to(device)
        gt = batch["landmarks"][0].numpy()
        with torch.inference_mode():
            hm = model(img_t)
        pred = heatmaps_to_points(hm, INPUT_SIZE)
        if pred.ndim == 3:
            pred = pred[0]

        nme_all = nme(pred, gt)
        nme55 = landmark_nme_55(pred, gt)
        pierce_px, pierce_n = piercing_point_error(pred, gt)
        pk = pck(pred, gt)
        for k, v in pk.items():
            pck_sums[k] = pck_sums.get(k, 0.0) + float(v)
        rows.append(
            {
                "name": name,
                "nme": float(nme_all),
                "nme55": float(nme55),
                "pierce_px": float(pierce_px),
                "pierce_n": float(pierce_n),
                **{k: float(v) for k, v in pk.items()},
            }
        )

        # visualize
        crop = cv2.imread(str(DATA_IMAGES / name))
        if crop is not None:
            crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
            vis = crop.copy()
            for i in range(min(55, len(pred))):
                px, py = int(pred[i, 0]), int(pred[i, 1])
                gx, gy = int(gt[i, 0]), int(gt[i, 1])
                cv2.circle(vis, (px, py), 2, (0, 165, 255), -1)
                cv2.circle(vis, (gx, gy), 2, (0, 255, 0), -1)
            x, y = int(pred[PIERCING_INDEX, 0]), int(pred[PIERCING_INDEX, 1])
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            gx, gy = int(gt[PIERCING_INDEX, 0]), int(gt[PIERCING_INDEX, 1])
            cv2.circle(vis, (gx, gy), 6, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"nme55={nme55:.3f} pierce={pierce_px:.1f}px",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imwrite(str(out_dir / f"{Path(name).stem}_vis.png"), vis)

    n = len(rows)
    summary = {
        "n": n,
        "mean_nme": float(np.mean([r["nme"] for r in rows])),
        "mean_nme55": float(np.mean([r["nme55"] for r in rows])),
        "mean_pierce_px": float(np.mean([r["pierce_px"] for r in rows])),
        "mean_pierce_n": float(np.mean([r["pierce_n"] for r in rows])),
        **{k: float(v / n) for k, v in pck_sums.items()},
        "checkpoint": str(ckpt_path),
        "data": str(DATA_IMAGES),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote vis → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
