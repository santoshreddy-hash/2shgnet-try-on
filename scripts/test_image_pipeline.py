#!/usr/bin/env python3
"""Step-by-step test of ear crop + SHGNet-56 on one image; save panels + montage."""

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

from train.config import CKPT_DIR, INPUT_SIZE, PIERCING_INDEX, resolve_pretrained_56
from train.crop import EarCropper, crop_with_meta, remap_points_to_full
from train.model import build_ldnet56
from train.shgnet_base import heatmaps_to_points, preprocess_ear_bgr, select_device


def ensure_yolo_pose(dest: Path) -> Path:
    """Prefer .pt pose weights (ORT CUDA EP often fails on Windows)."""
    import shutil

    candidates = [
        ROOT / "models" / "yolo" / "yolo11n-pose.pt",
        ROOT / "models" / "yolo11n-pose.pt",
        Path("yolo11n-pose.pt"),
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 100_000:
            if c.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c, dest)
                return dest
            return c

    dest.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    print(f"[yolo] downloading yolo11n-pose.pt → {dest}")
    YOLO("yolo11n-pose.pt")  # auto-download into cwd / Ultralytics cache
    for p in [
        Path("yolo11n-pose.pt"),
        Path.home() / ".cache" / "ultralytics" / "yolo11n-pose.pt",
        Path.home() / "AppData" / "Roaming" / "Ultralytics" / "yolo11n-pose.pt",
    ]:
        if p.is_file():
            shutil.copy2(p, dest)
            return dest
    return Path("yolo11n-pose.pt")


def load_shgnet(ckpt: Path, device: torch.device):
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    arch = blob.get("arch") or {}
    model = build_ldnet56(
        nstack=int(arch.get("nstack", 2)),
        layer=int(arch.get("layer", 4)),
        in_channel=int(arch.get("in_channel", 256)),
    )
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 480)) -> np.ndarray:
    """Resize to fit panel height, pad width, add title bar."""
    h, w = size[1], size[0]
    canvas = np.full((h, w, 3), 32, dtype=np.uint8)
    if img is None or img.size == 0:
        cv2.putText(canvas, "N/A", (w // 2 - 30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        src = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = min((w - 8) / src.shape[1], (h - 36) / src.shape[0])
        nw, nh = max(1, int(src.shape[1] * scale)), max(1, int(src.shape[0] * scale))
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = (w - nw) // 2
        y0 = 28 + (h - 28 - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    cv2.rectangle(canvas, (0, 0), (w - 1, 26), (50, 50, 50), -1)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def heatmap_overlay(crop_bgr: np.ndarray, hm: torch.Tensor) -> np.ndarray:
    """Overlay piercing heatmap (#56) on crop."""
    arr = hm.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    pierce_hm = arr[PIERCING_INDEX]
    pierce_hm = pierce_hm - pierce_hm.min()
    if pierce_hm.max() > 0:
        pierce_hm = pierce_hm / pierce_hm.max()
    heat = cv2.resize(pierce_hm, (crop_bgr.shape[1], crop_bgr.shape[0]))
    heat_u8 = (heat * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(crop_bgr, 0.45, color, 0.55, 0)


def draw_yolo(img: np.ndarray, cropper: EarCropper) -> tuple[np.ndarray, dict]:
    vis = img.copy()
    info: dict = {"yolo_loaded": cropper._yolo is not None}
    tip_pack = cropper.tip_only(img)
    if tip_pack is None:
        info["status"] = "no_tip_or_no_yolo"
        return vis, info
    tip, side_name, kp, box = tip_pack
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    for i in range(min(len(kp), 17)):
        x, y, c = float(kp[i][0]), float(kp[i][1]), float(kp[i][2])
        if c < 0.2:
            continue
        color = (0, 255, 255) if i in (3, 4) else (180, 180, 180)
        cv2.circle(vis, (int(x), int(y)), 3, color, -1)
    cv2.drawMarker(vis, (int(tip[0]), int(tip[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.putText(
        vis,
        f"ear tip {side_name} ({tip[0]:.0f},{tip[1]:.0f})",
        (int(tip[0]) + 8, max(20, int(tip[1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    info.update({"status": "ok", "side": side_name, "tip": [float(tip[0]), float(tip[1])], "box": box.tolist()})
    return vis, info


def draw_crop_box(img: np.ndarray, meta) -> np.ndarray:
    vis = img.copy()
    x0, y0, s = meta.x0, meta.y0, meta.side
    cv2.rectangle(vis, (x0, y0), (x0 + s, y0 + s), (255, 128, 0), 2)
    cv2.circle(vis, (int(meta.tip_x), int(meta.tip_y)), 6, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"crop {s}px flip={meta.flipped}",
        (max(0, x0), max(20, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 128, 0),
        2,
    )
    return vis


def draw_landmarks_crop(crop: np.ndarray, pts: np.ndarray) -> np.ndarray:
    vis = crop.copy()
    for i in range(55):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        cv2.circle(vis, (x, y), 2, (0, 220, 255), -1)
    x, y = int(round(pts[PIERCING_INDEX, 0])), int(round(pts[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(vis, (x, y), 8, (0, 0, 255), 2)
    cv2.putText(vis, f"#56 ({x},{y})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def draw_landmarks_full(img: np.ndarray, pts_full: np.ndarray) -> np.ndarray:
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "test_86306"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read {img_path}", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps: dict = {"image": str(img_path), "shape": list(img.shape)}

    # --- Step 0: original ---
    cv2.imwrite(str(out / "01_original.png"), img)
    print(f"[1/7] original  shape={img.shape}")

    # --- Step 1: YOLO ---
    yolo_w = ensure_yolo_pose(ROOT / "models" / "yolo" / "yolo11n-pose.pt")
    cropper = EarCropper(yolo_path=yolo_w)
    yolo_vis, yolo_info = draw_yolo(img, cropper)
    cv2.imwrite(str(out / "02_yolo_pose.png"), yolo_vis)
    steps["yolo"] = yolo_info
    print(f"[2/7] YOLO pose  status={yolo_info.get('status')} tip={yolo_info.get('tip')}")

    # --- Step 2–3: crop ---
    crop, meta = cropper.crop(img, allow_center_fallback=True)
    if crop is None or meta is None:
        print("Crop failed", file=sys.stderr)
        return 1
    box_vis = draw_crop_box(img, meta)
    cv2.imwrite(str(out / "03_crop_box.png"), box_vis)
    cv2.imwrite(str(out / "04_ear_crop_256.png"), crop)
    steps["crop"] = {
        "x0": meta.x0,
        "y0": meta.y0,
        "side": meta.side,
        "flipped": meta.flipped,
        "tip": [meta.tip_x, meta.tip_y],
        "fallback_center": yolo_info.get("status") != "ok",
    }
    print(
        f"[3/7] crop box  x0={meta.x0} y0={meta.y0} side={meta.side} flip={meta.flipped}"
    )
    print(f"[4/7] ear crop  {crop.shape}")

    # --- Step 4–5: SHGNet ---
    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_pretrained_56()
    if ckpt is None or not ckpt.is_file():
        ckpt = CKPT_DIR / "SHGNet-56_final.pth"
    device = select_device(args.device)
    model = load_shgnet(ckpt, device)
    steps["checkpoint"] = str(ckpt)
    steps["device"] = str(device)
    tensor = preprocess_ear_bgr(crop, INPUT_SIZE).to(device)
    with torch.inference_mode():
        hm = model(tensor)
    pred = heatmaps_to_points(hm, INPUT_SIZE)
    if pred.ndim == 3:
        pred = pred[0]
    heat_vis = heatmap_overlay(crop, hm)
    lm_crop = draw_landmarks_crop(crop, pred)
    cv2.imwrite(str(out / "05_piercing_heatmap.png"), heat_vis)
    cv2.imwrite(str(out / "06_landmarks_crop.png"), lm_crop)
    pierce_c = pred[PIERCING_INDEX]
    steps["piercing_crop"] = [float(pierce_c[0]), float(pierce_c[1])]
    print(f"[5/7] heatmap overlay  piercing_crop=({pierce_c[0]:.1f},{pierce_c[1]:.1f})")
    print(f"[6/7] landmarks on crop  #56=({pierce_c[0]:.1f},{pierce_c[1]:.1f})")

    # --- Step 6: remap to full ---
    pts_full = remap_points_to_full(
        pred, meta.x0, meta.y0, meta.side, INPUT_SIZE, flipped=meta.flipped
    )
    full_vis = draw_landmarks_full(img, pts_full)
    cv2.imwrite(str(out / "07_landmarks_full.png"), full_vis)
    pierce_f = pts_full[PIERCING_INDEX]
    steps["piercing_full"] = [float(pierce_f[0]), float(pierce_f[1])]
    steps["landmarks_56_full"] = np.asarray(pts_full, dtype=float).tolist()
    print(f"[7/7] remap full  piercing=({pierce_f[0]:.1f},{pierce_f[1]:.1f})")

    # --- Montage 2x4 ---
    panels = [
        panel(img, "1. Original"),
        panel(yolo_vis, "2. YOLO pose + ear tip"),
        panel(box_vis, "3. Ear crop box"),
        panel(crop, "4. Crop 256x256"),
        panel(heat_vis, "5. Piercing heatmap"),
        panel(lm_crop, "6. 56 landmarks (crop)"),
        panel(full_vis, "7. Full-frame result"),
        panel(
            np.full_like(img, 40),
            f"Pierce full ({pierce_f[0]:.0f},{pierce_f[1]:.0f})",
        ),
    ]
    # last panel: zoom around piercing on full
    zoom = full_vis.copy()
    cx, cy = int(pierce_f[0]), int(pierce_f[1])
    r = 120
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(zoom.shape[1], cx + r), min(zoom.shape[0], cy + r)
    zoom_crop = zoom[y1:y2, x1:x2]
    panels[7] = panel(zoom_crop, f"8. Zoom pierce ({cx},{cy})")

    row1 = np.hstack(panels[:4])
    row2 = np.hstack(panels[4:])
    montage = np.vstack([row1, row2])
    cv2.imwrite(str(out / "00_montage.png"), montage)

    (out / "results.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")
    print(f"\nSaved step panels + montage → {out}")
    print(f"Montage: {out / '00_montage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
