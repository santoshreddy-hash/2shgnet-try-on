#!/usr/bin/env python3
"""
Desktop live — jewellery-style SYNC pipeline so landmarks stick to the ear.

  webcam → mirror → YOLO tip (sparse) → tip-centered ear crop
        → SHGNet every frame → tip-relative One Euro → draw

Tip-relative hold: shape is (landmarks − tip). When the tip moves, the whole
cloud moves with it — points stay glued to the ear.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.landmark_stick import LandmarkStickTracker
from tracking.settings import load_one_euro_settings, make_landmark_filter, max_step_px
from train.config import (
    CAMERA_FPS,
    CAMERA_FPS_MAX,
    CAMERA_FPS_MIN,
    CROP_PAD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INPUT_SIZE,
    ONNX_EXPORT,
    OUTPUTS,
    PIERCING_INDEX,
    YOLO_ONNX,
)
from train.crop import (
    EarCropper,
    build_crop_meta,
    crop_with_meta,
    estimate_pinna_h,
    extract_square_crop,
    landmarks_ok,
    remap_points_to_full,
)
from train.shgnet_onnx import SHGNet56Onnx


def clamp_fps(v: float | int) -> int:
    return int(max(CAMERA_FPS_MIN, min(CAMERA_FPS_MAX, int(round(float(v))))))


def draw(img_bgr: np.ndarray, pts: np.ndarray, label: str = "") -> np.ndarray:
    vis = img_bgr.copy()
    for i in range(55):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 2, (0, 180, 255), -1)
    x, y = int(round(pts[PIERCING_INDEX, 0])), int(round(pts[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)
    text = f"#56 ({x},{y})"
    if label:
        text = f"{label} | {text}"
    cv2.putText(vis, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def open_camera(index: int, width: int, height: int, target_fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed camera {index}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def infer_once(
    landmarker: SHGNet56Onnx,
    frame: np.ndarray,
    cx: float,
    cy: float,
    side_len: float,
    flip: bool,
) -> tuple[np.ndarray, float, int, int, int]:
    crop_bgr, ox, oy, side_px = extract_square_crop(frame, cx, cy, side_len)
    if flip:
        crop_bgr = cv2.flip(crop_bgr, 1)
    crop256 = cv2.resize(crop_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    pred256, score = landmarker.predict_with_score(crop256)
    pts = remap_points_to_full(pred256, ox, oy, side_px, INPUT_SIZE, flipped=flip)
    return pts, float(score), ox, oy, side_px


_LK_WIN = (21, 21)
_LK_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)


def track_tip_lk(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    tip: Tuple[float, float],
) -> Tuple[Tuple[float, float], Optional[np.ndarray], bool]:
    """Move tip with Lucas–Kanade between YOLO refreshes. Returns (tip, gray, moved)."""
    pt = np.array([[[float(tip[0]), float(tip[1])]]], dtype=np.float32)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return tip, gray, False
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        pt,
        None,
        winSize=_LK_WIN,
        maxLevel=3,
        criteria=_LK_CRIT,
    )
    if nxt is None or st is None or int(st.reshape(-1)[0]) != 1:
        return tip, gray, False
    nx, ny = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
    dx, dy = nx - tip[0], ny - tip[1]
    step = float(np.hypot(dx, dy))
    if step > 28.0:
        s = 28.0 / step
        nx = tip[0] + dx * s
        ny = tip[1] + dy * s
    if step < 0.15:
        return tip, gray, False
    return (nx, ny), gray, True


def main() -> int:
    p = argparse.ArgumentParser(description="SHGNet-56 live (sync, tip-stick)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=min(960, FRAME_WIDTH))
    p.add_argument("--height", type=int, default=min(540, FRAME_HEIGHT))
    p.add_argument("--fps", type=int, default=CAMERA_FPS)
    p.add_argument("--yolo-every", type=int, default=2)
    p.add_argument("--onnx", default=str(ONNX_EXPORT))
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    target_fps = clamp_fps(args.fps)
    frame_interval = 1.0 / float(target_fps)
    mirror = not args.no_mirror

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"Missing {onnx_path}", file=sys.stderr)
        return 1
    if not Path(YOLO_ONNX).is_file():
        print(f"Missing {YOLO_ONNX}", file=sys.stderr)
        return 1

    oe = load_one_euro_settings()
    print("Pipeline (SYNC — SHG every frame, tip-relative stick):")
    print(f"  YOLO/{args.yolo_every} · SHG every frame · mirror={mirror}")
    print(
        f"  One Euro min={oe['min_cutoff']} β={oe['beta']} "
        f"rest={oe['rest_speed_px']} step≤{oe['max_step_px']}"
    )
    print("  Show a CLEAR SIDE PROFILE of one ear.")

    landmarker = SHGNet56Onnx(onnx_path)
    cropper = EarCropper(YOLO_ONNX)
    filt = make_landmark_filter()
    step_px = max(float(max_step_px()), 28.0)

    try:
        cap = open_camera(args.camera, args.width, args.height, target_fps)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(
        f"[Camera] → {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}@"
        f"{float(cap.get(cv2.CAP_PROP_FPS) or 0):.1f}"
    )

    win = "SHGNet-56 ONNX Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    save_dir = OUTPUTS / "desktop_captures"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_i = 0

    tip: Optional[Tuple[float, float]] = None
    side_name: Optional[str] = None  # LEFT / RIGHT
    geo: Optional[Tuple[float, float, float]] = None  # cx, cy, side_len
    raw_rel: Optional[np.ndarray] = None
    last_box = None
    last_score = 0.0
    first_lock = True
    lost = 0
    frame_idx = 0
    last_t = None
    prev_gray: Optional[np.ndarray] = None
    stick = LandmarkStickTracker(n_track=14)
    pts_gen = 0
    yolo_this = False

    try:
        while True:
            t0 = time.perf_counter()
            if last_t is not None:
                wait = frame_interval - (t0 - last_t)
                if wait > 0.0005:
                    time.sleep(wait)
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yolo_this = False

            # ── YOLO tip (sparse) ─────────────────────────────
            if frame_idx % max(1, args.yolo_every) == 0:
                crop_img, meta = cropper.crop(frame, allow_center_fallback=False)
                if meta is not None:
                    new_side = "LEFT" if meta.flipped else "RIGHT"
                    new_tip = (float(meta.tip_x), float(meta.tip_y))
                    if side_name is not None and new_side != side_name:
                        filt.reset()
                        stick.reset()
                        raw_rel = None
                        first_lock = True
                        geo = None
                        prev_gray = None
                    side_name, tip = new_side, new_tip
                    yolo_this = True
                    # Rebuild geo from cropper meta (already medial + pinna)
                    if geo is None:
                        geo = (float(meta.cx), float(meta.cy), float(meta.side_len or meta.side))
                    else:
                        a = 0.55
                        geo = (
                            (1 - a) * geo[0] + a * float(meta.cx),
                            (1 - a) * geo[1] + a * float(meta.cy),
                            (1 - a) * geo[2] + a * float(meta.side_len or meta.side),
                        )
                    lost = 0
                else:
                    lost += 1
                    if lost > 45:
                        tip = side_name = geo = None
                        raw_rel = None
                        first_lock = True
                        filt.reset()
                        stick.reset()
                        cropper.reset()
                        prev_gray = None

            # Between YOLO: LK tip so crop + cloud follow the ear
            if tip is not None and not yolo_this and prev_gray is not None:
                old_tip = tip
                tip, _, moved = track_tip_lk(prev_gray, gray, tip)
                if moved and geo is not None:
                    dx = tip[0] - old_tip[0]
                    dy = tip[1] - old_tip[1]
                    geo = (geo[0] + dx, geo[1] + dy, geo[2])

            prev_gray = gray

            # ── Tip-relative SHG + LK stick ───────────────────
            vis = frame
            pts_draw = None
            dt = float(
                np.clip(
                    (t0 - last_t) if last_t is not None else frame_interval,
                    1.0 / CAMERA_FPS_MAX,
                    1.0 / CAMERA_FPS_MIN,
                )
            )

            if tip is not None and side_name is not None and geo is not None:
                cx, cy, side_len = geo
                half = side_len * 0.5
                # Keep medial offset, but never let tip leave the crop
                if abs(tip[0] - cx) > half * 0.55 or abs(tip[1] - cy) > half * 0.55:
                    cx, cy = float(tip[0]), float(tip[1])
                    geo = (cx, cy, side_len)

                prefer_flip = side_name == "LEFT"
                # SHG every frame — this is what makes points stick
                pts, score, ox, oy, side_px = infer_once(
                    landmarker, frame, cx, cy, side_len, prefer_flip
                )
                ok = landmarks_ok(pts, tip, float(side_px))
                if first_lock or score < 0.12 or not ok:
                    pts2, sc2, _, _, _ = infer_once(
                        landmarker, frame, cx, cy, side_len, not prefer_flip
                    )
                    ok2 = landmarks_ok(pts2, tip, float(side_px))
                    if sc2 > score or (first_lock and ok2 and not ok):
                        pts, score, prefer_flip = pts2, sc2, (not prefer_flip)
                        ok = ok2

                last_score = score
                last_box = (
                    max(0, int(round(cx - side_px * 0.5))),
                    max(0, int(round(cy - side_px * 0.5))),
                    min(fw, int(round(cx + side_px * 0.5))),
                    min(fh, int(round(cy + side_px * 0.5))),
                )

                if ok or first_lock:
                    tip_arr = np.asarray(tip, dtype=np.float32).reshape(1, 2)
                    raw_rel = pts.astype(np.float32) - tip_arr
                    snap = first_lock
                    first_lock = False
                    pts_draw = filt.update_relative(
                        pts.astype(np.float32),
                        tip,
                        dt=dt,
                        side=side_name.lower(),
                        max_step_px=step_px,
                        snap=snap,
                    )
                    pts_gen += 1
                    stuck = stick.update(frame, pts_draw, pts_gen, last_box)
                    if stuck is not None:
                        pts_draw = stuck
                elif raw_rel is not None:
                    # Weak SHG — glue previous shape to tracked tip
                    tip_arr = np.asarray(tip, dtype=np.float32).reshape(1, 2)
                    abs_pts = raw_rel + tip_arr
                    pts_draw = filt.update_relative(
                        abs_pts,
                        tip,
                        dt=dt,
                        side=side_name.lower(),
                        max_step_px=step_px,
                        snap=True,
                    )
                    stuck = stick.update(frame, None, pts_gen, last_box)
                    if stuck is not None:
                        pts_draw = stuck
                else:
                    stuck = stick.update(frame, None, pts_gen, last_box)
                    if stuck is not None:
                        pts_draw = stuck

            t1 = time.perf_counter()
            inst = 1.0 / max(t1 - t0, 1e-6)
            last_t = t1

            if pts_draw is not None and tip is not None:
                hud = (
                    f"ONNX stick cam {inst:.0f}fps "
                    f"(set {target_fps}, lim {CAMERA_FPS_MIN}-{CAMERA_FPS_MAX})"
                )
                vis = draw(frame, pts_draw, hud)
                if last_box is not None:
                    cv2.rectangle(
                        vis,
                        (last_box[0], last_box[1]),
                        (last_box[2], last_box[3]),
                        (0, 255, 0),
                        2,
                    )
                cv2.circle(
                    vis,
                    (int(round(tip[0])), int(round(tip[1]))),
                    5,
                    (0, 140, 255),
                    -1,
                )
                pierce = pts_draw[PIERCING_INDEX]
                cv2.putText(
                    vis,
                    f"score={last_score:.3f} pierce=({pierce[0]:.0f},{pierce[1]:.0f}) "
                    f"pipe={(t1 - t0) * 1000:.0f}ms",
                    (8, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                )
            else:
                vis = frame.copy()
                cv2.putText(
                    vis,
                    f"Turn head — SIDE PROFILE · cam {inst:.0f}fps",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                filt.reset()
                stick.reset()
                tip = side_name = geo = None
                raw_rel = None
                first_lock = True
                prev_gray = None
                cropper.reset()
                print("Reset")
            if key == ord("s"):
                out = save_dir / f"capture_{save_i:04d}.png"
                cv2.imwrite(str(out), vis)
                print(f"Saved {out}")
                save_i += 1
            if key == ord("["):
                target_fps = clamp_fps(target_fps - 1)
                frame_interval = 1.0 / float(target_fps)
                cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                print(f"FPS → {target_fps}")
            if key == ord("]"):
                target_fps = clamp_fps(target_fps + 1)
                frame_interval = 1.0 / float(target_fps)
                cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                print(f"FPS → {target_fps}")
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
