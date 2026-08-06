#!/usr/bin/env python3
"""
Desktop live — tip-stick landmarks at a hard 20–30 FPS band.

  webcam/video → (optional mirror) → YOLO tip → tip-centered ear crop
        → SHGNet-56 → tip-relative One Euro + LK stick → draw

Quality defaults: dense YOLO/SHG every frame, aspect-preserving resize,
flip retry on. Use --low-compute only on weak machines.
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
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INPUT_SIZE,
    LIVE_SHG_EVERY,
    LIVE_YOLO_EVERY,
    OUTPUTS,
    PIERCING_INDEX,
    resolve_onnx_export,
    resolve_yolo_onnx,
)
from train.crop import (
    EarCropper,
    extract_square_crop,
    landmarks_ok,
    pierce_quality,
    remap_points_to_full,
)
from train.shgnet_onnx import SHGNet56Onnx

_DEFAULT_VIDEO = Path(r"D:\try on proj\Recording 2026-08-03 122709.mp4")


def clamp_fps(v: float | int) -> int:
    return int(max(CAMERA_FPS_MIN, min(CAMERA_FPS_MAX, int(round(float(v))))))


def clamp_fps_f(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 0.0
    return float(max(CAMERA_FPS_MIN, min(CAMERA_FPS_MAX, v)))


def fit_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Downscale only, preserving aspect ratio (never stretch)."""
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h:
        return frame
    scale = min(max_w / float(w), max_h / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


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
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    last_err = ""
    # Probe nearby indices too — Windows camera order is often unstable.
    indices = [index] + [i for i in range(0, 6) if i != index]
    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                last_err = f"idx={idx} backend={backend} open failed"
                continue
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
            ok, _ = cap.read()
            if ok:
                if idx != index:
                    print(f"[Camera] opened index {idx} (requested {index})")
                return cap
            last_err = f"idx={idx} backend={backend} first frame failed"
            cap.release()
    raise RuntimeError(f"Failed camera {index} ({last_err})")


def open_camera_with_retry(
    index: int,
    width: int,
    height: int,
    target_fps: int,
    retries: int = 8,
    delay_s: float = 1.25,
) -> cv2.VideoCapture:
    """Retry while the user plugs the webcam back in."""
    last: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return open_camera(index, width, height, target_fps)
        except RuntimeError as e:
            last = e
            print(
                f"[Camera] attempt {attempt}/{retries} failed — "
                f"plug in / reconnect USB webcam, close Zoom/Teams/browser tabs using it…"
            )
            if attempt < retries:
                time.sleep(delay_s)
    assert last is not None
    raise last

def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
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


_LK_WIN = (31, 31)
_LK_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
_TIP_MAX_STEP = 48.0


def track_tip_lk(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    tip: Tuple[float, float],
) -> Tuple[Tuple[float, float], Optional[np.ndarray], bool]:
    pt = np.array([[[float(tip[0]), float(tip[1])]]], dtype=np.float32)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return tip, gray, False
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pt, None, winSize=_LK_WIN, maxLevel=4, criteria=_LK_CRIT
    )
    if nxt is None or st is None or int(st.reshape(-1)[0]) != 1:
        return tip, gray, False
    nx, ny = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
    dx, dy = nx - tip[0], ny - tip[1]
    step = float(np.hypot(dx, dy))
    if step > _TIP_MAX_STEP:
        s = _TIP_MAX_STEP / step
        nx = tip[0] + dx * s
        ny = tip[1] + dy * s
    if step < 0.08:
        return tip, gray, False
    return (nx, ny), gray, True


def main() -> int:
    p = argparse.ArgumentParser(description="SHGNet-56 live (20-30 FPS)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=FRAME_WIDTH)
    p.add_argument("--height", type=int, default=FRAME_HEIGHT)
    p.add_argument("--fps", type=int, default=CAMERA_FPS, help=f"Target FPS ({CAMERA_FPS_MIN}-{CAMERA_FPS_MAX})")
    p.add_argument("--yolo-every", type=int, default=LIVE_YOLO_EVERY)
    p.add_argument("--shg-every", type=int, default=LIVE_SHG_EVERY)
    p.add_argument("--onnx", default=str(resolve_onnx_export()))
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--mirror", action="store_true", help="Force mirror (default on for webcam, off for video)")
    p.add_argument("--video", default=None, help="Optional video file instead of webcam")
    p.add_argument(
        "--allow-video-fallback",
        action="store_true",
        help="If webcam fails, fall back to default recording",
    )
    p.add_argument("--camera-retries", type=int, default=10, help="Webcam open retries while plugging in")
    p.add_argument("--low-compute", action="store_true", help="20 FPS, smaller frames, sparse YOLO/SHG")
    args = p.parse_args()

    target_fps = clamp_fps(args.fps)
    yolo_every = max(1, int(args.yolo_every))
    shg_every = max(1, int(args.shg_every))
    width, height = int(args.width), int(args.height)
    skip_flip_retry = False
    adapt_cap = 3  # quality: don't sparsify past every-3
    if args.low_compute:
        target_fps = CAMERA_FPS_MIN
        yolo_every = max(yolo_every, 3)
        shg_every = max(shg_every, 3)
        width, height = min(width, 640), min(height, 360)
        skip_flip_retry = True
        adapt_cap = 6

    frame_interval = 1.0 / float(target_fps)

    onnx_path = Path(args.onnx)
    yolo_path = resolve_yolo_onnx()
    if not onnx_path.is_file() or onnx_path.stat().st_size < 1_000_000:
        print(f"Missing real ONNX: {onnx_path}", file=sys.stderr)
        return 1
    if not yolo_path.is_file() or yolo_path.stat().st_size < 1_000_000:
        print(f"Missing real YOLO ONNX: {yolo_path}", file=sys.stderr)
        return 1

    oe = load_one_euro_settings()
    print(f"Pipeline (FPS band {CAMERA_FPS_MIN}-{CAMERA_FPS_MAX}, tip-relative stick):")
    print(f"  target={target_fps} · fit≤{width}x{height} · YOLO/{yolo_every} · SHG/{shg_every}")
    print(
        f"  One Euro min={oe['min_cutoff']} beta={oe['beta']} "
        f"rest={oe['rest_speed_px']} step<={oe['max_step_px']}"
    )
    if args.low_compute:
        print("  low-compute mode ON")
    print("  Show a CLEAR SIDE PROFILE of one ear.  [ / ] = FPS · r = reset · s = save · q = quit")

    landmarker = SHGNet56Onnx(onnx_path)
    cropper = EarCropper(yolo_path)
    filt = make_landmark_filter()
    step_px = max(float(max_step_px()), 42.0)

    using_video = False
    video_path: Optional[Path] = Path(args.video) if args.video else None
    try:
        if video_path is not None:
            cap = open_video(video_path)
            using_video = True
            print(f"[Video] {video_path}")
        else:
            print("[Camera] opening live webcam…")
            cap = open_camera_with_retry(
                args.camera, width, height, target_fps, retries=max(1, int(args.camera_retries))
            )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print(
            "Windows reports no connected camera (Present=False). "
            "Plug in USB webcam (FINGERS / Brio / RealSense), then re-run.",
            file=sys.stderr,
        )
        if args.allow_video_fallback:
            fallback = video_path if video_path and video_path.is_file() else _DEFAULT_VIDEO
            if fallback.is_file():
                print(f"[Fallback] webcam unavailable → {fallback}")
                cap = open_video(fallback)
                using_video = True
                video_path = fallback
            else:
                return 1
        else:
            return 1
    # Webcam: mirror by default (selfie). Video: no mirror (true side). Overrides win.
    if args.mirror:
        mirror = True
    elif args.no_mirror:
        mirror = False
    else:
        mirror = not using_video
    print(f"  mirror={mirror} · source={'video' if using_video else 'camera'}")

    print(
        f"[Capture] -> {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}@"
        f"{float(cap.get(cv2.CAP_PROP_FPS) or 0):.1f}"
    )

    win = "SHGNet-56 ONNX Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    save_dir = OUTPUTS / "desktop_captures"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_i = 0

    tip: Optional[Tuple[float, float]] = None
    side_name: Optional[str] = None
    geo: Optional[Tuple[float, float, float]] = None
    raw_rel: Optional[np.ndarray] = None
    last_box = None
    last_score = 0.0
    first_lock = True
    lost = 0
    transferring = False  # hide landmarks during left↔right ear switch
    frame_idx = 0
    last_t = None
    prev_gray: Optional[np.ndarray] = None
    stick = LandmarkStickTracker(n_track=12 if args.low_compute else 20)
    pts_gen = 0
    yolo_this = False
    fps_ema = 0.0
    adapt_yolo = yolo_every
    adapt_shg = shg_every
    prefer_side: Optional[str] = None  # stick to first locked ear side

    try:
        while True:
            t_sched = time.perf_counter()
            if last_t is not None:
                wait = frame_interval - (t_sched - last_t)
                if wait > 0.0005:
                    time.sleep(wait)
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok or frame is None:
                if using_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Reset trackers on loop so lock re-acquires cleanly
                    filt.reset()
                    stick.reset()
                    tip = side_name = geo = None
                    raw_rel = None
                    first_lock = True
                    prev_gray = None
                    cropper.reset()
                    prefer_side = None
                    lost = 0
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                else:
                    break

            frame = fit_frame(frame, width, height)
            if mirror:
                frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yolo_this = False

            if frame_idx % max(1, adapt_yolo) == 0:
                side_pref = prefer_side.lower() if prefer_side else None
                _crop_img, meta = cropper.crop(
                    frame, prefer_side=side_pref, allow_center_fallback=False
                )
                if meta is not None:
                    new_side = "LEFT" if meta.flipped else "RIGHT"
                    new_tip = (float(meta.tip_x), float(meta.tip_y))
                    side_switched = side_name is not None and new_side != side_name
                    tip_jumped = False
                    if tip is not None and not side_switched:
                        tip_jumped = float(
                            np.hypot(new_tip[0] - tip[0], new_tip[1] - tip[1])
                        ) > max(48.0, float(meta.side_len or meta.side) * 0.35)
                    if side_switched or tip_jumped:
                        filt.reset()
                        stick.reset()
                        cropper.reset()
                        raw_rel = None
                        first_lock = True
                        transferring = True
                        geo = None
                        prev_gray = None
                        tip = side_name = None
                    side_name, tip = new_side, new_tip
                    prefer_side = new_side
                    yolo_this = True
                    if geo is None:
                        geo = (float(meta.cx), float(meta.cy), float(meta.side_len or meta.side))
                    else:
                        a = 0.85  # snap crop to YOLO tip quickly on head turns
                        geo = (
                            (1 - a) * geo[0] + a * float(meta.cx),
                            (1 - a) * geo[1] + a * float(meta.cy),
                            (1 - a) * geo[2] + a * float(meta.side_len or meta.side),
                        )
                    lost = 0
                else:
                    lost += 1
                    # Mid-turn miss — hide landmarks immediately (no face trail)
                    if lost > 1:
                        transferring = True
                        raw_rel = None
                    # Drop sticky lock quickly so the other ear can be acquired
                    if lost > 8:
                        tip = side_name = geo = None
                        raw_rel = None
                        first_lock = True
                        transferring = True
                        prefer_side = None
                        filt.reset()
                        stick.reset()
                        cropper.reset()
                        prev_gray = None
                        lost = 0

            if tip is not None and not yolo_this and prev_gray is not None and not transferring:
                old_tip = tip
                tip, _, moved = track_tip_lk(prev_gray, gray, tip)
                if moved and geo is not None:
                    geo = (geo[0] + (tip[0] - old_tip[0]), geo[1] + (tip[1] - old_tip[1]), geo[2])

            prev_gray = gray
            vis = frame
            pts_draw = None
            dt = float(
                np.clip(
                    (t0 - last_t) if last_t is not None else frame_interval,
                    1.0 / CAMERA_FPS_MAX,
                    1.0 / CAMERA_FPS_MIN,
                )
            )

            run_shg = frame_idx % max(1, adapt_shg) == 0
            # While transferring, only run SHG to re-lock — never tip-hold old shape onto the face
            if tip is not None and side_name is not None and geo is not None and (
                not transferring or run_shg or first_lock
            ):
                cx, cy, side_len = geo
                half = side_len * 0.5
                if abs(tip[0] - cx) > half * 0.30 or abs(tip[1] - cy) > half * 0.30:
                    cx = 0.25 * cx + 0.75 * float(tip[0])
                    cy = 0.25 * cy + 0.75 * float(tip[1])
                    geo = (cx, cy, side_len)

                prefer_flip = side_name == "LEFT"
                did_shg = False
                if run_shg or first_lock or raw_rel is None:
                    pts, score, _ox, _oy, side_px = infer_once(
                        landmarker, frame, cx, cy, side_len, prefer_flip
                    )
                    q = pierce_quality(pts, tip, float(side_px), score)
                    ok_lm = landmarks_ok(pts, tip, float(side_px))
                    if (first_lock or score < 0.18 or not ok_lm or q < 0.35) and not skip_flip_retry:
                        pts2, sc2, _, _, _ = infer_once(
                            landmarker, frame, cx, cy, side_len, not prefer_flip
                        )
                        q2 = pierce_quality(pts2, tip, float(side_px), sc2)
                        ok2 = landmarks_ok(pts2, tip, float(side_px))
                        if q2 > q + 0.02 or (ok2 and not ok_lm):
                            pts, score, prefer_flip, q, ok_lm = pts2, sc2, (not prefer_flip), q2, ok2
                    did_shg = True
                    last_score = score
                    last_box = (
                        max(0, int(round(cx - side_px * 0.5))),
                        max(0, int(round(cy - side_px * 0.5))),
                        min(fw, int(round(cx + side_px * 0.5))),
                        min(fh, int(round(cy + side_px * 0.5))),
                    )
                    if ok_lm or (first_lock and q >= 0.30):
                        tip_arr = np.asarray(tip, dtype=np.float32).reshape(1, 2)
                        snap = first_lock or transferring
                        first_lock = False
                        transferring = False
                        pts_draw = filt.update_relative(
                            pts.astype(np.float32), tip, dt=dt,
                            side=side_name.lower(), max_step_px=step_px, snap=snap,
                        )
                        # Tip-rigid shape cache — tip-hold adds tip with zero lag
                        raw_rel = pts_draw.astype(np.float32) - tip_arr
                        pts_gen += 1
                        stuck = stick.update(frame, pts_draw, pts_gen, last_box)
                        if stuck is not None:
                            # Re-anchor to tip so landmarks never leave the ear tip
                            raw_rel = stuck.astype(np.float32) - tip_arr
                            pts_draw = raw_rel + tip_arr
                            stick.abs_pts = pts_draw.copy()
                    elif raw_rel is not None and not transferring:
                        tip_arr = np.asarray(tip, dtype=np.float32).reshape(1, 2)
                        # Zero-lag tip-rigid glue
                        pts_draw = raw_rel + tip_arr
                        stuck = stick.update(frame, None, pts_gen, last_box)
                        if stuck is not None:
                            raw_rel = stuck.astype(np.float32) - tip_arr
                            pts_draw = raw_rel + tip_arr
                            stick.abs_pts = pts_draw.copy()

                if not did_shg and raw_rel is not None and not transferring:
                    tip_arr = np.asarray(tip, dtype=np.float32).reshape(1, 2)
                    pts_draw = raw_rel + tip_arr
                    stuck = stick.update(frame, None, pts_gen, last_box)
                    if stuck is not None:
                        raw_rel = stuck.astype(np.float32) - tip_arr
                        pts_draw = raw_rel + tip_arr
                        stick.abs_pts = pts_draw.copy()

            t1 = time.perf_counter()
            work_ms = (t1 - t0) * 1000.0
            budget_ms = frame_interval * 1000.0 * 0.95
            # Dense lock: YOLO + SHG every frame
            adapt_yolo = yolo_every
            adapt_shg = shg_every

            if last_t is not None:
                inst = 1.0 / max(t1 - last_t, 1e-6)
                fps_ema = clamp_fps_f(0.85 * fps_ema + 0.15 * inst) if fps_ema else clamp_fps_f(inst)
            last_t = t1
            shown = clamp_fps_f(fps_ema) if fps_ema else float(target_fps)

            if pts_draw is not None and tip is not None:
                hud = (
                    f"ONNX {shown:.0f}/{target_fps} fps "
                    f"Y/{adapt_yolo} S/{adapt_shg}"
                )
                vis = draw(frame, pts_draw, hud)
                if last_box is not None:
                    cv2.rectangle(
                        vis, (last_box[0], last_box[1]), (last_box[2], last_box[3]), (0, 255, 0), 2
                    )
                cv2.circle(vis, (int(round(tip[0])), int(round(tip[1]))), 5, (0, 140, 255), -1)
                pierce = pts_draw[PIERCING_INDEX]
                cv2.putText(
                    vis,
                    f"score={last_score:.3f} pierce=({pierce[0]:.0f},{pierce[1]:.0f}) "
                    f"pipe={work_ms:.0f}ms {side_name}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                )
            else:
                vis = frame.copy()
                msg = (
                    "Switching ear…"
                    if transferring
                    else f"Turn head — SIDE PROFILE · {shown:.0f}/{target_fps} fps"
                )
                cv2.putText(
                    vis, msg,
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
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
                prefer_side = None
                prev_gray = None
                cropper.reset()
                adapt_yolo, adapt_shg = yolo_every, shg_every
                print("Reset")
            if key == ord("s"):
                out = save_dir / f"capture_{save_i:04d}.png"
                cv2.imwrite(str(out), vis)
                print(f"Saved {out}")
                save_i += 1
            if key == ord("["):
                target_fps = clamp_fps(target_fps - 1)
                frame_interval = 1.0 / float(target_fps)
                if not using_video:
                    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                print(f"FPS -> {target_fps}")
            if key == ord("]"):
                target_fps = clamp_fps(target_fps + 1)
                frame_interval = 1.0 / float(target_fps)
                if not using_video:
                    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                print(f"FPS -> {target_fps}")
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
