#!/usr/bin/env python3
"""
Desktop live — tip-stick landmarks with adaptive performance profiles.

  webcam/video → (optional mirror) → YOLO tip → tip-centered ear crop
        → SHGNet-56 → tip-relative One Euro + LK stick → draw

Profiles (performance_profiles.json):
  high / medium / low share frozen quality (640×360 infer, adaptive LEFT flip).
  Same landmark path; tip-hold keeps UI smooth. Auto detects tier at startup.
  Override with --performance high|medium|low
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from threading import Lock, Thread
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.landmark_stick import LandmarkStickTracker
from tracking.performance import DynamicScaler, PerfMode, resolve_profile
from tracking.settings import load_one_euro_settings, make_landmark_filter, max_step_px
from train.config import (
    CAMERA_FPS_MAX,
    CAMERA_FPS_MIN,
    INPUT_SIZE,
    OUTPUTS,
    PIERCING_INDEX,
    resolve_onnx_export,
    resolve_yolo_device,
    resolve_yolo_weights,
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


def clamp_fps(v: float | int, fps_max: int | None = None) -> int:
    hi = int(fps_max) if fps_max is not None else int(CAMERA_FPS_MAX)
    hi = max(int(CAMERA_FPS_MIN), min(int(CAMERA_FPS_MAX), hi))
    return int(max(CAMERA_FPS_MIN, min(hi, int(round(float(v))))))


def clamp_fps_f(v: float, fps_max: int | None = None) -> float:
    hi = float(fps_max) if fps_max is not None else float(CAMERA_FPS_MAX)
    hi = max(float(CAMERA_FPS_MIN), min(float(CAMERA_FPS_MAX), hi))
    if not np.isfinite(v) or v <= 0:
        return 0.0
    return float(max(CAMERA_FPS_MIN, min(hi, v)))


def fit_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Letterbox to exactly max_w×max_h (quality-frozen infer size)."""
    h, w = frame.shape[:2]
    tw, th = int(max_w), int(max_h)
    if tw < 2 or th < 2:
        return frame
    if w == tw and h == th:
        return frame
    scale = min(tw / float(w), th / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    if nw == tw and nh == th:
        return resized
    canvas = np.zeros((th, tw, 3), dtype=frame.dtype)
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def draw(
    img_bgr: np.ndarray,
    pts: np.ndarray,
    label: str = "",
    *,
    overlay_only: bool = False,
    scratch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw landmarks. overlay_only reuses scratch buffer to avoid full-frame alloc."""
    if overlay_only and scratch is not None and scratch.shape == img_bgr.shape:
        np.copyto(scratch, img_bgr)
        vis = scratch
    else:
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


class LatestFrameCamera:
    """Background capture so the infer loop never blocks on cap.read() (~25ms)."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._lock = Lock()
        self._frame: Optional[np.ndarray] = None
        self._ok = False
        self._running = True
        self._thread = Thread(target=self._loop, name="cam-latest", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._ok = True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            # View is fine — caller must not mutate across threads without copy.
            # Copy once for safety (960x540 BGR ≈ 1.5MB, ~1ms).
            return True, self._frame.copy()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


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
_TIP_MAX_STEP = 12.0  # clamp tip LK jitter (was 48 — caused landmark shake)


def track_tip_lk(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    tip: Tuple[float, float],
) -> Tuple[Tuple[float, float], Optional[np.ndarray], bool]:
    pt = np.array([[[float(tip[0]), float(tip[1])]]], dtype=np.float32)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return tip, gray, False
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pt, None, winSize=_LK_WIN, maxLevel=3, criteria=_LK_CRIT
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
    if step < 0.15:
        return tip, gray, False
    # Soft blend toward LK (kills single-frame tip noise)
    nx = 0.55 * tip[0] + 0.45 * nx
    ny = 0.55 * tip[1] + 0.45 * ny
    return (nx, ny), gray, True


def main() -> int:
    p = argparse.ArgumentParser(description="SHGNet-56 live with adaptive performance profiles")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=None, help="Override process width (else from profile)")
    p.add_argument("--height", type=int, default=None, help="Override process height (else from profile)")
    p.add_argument("--fps", type=int, default=None, help=f"Override target FPS ({CAMERA_FPS_MIN}-{CAMERA_FPS_MAX})")
    p.add_argument("--yolo-every", type=int, default=None)
    p.add_argument("--shg-every", type=int, default=None)
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
    p.add_argument(
        "--performance",
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Performance profile: auto, high, medium, or low (same landmark quality)",
    )
    p.add_argument(
        "--low-compute",
        action="store_true",
        help="Alias for --performance low",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Exit after N frames (0 = unlimited; useful for smoke tests)",
    )
    p.add_argument("--no-display", action="store_true", help="Skip OpenCV window (headless)")
    args = p.parse_args()

    mode: PerfMode = "low" if args.low_compute else args.performance  # type: ignore[assignment]
    profile, capability, dyn_cfg = resolve_profile(mode)

    fps_cap = int(getattr(profile, "fps_max", CAMERA_FPS_MAX) or CAMERA_FPS_MAX)
    target_fps = clamp_fps(
        args.fps if args.fps is not None else profile.target_fps, fps_max=fps_cap
    )
    yolo_every = max(1, int(args.yolo_every if args.yolo_every is not None else profile.yolo_every))
    shg_every = max(1, int(args.shg_every if args.shg_every is not None else profile.shg_every))
    cam_w = int(profile.camera_width)
    cam_h = int(profile.camera_height)
    width = int(args.width if args.width is not None else profile.process_width)
    height = int(args.height if args.height is not None else profile.process_height)

    # Canonical infer size from quality freeze (CLI width/height cannot diverge).
    infer_w = int(getattr(profile, "infer_width", width) or width)
    infer_h = int(getattr(profile, "infer_height", height) or height)
    width, height = infer_w, infer_h

    flip_raw = str(profile.flip_inference).lower()
    flip_locked = "adaptive" if flip_raw in ("off", "never", "disabled") else str(profile.flip_inference)

    # Apply CLI overrides onto the selected profile (throughput only).
    profile = replace(
        profile,
        target_fps=target_fps,
        yolo_every=yolo_every,
        shg_every=shg_every,
        process_width=width,
        process_height=height,
        infer_width=width,
        infer_height=height,
        fps_max=fps_cap,
        # Quality freeze — never disable flip retry or still-skip SHG on low.
        flip_inference=flip_locked,
        skip_shg_on_still=False,
        allow_resolution_scale=False,
    )

    flip_mode = str(profile.flip_inference).lower()
    skip_flip_retry = flip_mode in ("off", "never", "disabled")
    flip_always = flip_mode in ("always", "on")
    flip_thr = float(profile.flip_score_threshold)
    min_shg = float(profile.min_shg_score)
    still_px = float(profile.still_motion_px)
    yolo_lost_only = bool(profile.yolo_on_track_lost_only)
    skip_shg_still = False  # quality freeze: tip-hold, never skip SHG for "still"
    overlay_only = bool(profile.overlay_only_render)
    roi_tracking = bool(profile.roi_tracking)
    landmark_reuse = bool(profile.landmark_reuse)

    scaler = DynamicScaler(base=profile, cfg=dyn_cfg)
    frame_interval = 1.0 / float(target_fps)

    onnx_path = Path(args.onnx)
    yolo_path = resolve_yolo_weights(prefer_pt=True)
    yolo_device = resolve_yolo_device()
    if not onnx_path.is_file() or onnx_path.stat().st_size < 1_000_000:
        print(f"Missing real ONNX: {onnx_path}", file=sys.stderr)
        return 1
    if not yolo_path.is_file() or yolo_path.stat().st_size < 1_000_000:
        print(f"Missing YOLO weights: {yolo_path}", file=sys.stderr)
        return 1

    oe = load_one_euro_settings(profile.name)
    print(f"Device: {capability.platform}")
    print(
        f"  CPU={capability.cpu_cores} cores · RAM={capability.ram_gb:.1f}GB · "
        f"GPU_EP={capability.gpu_providers or ['none']} · score={capability.score:.0f}"
    )
    print(f"  auto→{capability.recommended} ({capability.detail})")
    print(
        f"Profile [{profile.name}] {profile.label} "
        f"(mode={mode}, FPS band {CAMERA_FPS_MIN}-{fps_cap}):"
    )
    print(
        f"  target={target_fps} · cam={cam_w}x{cam_h} · fit≤{width}x{height} · "
        f"YOLO/{yolo_every} · SHG/{shg_every} · flip={flip_mode}"
    )
    print(
        f"  ROI={roi_tracking} reuse={landmark_reuse} yolo_lost_only={yolo_lost_only} "
        f"still_skip={skip_shg_still} overlay={overlay_only} dynamic={dyn_cfg.enabled}"
    )
    print(f"  YOLO={yolo_path.name} device={yolo_device}")
    print(
        f"  One Euro [{oe.get('profile', profile.name)}] "
        f"min={oe['min_cutoff']} beta={oe['beta']} d={oe['d_cutoff']} "
        f"rest={oe['rest_speed_px']} hold={oe['rest_hold_frames']} "
        f"release×{oe['rest_release_mult']} step<={oe['max_step_px']}"
    )
    print("  Show an ear (any angle with a visible tip).  [ / ] = FPS · r = reset · s = save · q = quit")

    landmarker = SHGNet56Onnx(
        onnx_path,
        ort_opts=profile.onnx,
        reuse_buffers=bool(profile.reuse_preprocess_buffers),
    )
    yolo_imgsz = int(getattr(profile, "yolo_imgsz", 640) or 640)
    cropper = EarCropper(yolo_path, device=yolo_device, imgsz=yolo_imgsz, conf=0.12)
    # Warm YOLO/SHG so first live frames aren't 100ms+ spikes
    try:
        warm = np.zeros((480, 640, 3), dtype=np.uint8)
        cropper.crop(warm, allow_center_fallback=True)
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        landmarker.predict_with_score(dummy)
        print("[Warmup] YOLO + SHG ready")
    except Exception as exc:  # noqa: BLE001
        print(f"[Warmup] skipped ({exc})")
    filt = make_landmark_filter(profile=profile.name)
    step_px = float(max_step_px(profile.name))

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
                args.camera, cam_w, cam_h, target_fps, retries=max(1, int(args.camera_retries))
            )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print(
            "No camera available. Plug in a webcam, or pass --video PATH / --allow-video-fallback.",
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

    # Webcam: async latest-frame reader (removes ~25ms blocking cap.read from the hot path)
    stream: cv2.VideoCapture | LatestFrameCamera = cap
    if not using_video:
        stream = LatestFrameCamera(cap)
        print("  capture=threaded-latest-frame")
        # Wait briefly for first frame
        for _ in range(50):
            ok0, _f0 = stream.read()
            if ok0:
                break
            time.sleep(0.02)

    win = "SHGNet-56 ONNX Live"
    if not args.no_display:
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
    frame_idx = 0
    last_t = None
    prev_gray: Optional[np.ndarray] = None
    stick = LandmarkStickTracker(n_track=int(profile.stick_n_track))
    pts_gen = 0
    yolo_this = False
    fps_ema = 0.0
    prefer_side: Optional[str] = None
    tip_motion = 999.0
    overlay_scratch: Optional[np.ndarray] = None
    track_ok = False
    overrun_skip = 0  # after a heavy frame, next frame is landmarks-only

    try:
        while True:
            if args.max_frames and frame_idx >= int(args.max_frames):
                print(f"[Done] reached --max-frames={args.max_frames}")
                break

            t_loop = time.perf_counter()
            ok, frame = stream.read()
            if not ok or frame is None:
                if using_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    filt.reset()
                    stick.reset()
                    tip = side_name = geo = None
                    raw_rel = None
                    first_lock = True
                    prev_gray = None
                    cropper.reset()
                    prefer_side = None
                    lost = 0
                    track_ok = False
                    scaler.reset()
                    overrun_skip = 0
                    ok, frame = stream.read()
                    if not ok or frame is None:
                        break
                else:
                    # Threaded cam briefly empty — yield and retry
                    time.sleep(0.002)
                    continue

            t0 = time.perf_counter()
            proc_w, proc_h = scaler.process_size
            frame = fit_frame(frame, proc_w, proc_h)
            if mirror:
                frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            if overlay_only:
                if overlay_scratch is None or overlay_scratch.shape != frame.shape:
                    overlay_scratch = np.empty_like(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yolo_this = False

            light_frame = overrun_skip > 0
            if light_frame:
                overrun_skip -= 1

            need_yolo = False
            if not light_frame:
                if tip is None or geo is None:
                    # Searching: YOLO every 2nd frame (every-frame search → ~15fps from cam+YOLO)
                    need_yolo = frame_idx % 2 == 0
                elif first_lock and raw_rel is None:
                    need_yolo = frame_idx % 2 == 0
                elif last_score > 0 and last_score < float(profile.yolo_conf_drop):
                    need_yolo = frame_idx % max(1, scaler.adapt_yolo) == 0
                elif yolo_lost_only:
                    if lost > 2:
                        need_yolo = frame_idx % 2 == 0
                    elif frame_idx % max(1, scaler.adapt_yolo) == 0:
                        need_yolo = True
                else:
                    need_yolo = frame_idx % max(1, scaler.adapt_yolo) == 0

            if need_yolo:
                side_pref = prefer_side.lower() if prefer_side else None
                _crop_img, meta = cropper.crop(
                    frame, prefer_side=side_pref, allow_center_fallback=False
                )
                if meta is not None:
                    new_side = "LEFT" if meta.flipped else "RIGHT"
                    new_tip = (float(meta.tip_x), float(meta.tip_y))
                    if side_name is not None and new_side != side_name:
                        # L↔R switch: drop landmarks immediately (no face slide)
                        filt.reset()
                        stick.reset()
                        raw_rel = None
                        first_lock = True
                        geo = None
                        prev_gray = None
                        pts_draw = None
                        last_box = None
                    elif (
                        tip is not None
                        and raw_rel is not None
                        and geo is not None
                    ):
                        jump = float(
                            np.hypot(new_tip[0] - tip[0], new_tip[1] - tip[1])
                        )
                        lim = max(36.0, float(geo[2]) * 0.45)
                        if jump > lim:
                            filt.reset()
                            stick.reset()
                            raw_rel = None
                            first_lock = True
                            geo = None
                            prev_gray = None
                            pts_draw = None
                            last_box = None
                    side_name, tip = new_side, new_tip
                    prefer_side = new_side
                    yolo_this = True
                    track_ok = True
                    if geo is None:
                        geo = (float(meta.cx), float(meta.cy), float(meta.side_len or meta.side))
                    else:
                        # Slow geo blend — fast jumps shake the crop / landmarks
                        a = 0.35
                        geo = (
                            (1 - a) * geo[0] + a * float(meta.cx),
                            (1 - a) * geo[1] + a * float(meta.cy),
                            (1 - a) * geo[2] + a * float(meta.side_len or meta.side),
                        )
                    lost = 0
                else:
                    lost += 1
                    track_ok = False
                    # During head turn YOLO often fails — hide landmarks quickly
                    if lost > 2 and raw_rel is not None:
                        raw_rel = None
                        first_lock = True
                        filt.reset()
                        stick.reset()
                        pts_draw = None
                        last_box = None
                    if lost > 30:
                        tip = side_name = geo = None
                        raw_rel = None
                        first_lock = True
                        prefer_side = None
                        filt.reset()
                        stick.reset()
                        cropper.reset()
                        prev_gray = None

            if tip is not None and not yolo_this and prev_gray is not None and roi_tracking:
                old_tip = tip
                tip, _, moved = track_tip_lk(prev_gray, gray, tip)
                tip_motion = float(np.hypot(tip[0] - old_tip[0], tip[1] - old_tip[1]))
                # Large LK tip drift = turning head — drop lock, don't drag landmarks
                if raw_rel is not None and geo is not None and tip_motion > max(24.0, float(geo[2]) * 0.28):
                    raw_rel = None
                    first_lock = True
                    filt.reset()
                    stick.reset()
                    pts_draw = None
                    tip_motion = 999.0
                elif moved and geo is not None:
                    geo = (geo[0] + (tip[0] - old_tip[0]), geo[1] + (tip[1] - old_tip[1]), geo[2])
                    track_ok = True
                elif not moved and tip_motion < 0.08:
                    tip_motion = 0.0
            elif tip is not None and yolo_this:
                tip_motion = 999.0

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

            # Offset SHG when cadence equals YOLO so they never share the same slot
            # (Y/2 S/2 would otherwise only fire YOLO and never SHG).
            shg_n = max(1, int(scaler.adapt_shg))
            yolo_n = max(1, int(scaler.adapt_yolo))
            if shg_n == yolo_n and shg_n > 1:
                run_shg = frame_idx % shg_n == 1
            else:
                run_shg = frame_idx % shg_n == 0
            need_first_shg = bool(first_lock or raw_rel is None)
            if tip is not None and side_name is not None and geo is not None:
                cx, cy, side_len = geo
                half = side_len * 0.5
                if abs(tip[0] - cx) > half * 0.30 or abs(tip[1] - cy) > half * 0.30:
                    cx = 0.25 * cx + 0.75 * float(tip[0])
                    cy = 0.25 * cy + 0.75 * float(tip[1])
                    geo = (cx, cy, side_len)

                prefer_flip = side_name == "LEFT"
                did_shg = False
                ear_still = skip_shg_still and tip_motion < still_px and raw_rel is not None and not first_lock
                # Never stack YOLO+SHG same frame. Still allow first-lock SHG on a
                # light (overrun) frame so CPU-slow high profile can acquire.
                # LEFT: keep probing until landmarks_ok — never tip-hold a bad LEFT lock.
                left_unverified = prefer_flip and (
                    first_lock or raw_rel is None or last_score < min_shg
                )
                want_shg = (
                    (run_shg or need_first_shg or left_unverified)
                    and not ear_still
                    and not yolo_this
                    and (not light_frame or need_first_shg or left_unverified)
                )

                if want_shg:
                    pts, score, _ox, _oy, side_px = infer_once(
                        landmarker, frame, cx, cy, side_len, prefer_flip
                    )
                    q = pierce_quality(pts, tip, float(side_px), score)
                    ok_lm = landmarks_ok(pts, tip, float(side_px)) and score >= min_shg
                    do_flip = False
                    if not skip_flip_retry:
                        # LEFT must always compare both orientations (training
                        # convention). Never skip this on low devices.
                        if flip_always or prefer_flip:
                            do_flip = True
                        elif first_lock and (score < 0.35 or not ok_lm):
                            do_flip = True
                        elif score < 0.10 or (not ok_lm and score < flip_thr):
                            do_flip = True
                    elif prefer_flip:
                        # Even if flip_inference were "off", LEFT still compares.
                        do_flip = True
                    if do_flip:
                        pts2, sc2, _, _, _ = infer_once(
                            landmarker, frame, cx, cy, side_len, not prefer_flip
                        )
                        q2 = pierce_quality(pts2, tip, float(side_px), sc2)
                        ok2 = landmarks_ok(pts2, tip, float(side_px)) and sc2 >= min_shg
                        if q2 > q + 0.02 or (ok2 and not ok_lm) or (sc2 > score + 0.02):
                            pts, score, prefer_flip, q, ok_lm = pts2, sc2, (not prefer_flip), q2, ok2
                    did_shg = True
                    last_score = score
                    last_box = (
                        max(0, int(round(cx - side_px * 0.5))),
                        max(0, int(round(cy - side_px * 0.5))),
                        min(fw, int(round(cx + side_px * 0.5))),
                        min(fh, int(round(cy + side_px * 0.5))),
                    )
                    # Reject weak heatmaps — score~0.01 with geometric ok still draws junk
                    # LEFT: require landmarks_ok before accepting (no soft first_lock escape
                    # that can stick a mirrored-wrong LEFT pose on low devices).
                    if prefer_flip or side_name == "LEFT":
                        accept = score >= min_shg and ok_lm
                    else:
                        accept = score >= min_shg and (
                            ok_lm or (first_lock and q >= 0.30 and score >= min_shg)
                        )
                    if accept:
                        tip_arr = np.asarray(tip, dtype=np.float32).reshape(1, 2)
                        new_rel = pts.astype(np.float32) - tip_arr
                        # Light blend only — prefer fresh SHG so shape tracks ear
                        if raw_rel is not None and not first_lock:
                            new_rel = 0.20 * raw_rel + 0.80 * new_rel
                        snap = first_lock
                        first_lock = False
                        # One Euro on offsets only; compose with live tip (no tip lag)
                        raw_rel = filt.filter_offsets(
                            new_rel, dt, side=side_name.lower(), max_step_px=step_px, snap=snap
                        )
                        pts_draw = filt.compose(tip, raw_rel)
                        pts_gen += 1
                        stuck = stick.update(frame, pts_draw, pts_gen, last_box)
                        if stuck is not None:
                            pts_draw = stuck
                    elif side_name == "LEFT" and not ok_lm:
                        # Never tip-hold a failed LEFT orientation.
                        raw_rel = None
                        first_lock = True
                        lost += 1
                    elif score < min_shg:
                        # Soft fail: keep tip/ROI via LK; refresh YOLO on cadence only
                        lost += 1
                        if first_lock and lost > 12:
                            raw_rel = None
                            stick.reset()
                            filt.reset()
                            tip = side_name = geo = None
                            prefer_side = None
                            cropper.reset()
                            prev_gray = None
                            track_ok = False
                    elif landmark_reuse and raw_rel is not None and not first_lock:
                        # Tip-hold: rigid attach to latest tip (offsets already smoothed)
                        pts_draw = filt.compose(tip, raw_rel)

                if not did_shg and landmark_reuse and raw_rel is not None and not first_lock:
                    # Tip-hold only with a verified lock — never during L↔R reacquire
                    pts_draw = filt.compose(tip, raw_rel)
                    stuck = stick.update(frame, None, pts_gen, last_box)
                    if stuck is not None:
                        pts_draw = stuck

            # Build HUD before display (shown uses previous EMA; fine for overlay)
            hud_fps = float(fps_ema) if fps_ema else float(target_fps)
            hud_fps = float(max(1.0, min(float(fps_cap), hud_fps)))
            t1 = time.perf_counter()
            work_ms = (t1 - t0) * 1000.0

            if pts_draw is not None and tip is not None:
                hud = (
                    f"{profile.name.upper()} {hud_fps:.0f}/{target_fps}fps "
                    f"Y/{scaler.adapt_yolo} S/{scaler.adapt_shg} "
                    f"@{proc_w}x{proc_h}"
                )
                vis = draw(
                    frame, pts_draw, hud,
                    overlay_only=overlay_only, scratch=overlay_scratch,
                )
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
                if overlay_only and overlay_scratch is not None and overlay_scratch.shape == frame.shape:
                    np.copyto(overlay_scratch, frame)
                    vis = overlay_scratch
                else:
                    vis = frame.copy()
                cv2.putText(
                    vis,
                    f"Looking for ear… · {hud_fps:.0f}/{target_fps} fps [{profile.name}]",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )

            if not args.no_display:
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
                if frame_idx % 30 == 0:
                    print(
                        f"[{profile.name}] frame={frame_idx} fps≈"
                        f"{(fps_ema if fps_ema else target_fps):.1f} "
                        f"Y/{scaler.adapt_yolo} S/{scaler.adapt_shg} "
                        f"pipe={work_ms:.0f}ms score={last_score:.3f}"
                    )

            t_end = time.perf_counter()
            # Pace to exact deadline; sleep leaves ~4ms margin (macOS overshoot)
            deadline = t_loop + frame_interval
            remain = deadline - t_end
            if remain > 0.006:
                time.sleep(remain - 0.004)
            while time.perf_counter() < deadline:
                pass
            t_end = time.perf_counter()

            if last_t is not None:
                inst = 1.0 / max(t_end - last_t, 1e-6)
                if fps_ema:
                    fps_ema = 0.70 * fps_ema + 0.30 * float(inst)
                else:
                    fps_ema = float(inst)
                scaler.update(float(fps_ema), work_ms=float(work_ms))
            last_t = t_end
            # Don't starve first SHG lock with overrun skips on slow CPU.
            if work_ms > (frame_interval * 1000.0) and not (first_lock or raw_rel is None):
                overrun_skip = max(overrun_skip, 1)
            shown = float(fps_ema) if fps_ema else float(target_fps)
            shown = float(max(1.0, min(float(fps_cap), shown)))

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
                track_ok = False
                scaler.reset()
                print("Reset")
            if key == ord("s"):
                out = save_dir / f"capture_{save_i:04d}.png"
                cv2.imwrite(str(out), vis)
                print(f"Saved {out}")
                save_i += 1
            if key == ord("["):
                target_fps = clamp_fps(target_fps - 1, fps_max=fps_cap)
                frame_interval = 1.0 / float(target_fps)
                scaler.base = replace(scaler.base, target_fps=target_fps)
                if not using_video:
                    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                print(f"FPS -> {target_fps}")
            if key == ord("]"):
                target_fps = clamp_fps(target_fps + 1, fps_max=fps_cap)
                frame_interval = 1.0 / float(target_fps)
                scaler.base = replace(scaler.base, target_fps=target_fps)
                if not using_video:
                    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                print(f"FPS -> {target_fps}")
            frame_idx += 1
    finally:
        if isinstance(stream, LatestFrameCamera):
            stream.release()
        else:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
