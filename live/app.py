#!/usr/bin/env python3
"""
Local Gradio live demo — ONNX pipeline (YOLO ONNX + SHGNet-56.onnx + One Euro).

Prefer the browser WASM demo (no Gradio):
  cd web && npm install && npm start
  → http://127.0.0.1:8765

- One input at a time: Webcam OR Upload
- Webcam getUserMedia frameRate hard-limited to 20–30 FPS
- Shows true instantaneous FPS (no EMA / fake stable FPS)
- Auto-detect ear crop vs full image · One Euro always on
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.one_euro import OneEuroLandmarkFilter
from tracking.settings import load_one_euro_settings, make_landmark_filter, max_step_px
from train.config import (
    CAMERA_FPS,
    CAMERA_FPS_MAX,
    CAMERA_FPS_MIN,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INPUT_SIZE,
    PIERCING_INDEX,
    resolve_onnx_export,
    resolve_yolo_onnx,
)
from train.crop import EarCropper, crop_with_meta, landmarks_ok, remap_points_to_full
from train.shgnet_onnx import SHGNet56Onnx

_STATE: dict = {
    "landmarker": None,
    "cropper": None,
    "filt": None,
    "t_prev": None,
    "mode_key": None,
    "had_full_lock": False,
    "target_fps": float(CAMERA_FPS),
    "last_tick": None,
    "last_out": None,
    "last_status": "Waiting for input…",
}

_CROP_ASPECT_MIN = 0.85
_CROP_ASPECT_MAX = 1.15
_CROP_MAX_SIDE = 768


def _clamp_fps(v: float | int | None) -> float:
    if v is None:
        return float(CAMERA_FPS)
    return float(max(CAMERA_FPS_MIN, min(CAMERA_FPS_MAX, float(v))))


def _webcam_constraints(ideal_fps: float | int) -> dict:
    fps = int(_clamp_fps(ideal_fps))
    return {
        "facingMode": "user",
        "width": {"ideal": FRAME_WIDTH},
        "height": {"ideal": FRAME_HEIGHT},
        # Hard-limit the browser camera track itself
        "frameRate": {"ideal": fps, "min": CAMERA_FPS_MIN, "max": CAMERA_FPS_MAX},
    }


def _looks_like_ear_crop(img_bgr: np.ndarray) -> bool:
    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32:
        return False
    aspect = w / float(h)
    return _CROP_ASPECT_MIN <= aspect <= _CROP_ASPECT_MAX and max(h, w) <= _CROP_MAX_SIDE


def _make_filter() -> OneEuroLandmarkFilter:
    """Same One Euro as jewellery try-on (values from one_euro_settings.json)."""
    return make_landmark_filter()


def load_pipeline():
    if _STATE["landmarker"] is not None:
        return
    onnx_path = resolve_onnx_export()
    yolo_path = resolve_yolo_onnx()
    if not Path(onnx_path).is_file() or Path(onnx_path).stat().st_size < 1_000_000:
        raise FileNotFoundError(
            f"Missing real SHGNet-56 ONNX at {onnx_path} — place models/shgnet/SHGNet-56.onnx "
            "or run: python -m train.export_onnx"
        )
    _STATE["landmarker"] = SHGNet56Onnx(onnx_path)
    _STATE["cropper"] = EarCropper(yolo_path)
    _STATE["_onnx_name"] = Path(onnx_path).name
    _STATE["filt"] = _make_filter()
    _STATE["t_prev"] = None
    _STATE["mode_key"] = None
    _STATE["max_step"] = max_step_px()


def _ensure_filter(mode_key: str) -> Tuple[OneEuroLandmarkFilter, float, bool]:
    if _STATE["filt"] is None:
        _STATE["filt"] = _make_filter()
    filt: OneEuroLandmarkFilter = _STATE["filt"]
    snap = False
    if _STATE["mode_key"] != mode_key:
        filt.reset()
        _STATE["mode_key"] = mode_key
        _STATE["t_prev"] = None
        snap = True
    now = time.perf_counter()
    t_prev = _STATE["t_prev"]
    target = float(_STATE["target_fps"])
    if t_prev is None:
        dt = 1.0 / target
        snap = True
    else:
        dt = float(
            np.clip(now - t_prev, 1.0 / CAMERA_FPS_MAX, 1.0 / CAMERA_FPS_MIN)
        )
    _STATE["t_prev"] = now
    return filt, dt, snap


def _draw(img_bgr: np.ndarray, pts: np.ndarray, label: str = "") -> np.ndarray:
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
        text = f"{label}  |  {text}"
    cv2.putText(vis, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def predict(
    image_rgb: Optional[np.ndarray],
    target_fps: float | int | None = None,
) -> Tuple[Optional[np.ndarray], str]:
    if image_rgb is None:
        return None, "Waiting for input…"
    load_pipeline()

    target = _clamp_fps(target_fps if target_fps is not None else _STATE["target_fps"])
    _STATE["target_fps"] = target
    min_interval = 1.0 / target

    now = time.perf_counter()
    last_tick = _STATE["last_tick"]
    # Drop excess frames so effective input ≤ target (≤30)
    if last_tick is not None and (now - last_tick) < (min_interval * 0.92):
        return _STATE["last_out"], _STATE["last_status"]

    # True instantaneous FPS between accepted frames (no EMA)
    if last_tick is None:
        inst_fps = target
    else:
        inst_fps = 1.0 / max(now - last_tick, 1e-6)
    _STATE["last_tick"] = now

    landmarker: SHGNet56Onnx = _STATE["landmarker"]
    cropper: EarCropper = _STATE["cropper"]
    img_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)

    already_cropped = _looks_like_ear_crop(img_bgr)
    if already_cropped:
        crop = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        meta = None
        mode_key = "crop"
        path_label = "ear-crop"
        pred256, score = landmarker.predict_with_score(crop)
        h, w = img_bgr.shape[:2]
        pts_disp = pred256.astype(np.float32).copy()
        pts_disp[:, 0] *= w / float(INPUT_SIZE)
        pts_disp[:, 1] *= h / float(INPUT_SIZE)
        tip = (float(np.mean(pts_disp[:55, 0])), float(np.mean(pts_disp[:55, 1])))
        side_label = "crop"
        draw_on = img_bgr
    else:
        crop, meta = cropper.crop(img_bgr, allow_center_fallback=False)
        if meta is None:
            msg = "Turn head — clear SIDE PROFILE of one ear"
            vis = img_bgr.copy()
            cv2.putText(
                vis, msg, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            out_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            _STATE["last_out"] = out_rgb
            _STATE["last_status"] = msg
            return out_rgb, msg
        path_label = "YOLO"
        tip = (float(meta.tip_x), float(meta.tip_y))
        prefer = bool(meta.flipped)
        first_lock = _STATE.get("had_full_lock") is not True
        pred256, score = landmarker.predict_with_score(crop)
        pts_disp = remap_points_to_full(
            pred256, meta.x0, meta.y0, meta.side, INPUT_SIZE, flipped=prefer
        )
        ok = landmarks_ok(pts_disp, tip, float(meta.side))
        used = prefer
        if first_lock or score < 0.12 or not ok:
            crop2 = crop_with_meta(img_bgr, meta, flip=not prefer)
            pred2, sc2 = landmarker.predict_with_score(crop2)
            pts2 = remap_points_to_full(
                pred2, meta.x0, meta.y0, meta.side, INPUT_SIZE, flipped=(not prefer)
            )
            ok2 = landmarks_ok(pts2, tip, float(meta.side))
            if sc2 > score or (first_lock and ok2 and not ok):
                pred256, score, pts_disp, used = pred2, sc2, pts2, (not prefer)
        meta.flipped = used
        mode_key = f"full:{used}"
        side_label = "left" if used else "right"
        draw_on = img_bgr
        _STATE["had_full_lock"] = True

    filt, dt, snap = _ensure_filter(mode_key)
    step = float(_STATE.get("max_step") or max_step_px())
    pts_smooth = filt.update_relative(
        pts_disp,
        tip,
        dt=dt if not snap else (1.0 / target),
        side=side_label,
        max_step_px=step,
        snap=snap,
    )

    hud = (
        f"ONNX·{path_label} cam {inst_fps:.0f}fps "
        f"(set {target:.0f}, lim {CAMERA_FPS_MIN}-{CAMERA_FPS_MAX})"
    )
    vis = _draw(draw_on, pts_smooth, hud)
    if meta is not None:
        x1 = max(0, meta.x0)
        y1 = max(0, meta.y0)
        x2 = min(draw_on.shape[1], meta.x0 + meta.side)
        y2 = min(draw_on.shape[0], meta.y0 + meta.side)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pierce = pts_smooth[PIERCING_INDEX]
    status = (
        f"ONNX · {path_label} · cam {inst_fps:.0f}fps "
        f"(set {target:.0f}, lim {CAMERA_FPS_MIN}–{CAMERA_FPS_MAX}) · "
        f"score={score:.3f} · piercing=({pierce[0]:.1f},{pierce[1]:.1f})"
    )
    _STATE["last_out"] = out_rgb
    _STATE["last_status"] = status
    return out_rgb, status


def _cam_fps_head_js() -> str:
    """Patch getUserMedia + live tracks so browser webcam is 20–30 FPS."""
    return f"""
<script>
(function () {{
  const MIN = {CAMERA_FPS_MIN};
  const MAX = {CAMERA_FPS_MAX};
  window.__SHG_TARGET_FPS = window.__SHG_TARGET_FPS || {CAMERA_FPS};

  function clamp(v) {{
    v = Number(v);
    if (!Number.isFinite(v)) v = {CAMERA_FPS};
    return Math.max(MIN, Math.min(MAX, Math.round(v)));
  }}

  async function applyToStream(stream, fps) {{
    if (!stream) return;
    const track = stream.getVideoTracks && stream.getVideoTracks()[0];
    if (!track || !track.applyConstraints) return;
    const f = clamp(fps);
    try {{
      await track.applyConstraints({{ frameRate: {{ ideal: f, min: MIN, max: MAX }} }});
    }} catch (_) {{
      try {{
        await track.applyConstraints({{ frameRate: {{ ideal: f, max: MAX }} }});
      }} catch (__) {{}}
    }}
    try {{
      const s = track.getSettings ? track.getSettings() : {{}};
      console.log('[SHGNet cam] track frameRate=', s.frameRate, 'setpoint=', f);
    }} catch (_) {{}}
  }}

  window.__SHG_SET_CAM_FPS = async function (fps) {{
    window.__SHG_TARGET_FPS = clamp(fps);
    const vids = document.querySelectorAll('video');
    for (const v of vids) {{
      if (v.srcObject) await applyToStream(v.srcObject, window.__SHG_TARGET_FPS);
    }}
    return window.__SHG_TARGET_FPS;
  }};

  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia || md.__shgPatched) return;
  const orig = md.getUserMedia.bind(md);
  md.getUserMedia = async function (constraints) {{
    constraints = constraints ? {{ ...constraints }} : {{}};
    const want = clamp(window.__SHG_TARGET_FPS);
    let video = constraints.video;
    if (video === true || video == null) video = {{}};
    if (typeof video === 'object') {{
      video = {{ ...video }};
      video.frameRate = {{ ideal: want, min: MIN, max: MAX }};
      constraints.video = video;
    }}
    constraints.audio = false;
    const stream = await orig(constraints);
    await applyToStream(stream, want);
    return stream;
  }};
  md.__shgPatched = true;
}})();
</script>
"""


def build_demo() -> gr.Blocks:
    try:
        load_pipeline()
        oe = load_one_euro_settings()
        banner = (
            f"**ONNX pipeline** · `{_STATE.get('_onnx_name', 'SHGNet-56.onnx')}` + YOLO pose · "
            f"One Euro on (jewellery: min={oe['min_cutoff']} β={oe['beta']} "
            f"d={oe['d_cutoff']} rest={oe['rest_speed_px']} step≤{oe['max_step_px']}) · "
            f"**webcam FPS {CAMERA_FPS_MIN}–{CAMERA_FPS_MAX}**"
        )
    except Exception as e:
        banner = f"⚠️ {e}"

    # Never pull Gradio stream faster than max webcam FPS
    stream_every = 1.0 / float(CAMERA_FPS_MAX)

    css = """
    .live_input_hide img,
    .live_input_hide video,
    .live_input_hide .image-frame {
      max-height: 0 !important; height: 0 !important;
      overflow: hidden !important; opacity: 0 !important;
      pointer-events: none !important;
    }
    .live_input_hide .image-container {
      min-height: 56px !important; max-height: 80px !important;
    }
    #live_output img { max-height: 72vh; }
    """

    with gr.Blocks(
        title="SHGNet-56 ONNX Live",
        css=css,
        head=_cam_fps_head_js(),
    ) as demo:
        gr.Markdown(
            f"# SHGNet-56 ONNX Live\n{banner}\n\n"
            "Orange = 1–55 · **Red = piercing #56** · "
            "square ≤768px → skip YOLO."
        )
        source = gr.Radio(
            choices=["Webcam", "Upload"],
            value="Webcam",
            label="Input",
        )
        fps_slider = gr.Slider(
            minimum=CAMERA_FPS_MIN,
            maximum=CAMERA_FPS_MAX,
            value=CAMERA_FPS,
            step=1,
            label=f"Webcam FPS ({CAMERA_FPS_MIN}–{CAMERA_FPS_MAX})",
        )
        cam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            visible=True,
            show_label=False,
            elem_classes=["live_input_hide"],
            height=64,
            webcam_options=gr.WebcamOptions(
                mirror=True,
                constraints=_webcam_constraints(CAMERA_FPS),
            ),
        )
        upload = gr.Image(
            sources=["upload"],
            type="numpy",
            visible=False,
            show_label=False,
            elem_classes=["live_input_hide"],
            height=64,
        )
        out = gr.Image(label="Output", type="numpy", elem_id="live_output", height=560)
        status = gr.Textbox(label="Status", lines=1)

        def _switch_source(choice: str):
            if _STATE["filt"] is not None:
                _STATE["filt"].reset()
            _STATE["t_prev"] = None
            _STATE["mode_key"] = None
            _STATE["had_full_lock"] = False
            _STATE["last_tick"] = None
            _STATE["last_out"] = None
            _STATE["last_status"] = f"Switched to {choice} · One Euro reset"
            if _STATE.get("cropper") is not None:
                _STATE["cropper"].reset()
            is_cam = choice == "Webcam"
            return (
                gr.update(visible=is_cam),
                gr.update(visible=not is_cam),
                None,
                _STATE["last_status"],
            )

        def _on_fps(v: float):
            fps = _clamp_fps(v)
            _STATE["target_fps"] = fps
            _STATE["last_tick"] = None  # reset instant measurement
            return (
                f"Webcam FPS setpoint → {fps:.0f} "
                f"(hard limit {CAMERA_FPS_MIN}–{CAMERA_FPS_MAX})",
                gr.update(
                    webcam_options=gr.WebcamOptions(
                        mirror=True,
                        constraints=_webcam_constraints(fps),
                    )
                ),
            )

        source.change(_switch_source, inputs=[source], outputs=[cam, upload, out, status])
        fps_slider.change(
            _on_fps,
            inputs=[fps_slider],
            outputs=[status, cam],
            js="(v) => { if (window.__SHG_SET_CAM_FPS) window.__SHG_SET_CAM_FPS(v); return [v]; }",
        )
        cam.stream(
            predict,
            inputs=[cam, fps_slider],
            outputs=[out, status],
            time_limit=120,
            stream_every=stream_every,
        )
        upload.change(predict, inputs=[upload, fps_slider], outputs=[out, status])

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue().launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
