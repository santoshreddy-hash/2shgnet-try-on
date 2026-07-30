#!/usr/bin/env python3
"""
Gradio piercing-point annotator (one-click).

Supports:
  A) iBUG-style: image + sibling .pts  → writes landmark #56 into .pts
  B) YOLO ear_pose: images/{train,val} + labels/{train,val}/*.txt
     → appends/overwrites kpt #56 (normalized x y v) in the matching .txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.annotations import (
    get_landmark_56 as get_landmark_56_pts,
    list_dataset_images,
    save_landmark_56_to_pts,
)
from train.config import DEFAULT_ANNOTATE_DIR
from train.yolo_pose_labels import (
    DEFAULT_EAR_POSE_TRAIN,
    get_piercing_px,
    is_yolo_pose_images_dir,
    label_path_for,
    labels_dir_for_images,
    list_yolo_pose_images,
    save_piercing_px,
)

EAR_POSE_VAL = ROOT / "data" / "data" / "ear_pose" / "images" / "val"


def _draw_marker(img_bgr: np.ndarray, xy: Optional[Tuple[float, float]], color=(0, 0, 255)) -> np.ndarray:
    out = img_bgr.copy()
    if xy is None:
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.drawMarker(out, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    cv2.circle(out, (x, y), 8, color, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _detect_mode(folder: Path) -> str:
    if is_yolo_pose_images_dir(folder):
        return "yolo"
    return "pts"


def _list_images(folder: Path, mode: str) -> List[Path]:
    if mode == "yolo":
        return list_yolo_pose_images(folder)
    return list_dataset_images(folder)


def _get_56(path: Path, mode: str, labels_dir: Optional[Path]) -> Optional[Tuple[float, float]]:
    if mode == "yolo":
        return get_piercing_px(path, labels_dir)
    return get_landmark_56_pts(path)


def _save_56(path: Path, x: float, y: float, mode: str, labels_dir: Optional[Path]) -> Path:
    if mode == "yolo":
        return save_piercing_px(path, x, y, labels_dir=labels_dir)
    return save_landmark_56_to_pts(path, x, y)


def _coords_file(path: Path, mode: str, labels_dir: Optional[Path]) -> Path:
    if mode == "yolo":
        return label_path_for(path, labels_dir)
    return path.with_suffix(".pts")


def _first_unannotated(paths: List[Path], mode: str, labels_dir: Optional[Path]) -> int:
    for i, p in enumerate(paths):
        if _get_56(p, mode, labels_dir) is None:
            return i
    return 0


def _status_text(state: Dict[str, Any]) -> str:
    paths: List[Path] = state.get("paths") or []
    idx = state.get("idx", 0)
    folder = state.get("folder", EAR_POSE_VAL)
    mode = state.get("mode", "pts")
    labels_dir = state.get("labels_dir")
    if not paths:
        return (
            f"No images found in `{folder}`.\n\n"
            "For ear_pose use: `.../ear_pose/images/val` (labels → `.../labels/val`).\n"
            "For iBUG crops use a folder with image + sibling `.pts`."
        )
    path = paths[idx]
    saved = _get_56(path, mode, labels_dir)
    done = sum(1 for p in paths if _get_56(p, mode, labels_dir) is not None)
    coords = _coords_file(path, mode, labels_dir)
    mode_label = "YOLO labels/*.txt (kpt #56)" if mode == "yolo" else "sibling .pts (#56)"
    return "\n\n".join(
        [
            f"**Folder:** `{folder}`",
            f"**Mode:** {mode_label}",
            f"**Labels dir:** `{labels_dir}`" if mode == "yolo" else "**Labels:** sibling `.pts`",
            f"**Image {idx + 1} / {len(paths)}:** `{path.name}`",
            f"**Coords file:** `{coords}`",
            f"**Landmark #56:** {f'({saved[0]:.1f}, {saved[1]:.1f}) px' if saved else '— click to set'}",
            f"**Progress:** {done}/{len(paths)} annotated",
            "**Tip:** one click saves #56 and jumps to the next image. Use ← Previous to re-click.",
        ]
    )


def _render(state: Dict[str, Any]):
    paths: List[Path] = state.get("paths") or []
    idx = state.get("idx", 0)
    mode = state.get("mode", "pts")
    labels_dir = state.get("labels_dir")
    if not paths:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return blank, _status_text(state)
    path = paths[idx]
    img = cv2.imread(str(path))
    if img is None:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return blank, _status_text(state)
    return _draw_marker(img, _get_56(path, mode, labels_dir)), _status_text(state)


def _apply_folder(folder: Path, state: Dict[str, Any], jump_unannotated: bool = True) -> None:
    mode = _detect_mode(folder)
    paths = _list_images(folder, mode)
    state["folder"] = folder
    state["mode"] = mode
    state["labels_dir"] = labels_dir_for_images(folder) if mode == "yolo" else None
    state["paths"] = paths
    if jump_unannotated and paths:
        state["idx"] = _first_unannotated(paths, mode, state["labels_dir"])
    elif paths:
        state["idx"] = min(int(state.get("idx", 0)), len(paths) - 1)
    else:
        state["idx"] = 0


def on_load_folder(folder_str: str, state: Dict[str, Any]):
    folder = Path(folder_str.strip() or str(EAR_POSE_VAL)).expanduser()
    _apply_folder(folder, state, jump_unannotated=True)
    mode = state["mode"]
    n = len(state["paths"] or [])
    extra = f" · labels `{state['labels_dir']}`" if mode == "yolo" else ""
    return (*_render(state), state, f"Loaded {n} images ({mode}){extra}")


def on_refresh(state: Dict[str, Any]):
    folder = Path(state.get("folder") or EAR_POSE_VAL)
    cur = int(state.get("idx", 0))
    _apply_folder(folder, state, jump_unannotated=False)
    if state["paths"]:
        state["idx"] = min(cur, len(state["paths"]) - 1)
    return (*_render(state), state)


def on_click(evt: gr.SelectData, state: Dict[str, Any]):
    """Click → save landmark #56 → auto next image."""
    paths: List[Path] = state.get("paths") or []
    mode = state.get("mode", "pts")
    labels_dir = state.get("labels_dir")
    if not paths:
        return (*_render(state), state, "No images loaded.")
    if not (isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2):
        return (*_render(state), state, "Invalid click.")

    x, y = float(evt.index[0]), float(evt.index[1])
    idx = state.get("idx", 0)
    path = paths[idx]
    try:
        out_path = _save_56(path, x, y, mode, labels_dir)
    except Exception as e:
        return (*_render(state), state, f"Save failed: {e}")

    next_idx = (idx + 1) % len(paths)
    state["idx"] = next_idx
    msg = (
        f"Saved #56=({x:.1f}, {y:.1f}) px → `{out_path}`  ·  "
        f"next: `{paths[next_idx].name}` ({next_idx + 1}/{len(paths)})"
    )
    return (*_render(state), state, msg)


def on_prev(state: Dict[str, Any]):
    paths = state.get("paths") or []
    if paths:
        state["idx"] = (state["idx"] - 1) % len(paths)
    return (*_render(state), state)


def on_next(state: Dict[str, Any]):
    paths = state.get("paths") or []
    if paths:
        state["idx"] = (state["idx"] + 1) % len(paths)
    return (*_render(state), state)


def on_jump_unannotated(state: Dict[str, Any]):
    paths = state.get("paths") or []
    mode = state.get("mode", "pts")
    labels_dir = state.get("labels_dir")
    if paths:
        state["idx"] = _first_unannotated(paths, mode, labels_dir)
    return (*_render(state), state, f"Jumped to image {state['idx'] + 1}")


def build_app(folder0: Optional[Path] = None) -> gr.Blocks:
    if folder0 is None:
        if EAR_POSE_VAL.is_dir():
            folder0 = EAR_POSE_VAL
        elif DEFAULT_EAR_POSE_TRAIN.is_dir():
            folder0 = DEFAULT_EAR_POSE_TRAIN
        else:
            folder0 = DEFAULT_ANNOTATE_DIR
    folder0 = Path(folder0)
    initial: Dict[str, Any] = {
        "folder": folder0,
        "paths": [],
        "idx": 0,
        "mode": "pts",
        "labels_dir": None,
    }
    _apply_folder(folder0, initial, jump_unannotated=True)

    with gr.Blocks(title="SHGNet-56 Piercing Annotator") as demo:
        gr.Markdown(
            """
# SHGNet-56 Piercing Annotator
**One click** = save piercing as **landmark #56** → **auto next** (updates that image's coords file).

- **ear_pose val (300 test):** `.../ear_pose/images/val` → `.../labels/val/<stem>.txt`
- **ear_pose train:** `.../ear_pose/images/train` → `.../labels/train/<stem>.txt`
- **iBUG crops:** folder with image + sibling `.pts`
"""
        )
        state = gr.State(initial)
        with gr.Row():
            folder_in = gr.Textbox(label="Image folder", value=str(folder0), scale=4)
            btn_load = gr.Button("Load folder", variant="primary")
            btn_refresh = gr.Button("Refresh")
            btn_jump = gr.Button("Jump to next unfinished")
        status = gr.Markdown(_status_text(initial))
        canvas = gr.Image(
            label="Click piercing point (auto-saves + next)",
            type="numpy",
            interactive=True,
            height=640,
        )
        with gr.Row():
            btn_prev = gr.Button("← Previous (re-annotate)")
            btn_next = gr.Button("Next → (skip)")
        log = gr.Textbox(label="Log", interactive=False)

        demo.load(on_refresh, inputs=[state], outputs=[canvas, status, state])
        btn_load.click(
            on_load_folder, inputs=[folder_in, state], outputs=[canvas, status, state, log]
        )
        btn_refresh.click(on_refresh, inputs=[state], outputs=[canvas, status, state])
        btn_jump.click(
            on_jump_unannotated, inputs=[state], outputs=[canvas, status, state, log]
        )
        canvas.select(on_click, inputs=[state], outputs=[canvas, status, state, log])
        btn_prev.click(on_prev, inputs=[state], outputs=[canvas, status, state])
        btn_next.click(on_next, inputs=[state], outputs=[canvas, status, state])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="SHGNet-56 piercing annotator")
    p.add_argument(
        "--folder",
        default="",
        help="Image folder (default: ear_pose images/val — 300 test images)",
    )
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    if args.folder.strip():
        folder = Path(args.folder.strip()).expanduser()
    elif EAR_POSE_VAL.is_dir():
        folder = EAR_POSE_VAL
    elif DEFAULT_EAR_POSE_TRAIN.is_dir():
        folder = DEFAULT_EAR_POSE_TRAIN
    else:
        folder = DEFAULT_ANNOTATE_DIR

    mode = _detect_mode(folder)
    n = len(_list_images(folder, mode))
    print(f"Folder: {folder}")
    print(f"Mode: {mode}  images: {n}")
    if mode == "yolo":
        print(f"Labels: {labels_dir_for_images(folder)}")
        print("Click piercing → overwrites #56 in matching labels/*.txt")
    print(f"Open http://127.0.0.1:{args.port}")
    build_app(folder).launch(server_name="127.0.0.1", server_port=int(args.port), share=False)


if __name__ == "__main__":
    main()
