"""YOLO-Pose label I/O for ear_pose (images/ + labels/).

Label line format (Ultralytics):
  class cx cy w h  x1 y1 v1  x2 y2 v2  ...  (coords normalized 0–1)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from train.config import NUM_LANDMARKS_55, NUM_LANDMARKS_56, PIERCING_INDEX

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EAR_POSE = Path(
    "/Users/santoshreddy/Documents/virtual try on/ear_landmark_live/data/ear_pose"
)
DEFAULT_EAR_POSE_TRAIN = DEFAULT_EAR_POSE / "images" / "train"


def labels_dir_for_images(img_dir: Path) -> Path:
    """Map .../images/train → .../labels/train (YOLO layout)."""
    img_dir = Path(img_dir).resolve()
    if img_dir.parent.name == "images":
        return img_dir.parent.parent / "labels" / img_dir.name
    # fallback: sibling labels/<same-name>
    return img_dir.parent / "labels" / img_dir.name


def label_path_for(image_path: Path, labels_dir: Optional[Path] = None) -> Path:
    image_path = Path(image_path)
    if labels_dir is None:
        labels_dir = labels_dir_for_images(image_path.parent)
    return Path(labels_dir) / f"{image_path.stem}.txt"


def is_yolo_pose_images_dir(img_dir: Path) -> bool:
    """True when folder looks like YOLO images split with a matching labels/."""
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        return False
    lab = labels_dir_for_images(img_dir)
    if not lab.is_dir():
        return False
    # at least one image+label pair
    for p in img_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS and (lab / f"{p.stem}.txt").is_file():
            return True
    return False


def parse_yolo_pose_line(line: str) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Returns (header5, kpts Nx3, raw_parts_after_header).
    header5 = [cls, cx, cy, w, h]
    """
    parts = [float(x) for x in line.strip().split()]
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label (need ≥5 floats): {line[:80]!r}")
    header = parts[:5]
    rest = parts[5:]
    if len(rest) % 3 != 0:
        raise ValueError(f"Keypoints not multiple of 3 (got {len(rest)}): {line[:80]!r}")
    kpts = np.array(rest, dtype=np.float64).reshape(-1, 3) if rest else np.zeros((0, 3))
    return header, kpts, np.array(rest, dtype=np.float64)


def read_yolo_pose(label_path: Path) -> Tuple[List[float], np.ndarray]:
    text = Path(label_path).read_text().strip()
    if not text:
        raise ValueError(f"Empty label: {label_path}")
    # use first non-empty line (single-object ear labels)
    line = next(l for l in text.splitlines() if l.strip())
    header, kpts, _ = parse_yolo_pose_line(line)
    return header, kpts


def write_yolo_pose(label_path: Path, header: List[float], kpts: np.ndarray) -> None:
    """Write one-object YOLO-pose line. kpts shape (N, 3)."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    bits = [
        str(int(header[0])),
        f"{header[1]:.6f}",
        f"{header[2]:.6f}",
        f"{header[3]:.6f}",
        f"{header[4]:.6f}",
    ]
    for x, y, v in kpts:
        bits.append(f"{float(x):.6f}")
        bits.append(f"{float(y):.6f}")
        bits.append(str(int(round(float(v)))))
    label_path.write_text(" ".join(bits) + "\n")


def get_piercing_px(
    image_path: Path, labels_dir: Optional[Path] = None
) -> Optional[Tuple[float, float]]:
    """Return piercing (#56) in pixel coords, or None if not annotated."""
    image_path = Path(image_path)
    lp = label_path_for(image_path, labels_dir)
    if not lp.is_file():
        return None
    try:
        _, kpts = read_yolo_pose(lp)
    except ValueError:
        return None
    if len(kpts) < NUM_LANDMARKS_56:
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    x = float(kpts[PIERCING_INDEX, 0]) * w
    y = float(kpts[PIERCING_INDEX, 1]) * h
    return x, y


def save_piercing_px(
    image_path: Path,
    x_px: float,
    y_px: float,
    labels_dir: Optional[Path] = None,
    visibility: int = 2,
) -> Path:
    """
    Append/overwrite landmark #56 in the matching labels/*.txt (normalized).
    Keeps class + bbox + landmarks 1–55 unchanged.
    """
    image_path = Path(image_path)
    lp = label_path_for(image_path, labels_dir)
    if not lp.is_file():
        raise FileNotFoundError(f"No label for {image_path.name}: expected {lp}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    h, w = img.shape[:2]
    xn = float(np.clip(x_px / max(w, 1), 0.0, 1.0))
    yn = float(np.clip(y_px / max(h, 1), 0.0, 1.0))

    header, kpts = read_yolo_pose(lp)
    if len(kpts) < NUM_LANDMARKS_55:
        # pad missing with zeros (should not happen on ear_pose)
        pad = np.zeros((NUM_LANDMARKS_55 - len(kpts), 3), dtype=np.float64)
        kpts = np.vstack([kpts, pad]) if len(kpts) else pad
    kpts55 = kpts[:NUM_LANDMARKS_55].copy()
    piercing = np.array([[xn, yn, float(visibility)]], dtype=np.float64)
    kpts56 = np.vstack([kpts55, piercing])
    write_yolo_pose(lp, header, kpts56)
    return lp


def list_yolo_pose_images(img_dir: Path) -> List[Path]:
    """Images in img_dir that have a matching label .txt."""
    img_dir = Path(img_dir)
    lab = labels_dir_for_images(img_dir)
    if not img_dir.is_dir() or not lab.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and (lab / f"{p.stem}.txt").is_file():
            out.append(p)
    return out
