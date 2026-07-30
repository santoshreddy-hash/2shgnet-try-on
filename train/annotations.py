"""Annotation I/O: landmark #56 (piercing) written into the image's .pts file."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from train.config import (  # noqa: F401
    DATA_ANNOTATIONS,
    DATA_IMAGES,
    DATA_ROOT,
    DEFAULT_ANNOTATE_DIR,
    NUM_LANDMARKS_55,
    NUM_LANDMARKS_56,
    PIERCING_INDEX,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def pts_path_for(image_path: Path) -> Path:
    """Sibling .pts next to the image (train_0000.png → train_0000.pts)."""
    return image_path.with_suffix(".pts")


def parse_pts(path: Path) -> List[List[float]]:
    if not path.is_file():
        return []
    text = path.read_text()
    m = re.search(r"n_points:\s*(\d+)", text)
    n = int(m.group(1)) if m else 0
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        return []
    body = text[start + 1 : end]
    pts: List[List[float]] = []
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
    if n and len(pts) != n:
        # prefer actual parsed count
        pass
    return pts


def write_pts(path: Path, landmarks: List[List[float]]) -> None:
    """Write iBUG-style .pts with n_points = len(landmarks)."""
    lines = ["version: 1", f"n_points: {len(landmarks)}", "{"]
    for x, y in landmarks:
        lines.append(f"{float(x):.3f} {float(y):.3f}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def load_landmarks_from_pts(pts_path: Path) -> List[Optional[List[float]]]:
    raw = parse_pts(pts_path)
    out: List[Optional[List[float]]] = [None] * NUM_LANDMARKS_56
    for i, pt in enumerate(raw[:NUM_LANDMARKS_56]):
        out[i] = [float(pt[0]), float(pt[1])]
    return out


def get_landmark_56(image_path: Path) -> Optional[Tuple[float, float]]:
    pts = parse_pts(pts_path_for(image_path))
    if len(pts) >= NUM_LANDMARKS_56:
        return float(pts[PIERCING_INDEX][0]), float(pts[PIERCING_INDEX][1])
    return None


def has_landmark_56_path(image_path: Path) -> bool:
    return get_landmark_56(image_path) is not None


def save_landmark_56_to_pts(image_path: Path, x: float, y: float) -> Path:
    """
    Update the image's .pts file: keep landmarks 1–55, set/overwrite #56 = piercing.
    Creates .pts from empty 55 zeros only if missing (should not happen on iBUG).
    """
    image_path = Path(image_path)
    pts_path = pts_path_for(image_path)
    existing = parse_pts(pts_path)

    if len(existing) >= NUM_LANDMARKS_55:
        landmarks = [list(p) for p in existing[:NUM_LANDMARKS_55]]
    elif len(existing) > 0:
        landmarks = [list(p) for p in existing]
        while len(landmarks) < NUM_LANDMARKS_55:
            landmarks.append([0.0, 0.0])
        landmarks = landmarks[:NUM_LANDMARKS_55]
    else:
        landmarks = [[0.0, 0.0] for _ in range(NUM_LANDMARKS_55)]

    if len(existing) >= NUM_LANDMARKS_56:
        # replace 56th
        landmarks = [list(p) for p in existing[:NUM_LANDMARKS_55]]
        landmarks.append([float(x), float(y)])
    else:
        landmarks.append([float(x), float(y)])

    write_pts(pts_path, landmarks)
    return pts_path


# --- legacy JSON helpers (kept for older train paths) ---


def annotation_path_for(image_name: str, ann_dir: Path = DATA_ANNOTATIONS) -> Path:
    return ann_dir / f"{Path(image_name).stem}.json"


def empty_landmarks() -> List[Optional[List[float]]]:
    return [None] * NUM_LANDMARKS_56


def load_annotation(path: Path) -> Dict[str, Any]:
    """Load from JSON if present; else from sibling .pts (path may be .json or .pts)."""
    path = Path(path)
    if path.suffix.lower() == ".pts" or (path.suffix.lower() != ".json" and path.with_suffix(".pts").is_file()):
        pts_path = path if path.suffix.lower() == ".pts" else path.with_suffix(".pts")
        lms = load_landmarks_from_pts(pts_path)
        return {
            "image": pts_path.stem,
            "num_landmarks": NUM_LANDMARKS_56,
            "landmarks": lms,
            "pts_path": str(pts_path),
        }
    if not path.is_file():
        # try pts next to expected image stem
        pts_alt = path.with_suffix(".pts")
        if pts_alt.is_file():
            lms = load_landmarks_from_pts(pts_alt)
            return {
                "image": pts_alt.stem,
                "num_landmarks": NUM_LANDMARKS_56,
                "landmarks": lms,
                "pts_path": str(pts_alt),
            }
        return {
            "image": path.stem,
            "num_landmarks": NUM_LANDMARKS_56,
            "landmarks": empty_landmarks(),
        }
    import json

    data = json.loads(path.read_text())
    lms = data.get("landmarks") or empty_landmarks()
    while len(lms) < NUM_LANDMARKS_56:
        lms.append(None)
    data["landmarks"] = lms[:NUM_LANDMARKS_56]
    data["num_landmarks"] = NUM_LANDMARKS_56
    return data


def save_landmark_56(
    image_name: str,
    x: float,
    y: float,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    ann_dir: Path = DATA_ANNOTATIONS,
    image_path: Optional[Path] = None,
) -> Path:
    """Prefer writing into the image's .pts; fall back to JSON only if no image path."""
    if image_path is not None:
        return save_landmark_56_to_pts(Path(image_path), x, y)
    # resolve under DATA_IMAGES
    img = DATA_IMAGES / Path(image_name).name
    if img.is_file():
        return save_landmark_56_to_pts(img, x, y)
    # JSON fallback
    import json

    ann_dir.mkdir(parents=True, exist_ok=True)
    path = annotation_path_for(image_name, ann_dir)
    data = load_annotation(path)
    data["image"] = Path(image_name).name
    if image_width is not None:
        data["image_width"] = int(image_width)
    if image_height is not None:
        data["image_height"] = int(image_height)
    lms = data["landmarks"]
    while len(lms) < NUM_LANDMARKS_56:
        lms.append(None)
    lms[PIERCING_INDEX] = [float(x), float(y)]
    data["landmarks"] = lms
    data["landmark_56"] = {"x": float(x), "y": float(y)}
    data["piercing"] = [float(x), float(y)]
    path.write_text(json.dumps(data, indent=2))
    return path


def has_landmark_56(path: Path) -> bool:
    path = Path(path)
    if path.suffix.lower() == ".pts":
        return len(parse_pts(path)) >= NUM_LANDMARKS_56
    if path.suffix.lower() in IMAGE_EXTS:
        return has_landmark_56_path(path)
    data = load_annotation(path)
    pt = data["landmarks"][PIERCING_INDEX]
    return pt is not None and len(pt) == 2


def list_images(img_dir: Path = DATA_IMAGES) -> List[Path]:
    if not img_dir.is_dir():
        return []
    return [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def list_dataset_images(root: Path = DEFAULT_ANNOTATE_DIR) -> List[Path]:
    """List images that have a sibling .pts (iBUG style)."""
    root = Path(root)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and pts_path_for(p).is_file():
            out.append(p)
    return out


def ingest_uploads(file_paths: List[str], img_dir: Path = DATA_IMAGES) -> List[str]:
    img_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    for fp in file_paths:
        src = Path(fp)
        if not src.is_file():
            continue
        dst = img_dir / src.name
        if dst.exists() and dst.resolve() != src.resolve():
            stem, suf = src.stem, src.suffix
            i = 1
            while dst.exists():
                dst = img_dir / f"{stem}_{i}{suf}"
                i += 1
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        names.append(dst.name)
    return names
