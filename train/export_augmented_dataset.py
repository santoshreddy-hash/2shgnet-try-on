"""Export additive augmentation packs (44 variants per image) for documentation + training.

Per image (additive, not cartesian):
  flip=2, scale=5, translate=7, blur=5, noise=6,
  occlusion=6, smoke_blur=5, brightness=4, contrast=4  → 44
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.annotations import parse_pts, pts_path_for, write_pts
from train.augment import _affine_landmarks
from train.yolo_pose_labels import (
    IMAGE_EXTS,
    is_yolo_pose_images_dir,
    label_path_for,
    labels_dir_for_images,
    list_yolo_pose_images,
    read_yolo_pose,
    write_yolo_pose,
)

# Additive value counts
FLIP_N = 2
SCALE_N = 5
TRANSLATE_N = 7
BLUR_N = 5
NOISE_N = 6
OCCLUSION_N = 6
SMOKE_N = 5
BRIGHTNESS_N = 4
CONTRAST_N = 4
VARIANTS_PER_IMAGE = (
    FLIP_N
    + SCALE_N
    + TRANSLATE_N
    + BLUR_N
    + NOISE_N
    + OCCLUSION_N
    + SMOKE_N
    + BRIGHTNESS_N
    + CONTRAST_N
)  # 44


def _list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if is_yolo_pose_images_dir(folder):
        return list_yolo_pose_images(folder)
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return files


def _load_landmarks_px(
    image_path: Path, img_w: int, img_h: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return (xy Nx2, vis N, mode) where mode in {yolo, pts, none}."""
    if is_yolo_pose_images_dir(image_path.parent):
        lp = label_path_for(image_path)
        if lp.is_file():
            try:
                _, kpts = read_yolo_pose(lp)
                xy = np.zeros((len(kpts), 2), dtype=np.float32)
                vis = kpts[:, 2].astype(np.float32)
                xy[:, 0] = kpts[:, 0] * img_w
                xy[:, 1] = kpts[:, 1] * img_h
                return xy, vis, "yolo"
            except ValueError:
                pass
    pts_p = pts_path_for(image_path)
    if pts_p.is_file():
        raw = parse_pts(pts_p)
        if raw:
            xy = np.asarray(raw, dtype=np.float32)
            vis = np.ones((len(xy),), dtype=np.float32)
            return xy, vis, "pts"
    return None, None, "none"


def _save_landmarks(
    mode: str,
    out_img: Path,
    xy: Optional[np.ndarray],
    vis: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    labels_out: Path,
) -> None:
    if xy is None or mode == "none":
        return
    if mode == "yolo":
        # rebuild bbox from visible points
        valid = np.isfinite(xy).all(axis=1) & ((vis is None) | (vis > 0))
        pts = xy[valid] if np.any(valid) else xy
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        bw = max(1.0, float(x1 - x0))
        bh = max(1.0, float(y1 - y0))
        cx = float((x0 + x1) * 0.5) / img_w
        cy = float((y0 + y1) * 0.5) / img_h
        header = [0, cx, cy, bw / img_w, bh / img_h]
        kpts = np.zeros((len(xy), 3), dtype=np.float64)
        kpts[:, 0] = np.clip(xy[:, 0] / img_w, 0.0, 1.0)
        kpts[:, 1] = np.clip(xy[:, 1] / img_h, 0.0, 1.0)
        kpts[:, 2] = vis if vis is not None else 2.0
        write_yolo_pose(labels_out / f"{out_img.stem}.txt", header, kpts)
    elif mode == "pts":
        write_pts(out_img.with_suffix(".pts"), [[float(x), float(y)] for x, y in xy])


def _apply_flip(img: np.ndarray, pts: Optional[np.ndarray], do: bool):
    if not do:
        return img.copy(), None if pts is None else pts.copy()
    out = cv2.flip(img, 1)
    if pts is None:
        return out, None
    h, w = img.shape[:2]
    p = pts.copy()
    p[:, 0] = (w - 1) - p[:, 0]
    return out, p


def _apply_affine(
    img: np.ndarray, pts: Optional[np.ndarray], scale: float, tx: float, ty: float
):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, float(scale))
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    if pts is None:
        return out, None
    return out, _affine_landmarks(pts, M)


def _apply_brightness(img: np.ndarray, amount: float):
    beta = 255.0 * float(amount)
    return np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def _apply_contrast(img: np.ndarray, amount: float):
    alpha = 1.0 + float(amount)
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, k: int):
    k = int(k) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator):
    noise = rng.normal(0.0, float(sigma), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_occlusion(img: np.ndarray, idx: int, rng: np.random.Generator):
    out = img.copy()
    h, w = out.shape[:2]
    # deterministic per index but varied size
    frac = 0.08 + 0.04 * idx  # 0.08 .. 0.28
    bw = max(8, int(w * frac))
    bh = max(8, int(h * frac))
    x0 = int(rng.integers(0, max(1, w - bw)))
    y0 = int(rng.integers(0, max(1, h - bh)))
    color = int(rng.integers(0, 256))
    out[y0 : y0 + bh, x0 : x0 + bw] = (color, color, color)
    return out


def _apply_smoke(img: np.ndarray, strength: float, rng: np.random.Generator):
    """Soft atmospheric haze / smoke overlay + mild blur."""
    h, w = img.shape[:2]
    haze = np.full_like(img, 220, dtype=np.float32)
    # soft blob mask
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float(rng.uniform(0.3 * w, 0.7 * w))
    cy = float(rng.uniform(0.3 * h, 0.7 * h))
    rx = float(rng.uniform(0.25 * w, 0.55 * w))
    ry = float(rng.uniform(0.25 * h, 0.55 * h))
    mask = np.exp(-(((xx - cx) / max(1.0, rx)) ** 2 + ((yy - cy) / max(1.0, ry)) ** 2))
    mask = (mask * float(strength))[..., None]
    base = img.astype(np.float32)
    mixed = base * (1.0 - mask) + haze * mask
    k = 5 if strength < 0.45 else 9
    mixed = cv2.GaussianBlur(mixed.astype(np.uint8), (k, k), 0).astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def iter_variants(
    img: np.ndarray, pts: Optional[np.ndarray], seed: int
) -> Iterable[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
    """Yield (tag, image, landmarks) — additive only."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    # flip: 2
    for i, do in enumerate([False, True]):
        out, p = _apply_flip(img, pts, do)
        yield f"flip_{i}", out, p

    # scale: 5
    scales = np.linspace(0.85, 1.15, SCALE_N)
    for i, s in enumerate(scales):
        out, p = _apply_affine(img, pts, float(s), 0.0, 0.0)
        yield f"scale_{i}", out, p

    # translate: 7 along x (symmetric)
    fracs = np.linspace(-0.08, 0.08, TRANSLATE_N)
    for i, f in enumerate(fracs):
        out, p = _apply_affine(img, pts, 1.0, float(f) * w, 0.0)
        yield f"translate_{i}", out, p

    # blur: 5
    kernels = [3, 5, 7, 9, 11]
    for i, k in enumerate(kernels):
        yield f"blur_{i}", _apply_blur(img, k), None if pts is None else pts.copy()

    # noise: 6
    sigmas = [5, 10, 15, 20, 25, 30]
    for i, sig in enumerate(sigmas):
        yield (
            f"noise_{i}",
            _apply_noise(img, sig, rng),
            None if pts is None else pts.copy(),
        )

    # occlusion: 6
    for i in range(OCCLUSION_N):
        local = np.random.default_rng(seed * 100 + i)
        yield (
            f"occlusion_{i}",
            _apply_occlusion(img, i, local),
            None if pts is None else pts.copy(),
        )

    # smoke blur: 5
    strengths = np.linspace(0.15, 0.55, SMOKE_N)
    for i, st in enumerate(strengths):
        local = np.random.default_rng(seed * 200 + i)
        yield (
            f"smoke_blur_{i}",
            _apply_smoke(img, float(st), local),
            None if pts is None else pts.copy(),
        )

    # brightness: 4
    brights = np.linspace(-0.20, 0.20, BRIGHTNESS_N)
    for i, b in enumerate(brights):
        yield (
            f"brightness_{i}",
            _apply_brightness(img, float(b)),
            None if pts is None else pts.copy(),
        )

    # contrast: 4
    contrasts = np.linspace(-0.20, 0.20, CONTRAST_N)
    for i, c in enumerate(contrasts):
        yield (
            f"contrast_{i}",
            _apply_contrast(img, float(c)),
            None if pts is None else pts.copy(),
        )


def process_image(
    image_path: Path,
    out_images: Path,
    out_labels: Path,
    skip_existing: bool = True,
) -> Dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"name": image_path.name, "ok": False, "error": "read_fail", "n": 0}
    h, w = img.shape[:2]
    pts, vis, mode = _load_landmarks_px(image_path, w, h)
    seed = abs(hash(image_path.stem)) % (2**31 - 1)
    written = 0
    for tag, aug_img, aug_pts in iter_variants(img, pts, seed):
        out_name = f"{image_path.stem}__{tag}{image_path.suffix.lower()}"
        out_path = out_images / out_name
        if skip_existing and out_path.is_file():
            written += 1
            continue
        ah, aw = aug_img.shape[:2]
        cv2.imwrite(str(out_path), aug_img)
        _save_landmarks(mode, out_path, aug_pts, vis, aw, ah, out_labels)
        written += 1
    return {"name": image_path.name, "ok": True, "mode": mode, "n": written}


def export_split(
    images_dir: Path,
    out_root: Path,
    split_name: str,
    limit: Optional[int] = None,
) -> Dict:
    images = _list_images(images_dir)
    if limit is not None:
        images = images[:limit]
    out_images = out_root / split_name / "images"
    out_labels = out_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    for i, p in enumerate(images, 1):
        r = process_image(p, out_images, out_labels)
        rows.append(r)
        total += int(r.get("n", 0))
        if i % 25 == 0 or i == len(images):
            print(
                f"[{split_name}] {i}/{len(images)} images → {total} aug files",
                flush=True,
            )

    manifest = out_root / split_name / "manifest.csv"
    with manifest.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["name", "ok", "mode", "n", "error"])
        wri.writeheader()
        for r in rows:
            wri.writerow(
                {
                    "name": r.get("name"),
                    "ok": r.get("ok"),
                    "mode": r.get("mode", ""),
                    "n": r.get("n", 0),
                    "error": r.get("error", ""),
                }
            )
    summary = {
        "split": split_name,
        "source": str(images_dir),
        "n_source": len(images),
        "variants_per_image": VARIANTS_PER_IMAGE,
        "n_augmented": total,
        "out_images": str(out_images),
        "out_labels": str(out_labels),
    }
    (out_root / split_name / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Export additive augmented dataset (44/img)")
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "augmented"),
        help="Output root",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit images per split")
    p.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "ear_pose", "collectiona"],
        help="Which asset to process",
    )
    args = p.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    assert VARIANTS_PER_IMAGE == 44, VARIANTS_PER_IMAGE
    print(f"Variants per image: {VARIANTS_PER_IMAGE}", flush=True)

    jobs: List[Tuple[str, Path]] = []
    ear = ROOT / "data" / "data" / "ear_pose" / "images"
    col = ROOT / "data" / "data" / "ibug_crops"
    if args.only in ("all", "ear_pose"):
        jobs += [
            ("ear_pose_train", ear / "train"),
            ("ear_pose_val", ear / "val"),
        ]
    if args.only in ("all", "collectiona"):
        jobs += [
            ("collectiona_train", col / "collectiona_train"),
            ("collectiona_test", col / "collectiona_test"),
        ]

    all_sum = []
    for name, src in jobs:
        if not src.is_dir():
            print(f"SKIP missing: {src}", flush=True)
            continue
        all_sum.append(export_split(src, out_root, name, limit=args.limit))

    grand = {
        "variants_per_image": VARIANTS_PER_IMAGE,
        "splits": all_sum,
        "total_source": sum(s["n_source"] for s in all_sum),
        "total_augmented": sum(s["n_augmented"] for s in all_sum),
    }
    (out_root / "grand_summary.json").write_text(json.dumps(grand, indent=2))
    print("DONE", json.dumps(grand, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
