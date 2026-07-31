"""On-the-fly additive augmentation: 45 variants per image (no disk writes).

Families (additive, not cartesian):
  flip=2, rotation=7, scale=5, translate=9,
  brightness=5, contrast=5, blur=3, noise_jpeg=4, occlusion_hair=5
→ 45
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from train.augment import _affine_landmarks

FLIP_N = 2
ROTATION_N = 7
SCALE_N = 5
TRANSLATE_N = 9
BRIGHTNESS_N = 5
CONTRAST_N = 5
BLUR_N = 3
NOISE_JPEG_N = 4
OCCLUSION_HAIR_N = 5

VARIANTS_PER_IMAGE = (
    FLIP_N
    + ROTATION_N
    + SCALE_N
    + TRANSLATE_N
    + BRIGHTNESS_N
    + CONTRAST_N
    + BLUR_N
    + NOISE_JPEG_N
    + OCCLUSION_HAIR_N
)  # 45

assert VARIANTS_PER_IMAGE == 45, VARIANTS_PER_IMAGE


def variant_tags() -> List[str]:
    tags: List[str] = []
    tags += [f"flip_{i}" for i in range(FLIP_N)]
    tags += [f"rotation_{i}" for i in range(ROTATION_N)]
    tags += [f"scale_{i}" for i in range(SCALE_N)]
    tags += [f"translate_{i}" for i in range(TRANSLATE_N)]
    tags += [f"brightness_{i}" for i in range(BRIGHTNESS_N)]
    tags += [f"contrast_{i}" for i in range(CONTRAST_N)]
    tags += [f"blur_{i}" for i in range(BLUR_N)]
    tags += [f"noise_jpeg_{i}" for i in range(NOISE_JPEG_N)]
    tags += [f"occlusion_hair_{i}" for i in range(OCCLUSION_HAIR_N)]
    assert len(tags) == VARIANTS_PER_IMAGE
    return tags


_TAGS = variant_tags()


def _apply_flip(
    img: np.ndarray, pts: Optional[np.ndarray], do: bool
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not do:
        return img.copy(), None if pts is None else pts.copy()
    out = cv2.flip(img, 1)
    if pts is None:
        return out, None
    p = pts.copy()
    p[:, 0] = (img.shape[1] - 1) - p[:, 0]
    return out, p


def _apply_affine(
    img: np.ndarray,
    pts: Optional[np.ndarray],
    scale: float,
    angle_deg: float,
    tx: float,
    ty: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    if pts is None:
        return out, None
    return out, _affine_landmarks(pts, M)


def _apply_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) + 255.0 * delta, 0, 255).astype(np.uint8)


def _apply_contrast(img: np.ndarray, delta: float) -> np.ndarray:
    alpha = 1.0 + delta
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img.copy()
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _apply_occlusion_hair(
    img: np.ndarray, idx: int, rng: np.random.Generator
) -> np.ndarray:
    """Soft dark/brown strand-like blobs (stand-in for hair overlays)."""
    out = img.copy()
    h, w = out.shape[:2]
    n_strokes = 2 + (idx % 3)
    for _ in range(n_strokes):
        x0 = int(rng.integers(0, max(1, w)))
        y0 = int(rng.integers(0, max(1, h // 2)))
        length = int(rng.integers(h // 4, max(h // 4 + 1, h)))
        thickness = int(rng.integers(3, 12))
        color = (
            int(rng.integers(10, 60)),
            int(rng.integers(10, 50)),
            int(rng.integers(5, 40)),
        )
        x1 = int(np.clip(x0 + rng.integers(-w // 8, w // 8 + 1), 0, w - 1))
        y1 = int(np.clip(y0 + length, 0, h - 1))
        cv2.line(out, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
        # soft edge
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def apply_variant(
    img: np.ndarray,
    pts: Optional[np.ndarray],
    variant_id: int,
    seed: int = 0,
) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    """
    Apply exactly one additive family member by variant_id in [0, 44].
    Returns (tag, aug_image, aug_landmarks). Photometric augs keep pts.
    """
    if not (0 <= variant_id < VARIANTS_PER_IMAGE):
        raise IndexError(f"variant_id {variant_id} out of range 0..{VARIANTS_PER_IMAGE-1}")

    tag = _TAGS[variant_id]
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed * 10007 + variant_id)

    # flip: 0..1
    if variant_id < FLIP_N:
        i = variant_id
        return tag, *_apply_flip(img, pts, do=bool(i))

    variant_id -= FLIP_N
    # rotation: 7 angles
    if variant_id < ROTATION_N:
        i = variant_id
        angles = np.linspace(-15.0, 15.0, ROTATION_N)
        out, p = _apply_affine(img, pts, 1.0, float(angles[i]), 0.0, 0.0)
        return tag, out, p

    variant_id -= ROTATION_N
    # scale: 5
    if variant_id < SCALE_N:
        i = variant_id
        scales = np.linspace(0.85, 1.15, SCALE_N)
        out, p = _apply_affine(img, pts, float(scales[i]), 0.0, 0.0, 0.0)
        return tag, out, p

    variant_id -= SCALE_N
    # translate: 9
    if variant_id < TRANSLATE_N:
        i = variant_id
        fracs = np.linspace(-0.10, 0.10, TRANSLATE_N)
        f = float(fracs[i])
        out, p = _apply_affine(img, pts, 1.0, 0.0, f * w, f * h * 0.5)
        return tag, out, p

    variant_id -= TRANSLATE_N
    # brightness: 5
    if variant_id < BRIGHTNESS_N:
        i = variant_id
        deltas = np.linspace(-0.25, 0.25, BRIGHTNESS_N)
        out = _apply_brightness(img, float(deltas[i]))
        return tag, out, None if pts is None else pts.copy()

    variant_id -= BRIGHTNESS_N
    # contrast: 5
    if variant_id < CONTRAST_N:
        i = variant_id
        deltas = np.linspace(-0.25, 0.25, CONTRAST_N)
        out = _apply_contrast(img, float(deltas[i]))
        return tag, out, None if pts is None else pts.copy()

    variant_id -= CONTRAST_N
    # blur: 3
    if variant_id < BLUR_N:
        i = variant_id
        kernels = [3, 5, 7]
        out = _apply_blur(img, kernels[i])
        return tag, out, None if pts is None else pts.copy()

    variant_id -= BLUR_N
    # noise / jpeg: 4
    if variant_id < NOISE_JPEG_N:
        i = variant_id
        if i % 2 == 0:
            out = _apply_noise(img, sigma=8.0 + 6.0 * (i // 2), rng=rng)
        else:
            out = _apply_jpeg(img, quality=85 - 15 * (i // 2))
        return tag, out, None if pts is None else pts.copy()

    variant_id -= NOISE_JPEG_N
    # occlusion hair: 5
    i = variant_id
    out = _apply_occlusion_hair(img, i, rng)
    return tag, out, None if pts is None else pts.copy()
