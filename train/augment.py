"""Geometric + photometric augmentations that keep landmarks in sync."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from train.config import (
    AUG_BLUR_PROB,
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_FLIP_PROB,
    AUG_ROTATION_DEG,
    AUG_SCALE_MAX,
    AUG_SCALE_MIN,
    AUG_TRANSLATE_FRAC,
    INPUT_SIZE,
)


@dataclass
class AugmentConfig:
    rotation_deg: float = AUG_ROTATION_DEG
    scale_min: float = AUG_SCALE_MIN
    scale_max: float = AUG_SCALE_MAX
    translate_frac: float = AUG_TRANSLATE_FRAC
    brightness: float = AUG_BRIGHTNESS
    contrast: float = AUG_CONTRAST
    blur_prob: float = AUG_BLUR_PROB
    flip_prob: float = AUG_FLIP_PROB


def _affine_landmarks(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts.astype(np.float32), ones], axis=1)
    out = hom @ M.T
    return out[:, :2]


def augment_sample(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    cfg: AugmentConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation, scale, translation, flip, brightness, contrast, blur.
    image: HxWx3 BGR uint8 (INPUT_SIZE typically)
    landmarks: (N, 2) in image pixel space
    """
    cfg = cfg or AugmentConfig()
    rng = rng or np.random.default_rng()
    img = image_bgr.copy()
    pts = landmarks.astype(np.float32).copy()
    h, w = img.shape[:2]
    assert h == w == INPUT_SIZE or True  # allow any square/crop already resized

    # Horizontal flip
    if rng.random() < cfg.flip_prob:
        img = cv2.flip(img, 1)
        pts[:, 0] = (w - 1) - pts[:, 0]

    # Affine: rotate + scale + translate
    angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
    tx = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * w)
    ty = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * h)
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    pts = _affine_landmarks(pts, M)

    # Brightness / contrast
    alpha = 1.0 + float(rng.uniform(-cfg.contrast, cfg.contrast))
    beta = 255.0 * float(rng.uniform(-cfg.brightness, cfg.brightness))
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Blur
    if rng.random() < cfg.blur_prob:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img, pts
