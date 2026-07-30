"""Gaussian heatmap generation for landmark targets."""

from __future__ import annotations

import numpy as np

from train.config import GAUSSIAN_SIGMA, HEATMAP_SIZE, INPUT_SIZE, NUM_LANDMARKS_56


def generate_gaussian_heatmaps(
    landmarks: np.ndarray,
    input_size: int = INPUT_SIZE,
    heatmap_size: int = HEATMAP_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
    num_landmarks: int = NUM_LANDMARKS_56,
) -> np.ndarray:
    """
    landmarks: (N, 2) in input_size pixel coords.
    returns: (num_landmarks, H, W) float32 in [0, 1]
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (num_landmarks, 2):
        raise ValueError(f"Expected landmarks ({num_landmarks}, 2), got {pts.shape}")

    scale = heatmap_size / float(input_size)
    heatmaps = np.zeros((num_landmarks, heatmap_size, heatmap_size), dtype=np.float32)
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size]

    for i in range(num_landmarks):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < 0 or y < 0 or x >= input_size or y >= input_size:
            continue
        cx = x * scale
        cy = y * scale
        heatmaps[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return heatmaps
