"""Validation metrics: NME, PCK, piercing error, heatmap MSE."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train.config import INPUT_SIZE, NUM_LANDMARKS_55, PCK_THRESHOLDS, PIERCING_INDEX
from train.shgnet_base import heatmaps_to_points


def heatmap_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def nme(pred_pts: np.ndarray, gt_pts: np.ndarray, norm: float | None = None) -> float:
    """Normalized Mean Error over valid landmarks. Default norm = crop diagonal."""
    pred = np.asarray(pred_pts, dtype=np.float32)
    gt = np.asarray(gt_pts, dtype=np.float32)
    valid = np.isfinite(gt).all(axis=1) & np.isfinite(pred).all(axis=1)
    if not np.any(valid):
        return float("nan")
    if norm is None or norm <= 0:
        norm = float(np.sqrt(2.0) * INPUT_SIZE)
    err = np.linalg.norm(pred[valid] - gt[valid], axis=1).mean()
    return float(err / norm)


def pck(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    thresholds: Iterable[float] = PCK_THRESHOLDS,
    norm: float | None = None,
) -> Dict[str, float]:
    pred = np.asarray(pred_pts, dtype=np.float32)
    gt = np.asarray(gt_pts, dtype=np.float32)
    valid = np.isfinite(gt).all(axis=1) & np.isfinite(pred).all(axis=1)
    if not np.any(valid):
        return {f"pck@{t}": float("nan") for t in thresholds}
    if norm is None or norm <= 0:
        norm = float(np.sqrt(2.0) * INPUT_SIZE)
    d = np.linalg.norm(pred[valid] - gt[valid], axis=1) / norm
    out = {}
    for t in thresholds:
        out[f"pck@{t}"] = float((d <= t).mean())
    return out


def piercing_point_error(
    pred_pts: np.ndarray, gt_pts: np.ndarray
) -> Tuple[float, float]:
    """Return (pixel_error, normalized_error) for landmark #56."""
    pred = np.asarray(pred_pts, dtype=np.float32)
    gt = np.asarray(gt_pts, dtype=np.float32)
    pe = float(np.linalg.norm(pred[PIERCING_INDEX] - gt[PIERCING_INDEX]))
    norm = float(np.sqrt(2.0) * INPUT_SIZE)
    return pe, pe / norm


def decode_heatmaps(heatmaps: torch.Tensor) -> np.ndarray:
    """(B, C, H, W) or (C, H, W) → points in INPUT_SIZE space."""
    return heatmaps_to_points(heatmaps, INPUT_SIZE)


def landmark_nme_55(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    return nme(pred_pts[:NUM_LANDMARKS_55], gt_pts[:NUM_LANDMARKS_55])
