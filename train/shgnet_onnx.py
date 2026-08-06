#!/usr/bin/env python3
"""ONNX Runtime backend for SHGNet-56 (55 ear + piercing)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from train.config import INPUT_SIZE, NUM_LANDMARKS_56, resolve_onnx_export


def preprocess_ear_bgr_numpy(ear_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Match training preprocess → float32 NCHW BGR [0,1]."""
    if ear_bgr.size == 0:
        raise ValueError("Empty ear crop")
    img = cv2.resize(ear_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    img = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    arr = img.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def heatmaps_to_points_soft(
    hm: np.ndarray, input_size: int = INPUT_SIZE, radius: int = 2
) -> np.ndarray:
    """Soft-argmax around peak → (C, 2) in input_size space."""
    single = hm.ndim == 3
    if single:
        hm = hm[np.newaxis, ...]
    b, n, h, w = hm.shape
    scale_x = input_size / float(w)
    scale_y = input_size / float(h)
    pts = np.zeros((b, n, 2), dtype=np.float32)
    for bi in range(b):
        for i in range(n):
            flat = hm[bi, i]
            idx = int(flat.argmax())
            yy, xx = divmod(idx, w)
            y0 = max(0, yy - radius)
            y1 = min(h - 1, yy + radius)
            x0 = max(0, xx - radius)
            x1 = min(w - 1, xx + radius)
            patch = flat[y0 : y1 + 1, x0 : x1 + 1]
            patch = patch - float(patch.max())
            wt = np.exp(patch)
            s = float(wt.sum())
            if s < 1e-12:
                pts[bi, i, 0] = xx * scale_x
                pts[bi, i, 1] = yy * scale_y
            else:
                ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
                pts[bi, i, 0] = float((wt * xs).sum() / s) * scale_x
                pts[bi, i, 1] = float((wt * ys).sum() / s) * scale_y
    return pts[0] if single else pts


class SHGNet56Onnx:
    """SHGNet-56 via ONNX Runtime — no PyTorch at inference."""

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        input_size: int = INPUT_SIZE,
        providers: Optional[list[str]] = None,
    ) -> None:
        import onnxruntime as ort

        path = Path(onnx_path) if onnx_path else resolve_onnx_export()
        path = path.resolve()
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise FileNotFoundError(
                f"ONNX not found or stub: {path}\n"
                "Place models/shgnet/SHGNet-56.onnx or run: python -m train.export_onnx"
            )

        avail = set(ort.get_available_providers())
        if providers is None:
            # Prefer DirectML on Windows (no CUDA 13 DLL hell), then CUDA/CoreML, else CPU.
            preferred = [
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred if p in avail] or ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # let EP decide
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=so, providers=providers
            )
        except Exception as first_err:
            if providers != ["CPUExecutionProvider"]:
                print(f"[SHGNet-56-ONNX] {providers} failed ({first_err}); CPU fallback")
                self.session = ort.InferenceSession(
                    str(path),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(
            f"[SHGNet-56-ONNX] Loaded {path.name} "
            f"providers={self.session.get_providers()}"
        )

    def predict(self, ear_bgr: np.ndarray) -> np.ndarray:
        pts, _ = self.predict_with_score(ear_bgr)
        return pts

    def predict_with_score(self, ear_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return ((56, 2) in 256-space, mean peak heatmap score)."""
        x = preprocess_ear_bgr_numpy(ear_bgr, self.input_size)
        outs = self.session.run([self.output_name], {self.input_name: x})
        hm = np.asarray(outs[0])
        hm0 = hm[0] if hm.ndim == 4 else hm
        pts = heatmaps_to_points_soft(hm0, self.input_size)
        if pts.shape[0] != NUM_LANDMARKS_56:
            raise ValueError(f"Expected ({NUM_LANDMARKS_56}, 2), got {pts.shape}")
        flat = hm0.reshape(hm0.shape[0], -1)
        score = float(flat.max(axis=1).mean())
        return pts, score
