"""Keep landmarks glued to the moving ear between / with SHG inferences.

Lucas–Kanade tracks a subset of landmarks every display frame. When a new SHG
pose arrives, it is blended with the texture-predicted cloud so points stay
locked to ear skin during head motion instead of lagging behind.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

_LK_WIN = (31, 31)
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
_MAX_STEP = 48.0  # px/frame clamp — rejects face-feature jumps
# How much texture LK pulls a new SHG pose toward the ear surface
_SHG_BLEND = 0.65  # stronger texture pull — landmarks stay glued to ear skin


class LandmarkStickTracker:
    def __init__(self, n_track: int = 18) -> None:
        self.n_track = int(n_track)
        self.prev_gray: Optional[np.ndarray] = None
        self.track_pts: Optional[np.ndarray] = None  # (K, 1, 2)
        self.track_idx: Optional[np.ndarray] = None
        self.abs_pts: Optional[np.ndarray] = None  # (N, 2)
        self.pts_gen = -1

    def reset(self) -> None:
        self.prev_gray = None
        self.track_pts = None
        self.track_idx = None
        self.abs_pts = None
        self.pts_gen = -1

    def _seed(self, gray: np.ndarray, pts: np.ndarray, gen: int) -> None:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        self.abs_pts = pts.copy()
        n = pts.shape[0]
        # Spread trackers across helix/antihelix/lobe (skip piercing alone)
        k = min(self.n_track, max(4, n - 1))
        self.track_idx = np.linspace(0, min(54, n - 2), k, dtype=np.int32)
        self.track_pts = self.abs_pts[self.track_idx].reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray
        self.pts_gen = int(gen)

    def _lk_delta(
        self,
        gray: np.ndarray,
        crop_xyxy: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        """Median Lucas–Kanade translation of track points, or None if lost."""
        if self.prev_gray is None or self.track_pts is None or self.abs_pts is None:
            return None
        if self.prev_gray.shape != gray.shape:
            return None

        nxt, st, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.track_pts,
            None,
            winSize=_LK_WIN,
            maxLevel=4,
            criteria=_LK_CRITERIA,
        )
        if nxt is None or st is None:
            return None

        good = st.reshape(-1) == 1
        if int(good.sum()) < 3:
            return None

        prev = self.track_pts.reshape(-1, 2)[good]
        cur = nxt.reshape(-1, 2)[good]
        if crop_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in crop_xyxy]
            pad = 12.0
            inside = (
                (cur[:, 0] >= x1 - pad)
                & (cur[:, 0] <= x2 + pad)
                & (cur[:, 1] >= y1 - pad)
                & (cur[:, 1] <= y2 + pad)
            )
            if int(inside.sum()) >= 3:
                prev, cur = prev[inside], cur[inside]
            else:
                return None

        delta = cur - prev
        med = np.median(delta, axis=0).astype(np.float32)
        step = float(np.hypot(med[0], med[1]))
        if step > _MAX_STEP:
            med *= np.float32(_MAX_STEP / step)
        return med

    def update(
        self,
        frame_bgr: np.ndarray,
        worker_pts: Optional[np.ndarray],
        pts_gen: int,
        crop_xyxy: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lk = self._lk_delta(gray, crop_xyxy)

        # New SHG / filter pose: glue it toward texture-predicted ear motion
        if worker_pts is not None and int(pts_gen) != self.pts_gen:
            pts = np.asarray(worker_pts, dtype=np.float32).reshape(-1, 2).copy()
            if lk is not None and self.abs_pts is not None:
                predicted = self.abs_pts + lk.reshape(1, 2)
                # Tip-rigid: match SHG tip/centroid to texture cloud, then blend
                if pts.shape[0] == predicted.shape[0]:
                    shg_c = pts.mean(axis=0)
                    pred_c = predicted.mean(axis=0)
                    predicted = predicted + (shg_c - pred_c).reshape(1, 2)
                    w = float(_SHG_BLEND)
                    pts = (1.0 - w) * pts + w * predicted
            self._seed(gray, pts, pts_gen)
            return self.abs_pts

        if self.prev_gray is None or self.track_pts is None or self.abs_pts is None:
            if worker_pts is not None:
                self._seed(gray, worker_pts, pts_gen)
            return self.abs_pts

        # No new SHG this call: pure LK stick
        if lk is None:
            self.prev_gray = gray
            return self.abs_pts

        self.abs_pts = self.abs_pts + lk.reshape(1, 2)
        self.track_pts = self.abs_pts[self.track_idx].reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray
        return self.abs_pts
