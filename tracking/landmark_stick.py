"""Keep landmarks glued to the moving ear between SHG inferences.

Lucas–Kanade tracks a subset of landmarks every display frame and applies the
median translation to the full set — so points stick to ear texture instead of
freezing on a stale YOLO tip.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

_LK_WIN = (21, 21)
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
_MAX_STEP = 28.0  # px/frame clamp — rejects face-feature jumps


class LandmarkStickTracker:
    def __init__(self, n_track: int = 14) -> None:
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

    def update(
        self,
        frame_bgr: np.ndarray,
        worker_pts: Optional[np.ndarray],
        pts_gen: int,
        crop_xyxy: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if worker_pts is not None and int(pts_gen) != self.pts_gen:
            self._seed(gray, worker_pts, pts_gen)
            return self.abs_pts

        if self.prev_gray is None or self.track_pts is None or self.abs_pts is None:
            if worker_pts is not None:
                self._seed(gray, worker_pts, pts_gen)
            return self.abs_pts

        nxt, st, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.track_pts,
            None,
            winSize=_LK_WIN,
            maxLevel=3,
            criteria=_LK_CRITERIA,
        )
        self.prev_gray = gray
        if nxt is None or st is None:
            return self.abs_pts

        good = st.reshape(-1) == 1
        if int(good.sum()) < 3:
            return self.abs_pts

        prev = self.track_pts.reshape(-1, 2)[good]
        cur = nxt.reshape(-1, 2)[good]
        # Drop trackers that left the ear crop (if known)
        if crop_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in crop_xyxy]
            pad = 8.0
            inside = (
                (cur[:, 0] >= x1 - pad)
                & (cur[:, 0] <= x2 + pad)
                & (cur[:, 1] >= y1 - pad)
                & (cur[:, 1] <= y2 + pad)
            )
            if int(inside.sum()) >= 3:
                prev, cur = prev[inside], cur[inside]
            else:
                # Lost ear lock — keep last pose, wait for next SHG
                return self.abs_pts

        delta = cur - prev
        med = np.median(delta, axis=0).astype(np.float32)
        step = float(np.hypot(med[0], med[1]))
        if step > _MAX_STEP:
            med *= np.float32(_MAX_STEP / step)

        self.abs_pts = self.abs_pts + med.reshape(1, 2)
        self.track_pts = self.abs_pts[self.track_idx].reshape(-1, 1, 2).astype(np.float32)
        return self.abs_pts
