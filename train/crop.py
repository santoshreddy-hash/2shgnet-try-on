"""Ear crop from full-ear or side-face images; remap landmark coords into crop.

Matches jewellery / ear_pose framing:
  medial-offset tip crop + gray pad (ox/oy may be negative) + LEFT flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from train.config import (
    CROP_PAD,
    EAR_KEYPOINT_MIN_CONF,
    INPUT_SIZE,
    resolve_yolo_device,
    resolve_yolo_weights,
)


@dataclass
class CropMeta:
    x0: int  # crop origin ox (may be negative when gray-padded)
    y0: int  # crop origin oy
    side: int
    flipped: bool = False  # left ear flipped to right-like
    tip_x: float = 0.0
    tip_y: float = 0.0
    cx: float = 0.0  # square center used for crop
    cy: float = 0.0
    side_len: float = 0.0  # unrounded side length


def extract_square_crop(
    frame_bgr: np.ndarray,
    cx: float,
    cy: float,
    side: float,
    pad_color: int = 114,
) -> tuple[np.ndarray, int, int, int]:
    """
    Always-square ear crop with gray pad when the box leaves the frame.

    Returns (crop_bgr, origin_x, origin_y, side_px). Remap:
      frame_x = origin_x + x_256 * side_px / 256
    """
    h, w = frame_bgr.shape[:2]
    side_i = max(32, int(round(side)))
    ox = int(round(cx - side_i * 0.5))
    oy = int(round(cy - side_i * 0.5))
    canvas = np.full((side_i, side_i, 3), int(pad_color), dtype=np.uint8)
    sx1, sy1 = max(0, ox), max(0, oy)
    sx2, sy2 = min(w, ox + side_i), min(h, oy + side_i)
    dx, dy = sx1 - ox, sy1 - oy
    if sx2 > sx1 and sy2 > sy1:
        canvas[dy : dy + (sy2 - sy1), dx : dx + (sx2 - sx1)] = frame_bgr[
            sy1:sy2, sx1:sx2
        ]
    return canvas, ox, oy, side_i


def extract_crop(
    image_bgr: np.ndarray, x0: int, y0: int, side: int, flip: bool = False
) -> np.ndarray:
    """Legacy wrapper — prefer extract_square_crop for live webcam."""
    cx = x0 + side * 0.5
    cy = y0 + side * 0.5
    crop, _, _, _ = extract_square_crop(image_bgr, cx, cy, float(side))
    if flip:
        crop = cv2.flip(crop, 1)
    return crop


def tip_centered_square(
    tip_xy: Tuple[float, float],
    pinna_h: float,
    frame_shape: Tuple[int, int, int],
    pad: float = CROP_PAD,
) -> Tuple[int, int, int]:
    """Deprecated clamped crop (kept for older callers). Prefer medial + gray pad."""
    h, w = frame_shape[:2]
    tip_x, tip_y = tip_xy
    side = int(max(48, pinna_h * pad))
    side = int(min(side, 0.7 * min(h, w)))
    x0 = int(round(tip_x - side / 2.0))
    y0 = int(round(tip_y - side / 2.0))
    x0 = max(0, min(x0, max(0, w - side)))
    y0 = max(0, min(y0, max(0, h - side)))
    side = min(side, w - x0, h - y0)
    return x0, y0, side


def remap_points_to_crop(
    points: np.ndarray,
    x0: int,
    y0: int,
    side: int,
    out_size: int = INPUT_SIZE,
    flipped: bool = False,
) -> np.ndarray:
    """Map full-image points → out_size crop space."""
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    scale = out_size / float(side)
    pts[:, 0] = (pts[:, 0] - x0) * scale
    pts[:, 1] = (pts[:, 1] - y0) * scale
    if flipped:
        pts[:, 0] = (out_size - 1) - pts[:, 0]
    return pts


def remap_points_to_full(
    points_crop: np.ndarray,
    x0: int,
    y0: int,
    side: int,
    out_size: int = INPUT_SIZE,
    flipped: bool = False,
) -> np.ndarray:
    pts = np.asarray(points_crop, dtype=np.float32).copy()
    if flipped:
        pts[:, 0] = (out_size - 1) - pts[:, 0]
    scale = side / float(out_size)
    pts[:, 0] = pts[:, 0] * scale + x0
    pts[:, 1] = pts[:, 1] * scale + y0
    return pts


def is_side_profile(
    kp: np.ndarray, side_name: str, tip: Tuple[float, float]
) -> bool:
    """
    Accept a usable single-ear view; reject weak tips and near-frontal transitions
    so landmarks are not dragged across the face during L↔R turns.
    """
    le, re = kp[3], kp[4]
    lc, rc = float(le[2]), float(re[2])
    if side_name == "LEFT":
        if lc < EAR_KEYPOINT_MIN_CONF:
            return False
        # Other ear also strong → frontal / turning
        if rc >= 0.35 and rc >= lc * 0.7:
            return False
        ear = le
    else:
        if rc < EAR_KEYPOINT_MIN_CONF:
            return False
        if lc >= 0.35 and lc >= rc * 0.7:
            return False
        ear = re
    nose = kp[0]
    if float(nose[2]) >= 0.2:
        dx = abs(float(ear[0]) - float(nose[0]))
        d = float(np.hypot(float(ear[0]) - float(nose[0]), float(ear[1]) - float(nose[1])))
        if dx < 28.0 or d < 36.0:
            return False
        if dx < d * 0.55:
            return False
    return True


def ear_visibility_score(kp: np.ndarray, side_name: str) -> float:
    """0–1 preference for how usable this ear tip looks (conf + lateral offset)."""
    le, re = kp[3], kp[4]
    ear = le if side_name == "LEFT" else re
    conf = float(ear[2])
    nose = kp[0]
    lateral = 0.5
    if float(nose[2]) >= 0.15:
        dx = abs(float(ear[0]) - float(nose[0]))
        # Prefer ears that aren't on top of the nose (true frontal)
        lateral = float(np.clip(dx / 40.0, 0.0, 1.0))
    return float(0.75 * conf + 0.25 * lateral)


def estimate_pinna_h(
    kp: np.ndarray, box: np.ndarray, tip: Tuple[float, float], frame_hw: Tuple[int, int]
) -> float:
    """Tight pinna height for ear-only crop (avoid face-sized boxes)."""
    h, w = frame_hw
    fmin = float(min(h, w))
    tip_x, tip_y = tip
    cands: list[float] = []
    tip_nose = None

    nose = kp[0]
    if float(nose[2]) >= 0.2:
        tip_nose = float(np.hypot(tip_x - nose[0], tip_y - nose[1]))
        if tip_nose > fmin * 0.04:
            # Side / three-quarter: ear ≈ 0.5–0.65 of tip–nose
            cands.append(tip_nose * 0.58)

    le, re = kp[1], kp[2]  # eyes
    if float(le[2]) > 0.2 and float(re[2]) > 0.2:
        iod = float(np.hypot(le[0] - re[0], le[1] - re[1]))
        if iod > fmin * 0.02:
            cands.append(iod * 1.05)

    bh = float(box[3] - box[1])
    bw = float(box[2] - box[0])
    if bh > 1:
        cands.append(bh * 0.14)
    if bw > 1:
        cands.append(bw * 0.22)

    if not cands:
        return float(max(48.0, fmin * 0.14))

    pinna = float(np.median(cands))
    # Only cap by tip–nose when that cue is trustworthy (not near-frontal)
    if tip_nose is not None and tip_nose > fmin * 0.06:
        pinna = min(pinna, tip_nose * 0.75)
    return float(np.clip(pinna, 48.0, 0.28 * fmin))


def medial_unit(
    tip: Tuple[float, float],
    side: str,
    kp: np.ndarray,
    frame_w: int,
) -> Tuple[float, float]:
    """Unit vector from tip toward face midline (nose preferred)."""
    tip_x, tip_y = tip
    nose = kp[0]
    if float(nose[2]) >= 0.2:
        vx = float(nose[0] - tip_x)
        vy = float(nose[1] - tip_y)
        n = float(np.hypot(vx, vy))
        if n > 1e-3:
            return vx / n, vy / n
    vx = 0.5 * frame_w - tip_x
    if abs(vx) > 1e-3:
        return float(np.sign(vx)), 0.0
    return (-1.0 if side == "LEFT" else 1.0), 0.0


def landmarks_ok(
    pts: np.ndarray, tip: Tuple[float, float], side_px: float
) -> bool:
    """Reject flipped/wrong-scale landmark clouds (jewellery gate)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
    x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    span = max(bw, bh)
    ratio = span / max(1.0, float(side_px))
    # Wider band for three-quarter crops (pinna foreshortening)
    if ratio < 0.28 or ratio > 0.98:
        return False
    if min(bw, bh) < span * 0.18:
        return False
    mx, my = float(p[:, 0].mean()), float(p[:, 1].mean())
    if float(np.hypot(mx - tip[0], my - tip[1])) > float(side_px) * 0.58:
        return False
    pad_x, pad_y = 0.15 * bw, 0.15 * bh
    if tip[0] < x0 - pad_x or tip[0] > x1 + pad_x:
        return False
    if tip[1] < y0 - pad_y or tip[1] > y1 + pad_y:
        return False
    # Piercing (#56) should sit on/near the lobe — below tip when possible.
    pierce = p[min(55, p.shape[0] - 1)]
    tip_d = float(np.hypot(pierce[0] - tip[0], pierce[1] - tip[1]))
    if tip_d < 0.06 * float(side_px) or tip_d > 0.72 * float(side_px):
        return False
    # Soft: allow slight above-tip for angled views; hard-reject only if far above
    if pierce[1] < tip[1] - 0.08 * float(side_px):
        return False
    return True


def pierce_quality(
    pts: np.ndarray, tip: Tuple[float, float], side_px: float, score: float
) -> float:
    """Higher = better lobe placement + heatmap confidence (for flip pick)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    pierce = p[min(55, p.shape[0] - 1)]
    tip_d = float(np.hypot(pierce[0] - tip[0], pierce[1] - tip[1]))
    below = max(0.0, float(pierce[1] - tip[1])) / max(1.0, float(side_px))
    ratio = tip_d / max(1.0, float(side_px))
    # Ideal tip→pierce ≈ 0.22–0.40 of crop side, clearly below tip
    ratio_score = 1.0 - min(1.0, abs(ratio - 0.28) / 0.28)
    below_score = float(np.clip(below / 0.20, 0.0, 1.0))
    ok_bonus = 1.0 if landmarks_ok(pts, tip, side_px) else 0.0
    return float(0.45 * score + 0.25 * ratio_score + 0.20 * below_score + 0.10 * ok_bonus)


def build_crop_meta(
    tip: Tuple[float, float],
    pinna: float,
    side_name: str,
    kp: np.ndarray,
    frame_w: int,
    pad: float = CROP_PAD,
    prev: Optional[CropMeta] = None,
) -> CropMeta:
    """Mostly tip-centered square — light medial/down so lobe stays in crop."""
    mx, _my = medial_unit(tip, side_name, kp, frame_w)
    side_len = float(pinna * pad)
    # Medial toward face + downward so helix tip isn't the crop center (lobe clipped).
    ncx = tip[0] + mx * (0.10 * pinna)
    ncy = tip[1] + 0.17 * pinna
    if prev is None or prev.side_len <= 0:
        cx, cy, sl = ncx, ncy, side_len
    else:
        a = 0.45
        cx = (1 - a) * prev.cx + a * ncx
        cy = (1 - a) * prev.cy + a * ncy
        sl = (1 - a) * prev.side_len + a * side_len

    # Tip must stay well inside the square
    half = sl * 0.5
    if abs(tip[0] - cx) > half * 0.55 or abs(tip[1] - cy) > half * 0.55:
        cx = tip[0] + mx * (0.10 * pinna)
        cy = tip[1] + 0.06 * pinna

    side_i = max(32, int(round(sl)))
    ox = int(round(cx - side_i * 0.5))
    oy = int(round(cy - side_i * 0.5))
    return CropMeta(
        x0=ox,
        y0=oy,
        side=side_i,
        flipped=(side_name == "LEFT"),
        tip_x=float(tip[0]),
        tip_y=float(tip[1]),
        cx=float(cx),
        cy=float(cy),
        side_len=float(sl),
    )


def crop_with_meta(
    image_bgr: np.ndarray, meta: CropMeta, flip: Optional[bool] = None
) -> np.ndarray:
    """Extract + optional flip + resize to INPUT_SIZE."""
    do_flip = meta.flipped if flip is None else flip
    raw, _, _, _ = extract_square_crop(image_bgr, meta.cx, meta.cy, meta.side_len or meta.side)
    if do_flip:
        raw = cv2.flip(raw, 1)
    return cv2.resize(raw, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)


class EarCropper:
    """
    Prefer YOLO pose tip for side-face / full images.
    Fallback: use image center square (for already-tight ear photos).
    """

    LEFT_EAR_IDX = 3
    RIGHT_EAR_IDX = 4

    def __init__(
        self,
        yolo_path: Path | str | None = None,
        *,
        device: str | None = None,
        imgsz: int = 640,
        conf: float = 0.12,
    ) -> None:
        self._yolo = None
        self._prev: Optional[CropMeta] = None
        self.device = device or resolve_yolo_device()
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self._is_pt = False
        path = Path(yolo_path) if yolo_path else resolve_yolo_weights(prefer_pt=True)
        if path and path.is_file():
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(str(path), task="pose")
                self._is_pt = path.suffix.lower() == ".pt"
                print(f"[EarCropper] {path.name} device={self.device} imgsz={self.imgsz}")
            except Exception as exc:  # noqa: BLE001
                print(f"[EarCropper] YOLO unavailable ({exc}); using center crop fallback")

    def reset(self) -> None:
        self._prev = None

    def _predict(self, image_bgr: np.ndarray):
        assert self._yolo is not None
        kwargs: dict = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        if self._is_pt:
            kwargs["device"] = self.device
        return self._yolo.predict(image_bgr, **kwargs)

    def crop(
        self,
        image_bgr: np.ndarray,
        prefer_side: Optional[str] = None,
        *,
        allow_center_fallback: bool = True,
    ) -> tuple[Optional[np.ndarray], Optional[CropMeta]]:
        h, w = image_bgr.shape[:2]
        tip = None
        side_name = "RIGHT"
        kp = None
        box = None

        if self._yolo is not None:
            try:
                # Low conf so partial / three-quarter ears still produce keypoints
                results = self._predict(image_bgr)
                if results and results[0].keypoints is not None and len(results[0].boxes):
                    r0 = results[0]
                    boxes = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()
                    kpts = r0.keypoints.data.cpu().numpy()

                    best_i, best_score = 0, -1.0
                    for i in range(len(boxes)):
                        le_c = float(kpts[i][self.LEFT_EAR_IDX][2])
                        re_c = float(kpts[i][self.RIGHT_EAR_IDX][2])
                        # Prefer the detection with the stronger ear keypoint
                        score = max(le_c, re_c) * 0.85 + float(confs[i]) * 0.15
                        if score > best_score:
                            best_score, best_i = score, i

                    box = boxes[best_i]
                    kp = kpts[best_i]
                    le = kp[self.LEFT_EAR_IDX]
                    re = kp[self.RIGHT_EAR_IDX]

                    # Rank candidate ears; honor prefer_side as a soft boost
                    candidates: list[tuple[float, str, tuple[float, float]]] = []
                    for use_left, ear in ((True, le), (False, re)):
                        if float(ear[2]) < EAR_KEYPOINT_MIN_CONF:
                            continue
                        cand_side = "LEFT" if use_left else "RIGHT"
                        cand_tip = (float(ear[0]), float(ear[1]))
                        if not is_side_profile(kp, cand_side, cand_tip):
                            continue
                        vis = ear_visibility_score(kp, cand_side)
                        if prefer_side == "left" and use_left:
                            vis += 0.15
                        elif prefer_side == "right" and not use_left:
                            vis += 0.15
                        candidates.append((vis, cand_side, cand_tip))

                    if candidates:
                        candidates.sort(key=lambda t: t[0], reverse=True)
                        _, side_name, tip = candidates[0]
            except Exception as exc:  # noqa: BLE001
                print(f"[EarCropper] YOLO crop failed: {exc}")

        if tip is None or kp is None or box is None:
            # Live: never fall back to a face-sized center crop
            if not allow_center_fallback:
                if self._prev is not None:
                    meta = self._prev
                    return crop_with_meta(image_bgr, meta), meta
                return None, None
            side = int(min(h, w))
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            meta = CropMeta(
                x0=x0,
                y0=y0,
                side=side,
                flipped=False,
                tip_x=float(x0 + side / 2.0),
                tip_y=float(y0 + side / 2.0),
                cx=float(x0 + side / 2.0),
                cy=float(y0 + side / 2.0),
                side_len=float(side),
            )
            self._prev = meta
            crop = crop_with_meta(image_bgr, meta, flip=False)
            return crop, meta

        pinna = estimate_pinna_h(kp, box, tip, (h, w))
        meta = build_crop_meta(tip, pinna, side_name, kp, w, pad=CROP_PAD, prev=self._prev)
        self._prev = meta
        crop = crop_with_meta(image_bgr, meta)
        return crop, meta

    def tip_only(
        self, image_bgr: np.ndarray, prefer_side: Optional[str] = None
    ) -> Optional[Tuple[Tuple[float, float], str, np.ndarray, np.ndarray]]:
        """Cheap-ish tip refresh for tip-hold between SHG ticks. Returns tip, side, kp, box."""
        if self._yolo is None:
            return None
        try:
            results = self._predict(image_bgr)
            if not (results and results[0].keypoints is not None and len(results[0].boxes)):
                return None
            r0 = results[0]
            boxes = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            kpts = r0.keypoints.data.cpu().numpy()
            best_i, best_score = 0, -1.0
            for i in range(len(boxes)):
                le_c = float(kpts[i][self.LEFT_EAR_IDX][2])
                re_c = float(kpts[i][self.RIGHT_EAR_IDX][2])
                score = max(le_c, re_c) * 0.85 + float(confs[i]) * 0.15
                if score > best_score:
                    best_score, best_i = score, i
            box = boxes[best_i]
            kp = kpts[best_i]
            le = kp[self.LEFT_EAR_IDX]
            re = kp[self.RIGHT_EAR_IDX]
            candidates: list[tuple[float, str, tuple[float, float]]] = []
            for use_left, ear in ((True, le), (False, re)):
                if float(ear[2]) < EAR_KEYPOINT_MIN_CONF:
                    continue
                cand_side = "LEFT" if use_left else "RIGHT"
                cand_tip = (float(ear[0]), float(ear[1]))
                if not is_side_profile(kp, cand_side, cand_tip):
                    continue
                vis = ear_visibility_score(kp, cand_side)
                if prefer_side == "left" and use_left:
                    vis += 0.15
                elif prefer_side == "right" and not use_left:
                    vis += 0.15
                candidates.append((vis, cand_side, cand_tip))
            if not candidates:
                return None
            candidates.sort(key=lambda t: t[0], reverse=True)
            _, side_name, tip = candidates[0]
            return tip, side_name, kp, box
        except Exception:
            return None
