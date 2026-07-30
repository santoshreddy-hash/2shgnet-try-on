"""Dataset: ear crop + 55 LM + landmark #56 piercing GT.

Supports:
  - YOLO ear_pose: images/train + labels/train/*.txt (normalized kpts)
  - iBUG-style: sibling .pts next to images
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from train.annotations import (
    has_landmark_56,
    list_images,
    load_annotation,
    pts_path_for,
)
from train.augment import AugmentConfig, augment_sample
from train.config import (
    DATA_ANNOTATIONS,
    DATA_IMAGES,
    INPUT_SIZE,
    NUM_LANDMARKS_55,
    NUM_LANDMARKS_56,
    PIERCING_INDEX,
    PRETRAINED_55,
)
from train.crop import EarCropper, remap_points_to_crop
from train.heatmaps import generate_gaussian_heatmaps
from train.shgnet_base import SHGNetEarLandmarker, preprocess_ear_bgr
from train.yolo_pose_labels import (
    get_piercing_px,
    is_yolo_pose_images_dir,
    label_path_for,
    labels_dir_for_images,
    list_yolo_pose_images,
    read_yolo_pose,
)


def _yolo_landmarks_px(image_path: Path, labels_dir: Optional[Path] = None) -> np.ndarray:
    """Load up to 56 landmarks from YOLO label as (56, 2) pixel coords."""
    header, kpts = read_yolo_pose(label_path_for(image_path, labels_dir))
    del header
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    out = np.full((NUM_LANDMARKS_56, 2), np.nan, dtype=np.float32)
    n = min(len(kpts), NUM_LANDMARKS_56)
    out[:n, 0] = kpts[:n, 0] * w
    out[:n, 1] = kpts[:n, 1] * h
    return out


class Piercing56Dataset(Dataset):
    """
    For each annotated image:
      YOLO ear_pose: resize square ear image → 256, scale all 56 GT landmarks
      iBUG .pts: crop / resize as before
      Build 56 Gaussian heatmaps; optional augment
    """

    def __init__(
        self,
        image_names: List[str],
        img_dir: Path = DATA_IMAGES,
        ann_dir: Path = DATA_ANNOTATIONS,
        augment: bool = False,
        cache_dir: Optional[Path] = None,
        fill_55_with_pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.augment = augment
        self.aug_cfg = AugmentConfig()
        self.mode = "yolo" if is_yolo_pose_images_dir(self.img_dir) else "pts"
        self.labels_dir = (
            labels_dir_for_images(self.img_dir) if self.mode == "yolo" else None
        )

        self.names: List[str] = []
        for n in image_names:
            p = self.img_dir / n
            if self.mode == "yolo":
                if get_piercing_px(p, self.labels_dir) is not None:
                    self.names.append(n)
            elif has_landmark_56(p):
                self.names.append(n)

        self.cropper = EarCropper()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._fill_55 = fill_55_with_pretrained and PRETRAINED_55.is_file()
        self._landmarker_device = "cpu"
        self.landmarker: Optional[SHGNetEarLandmarker] = None

    def __len__(self) -> int:
        return len(self.names)

    def _cached_55(self, stem: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{stem}_lm55.npy"
        if p.is_file():
            return np.load(p)
        return None

    def _save_cached_55(self, stem: str, pts: np.ndarray) -> None:
        if self.cache_dir:
            np.save(self.cache_dir / f"{stem}_lm55.npy", pts.astype(np.float32))

    def _get_landmarker(self) -> Optional[SHGNetEarLandmarker]:
        if not self._fill_55:
            return None
        if self.landmarker is None:
            self.landmarker = SHGNetEarLandmarker(
                str(PRETRAINED_55), device=self._landmarker_device
            )
        return self.landmarker

    def _fill_55_from_pretrained(self, crop_bgr: np.ndarray, stem: str) -> np.ndarray:
        cached = self._cached_55(stem)
        if cached is not None and cached.shape == (NUM_LANDMARKS_55, 2):
            return cached.astype(np.float32)
        lm = self._get_landmarker()
        if lm is not None:
            pts55 = lm.predict(crop_bgr).astype(np.float32)
            self._save_cached_55(stem, pts55)
            return pts55
        pts = np.zeros((NUM_LANDMARKS_55, 2), dtype=np.float32)
        for i in range(NUM_LANDMARKS_55):
            ang = 2 * np.pi * i / NUM_LANDMARKS_55
            pts[i] = [
                INPUT_SIZE / 2 + 40 * np.cos(ang),
                INPUT_SIZE / 2 + 60 * np.sin(ang),
            ]
        return pts

    def _getitem_yolo(self, name: str, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (crop_256_bgr, landmarks_56 in 256 coords)."""
        pts_full = _yolo_landmarks_px(self.img_dir / name, self.labels_dir)
        h, w = image.shape[:2]
        crop = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        scale_x = INPUT_SIZE / float(w)
        scale_y = INPUT_SIZE / float(h)
        landmarks = np.zeros((NUM_LANDMARKS_56, 2), dtype=np.float32)
        landmarks[:, 0] = pts_full[:, 0] * scale_x
        landmarks[:, 1] = pts_full[:, 1] * scale_y
        if not np.isfinite(landmarks[:NUM_LANDMARKS_55]).all():
            landmarks[:NUM_LANDMARKS_55] = self._fill_55_from_pretrained(
                crop, Path(name).stem
            )
        if not np.isfinite(landmarks[PIERCING_INDEX]).all():
            raise ValueError(f"Missing piercing GT for {name}")
        return crop, landmarks

    def _getitem_pts(self, name: str, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_dir / name
        ann = load_annotation(pts_path_for(img_path))
        raw = ann["landmarks"]
        already_cropped = img_path.with_suffix(".crop_meta.json").is_file()
        h, w = image.shape[:2]

        if already_cropped:
            crop = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            scale_x = INPUT_SIZE / float(w)
            scale_y = INPUT_SIZE / float(h)
            landmarks = np.zeros((NUM_LANDMARKS_56, 2), dtype=np.float32)
            for i in range(NUM_LANDMARKS_56):
                pt = raw[i]
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    landmarks[i, 0] = float(pt[0]) * scale_x
                    landmarks[i, 1] = float(pt[1]) * scale_y
                else:
                    landmarks[i] = np.nan
            if not np.isfinite(landmarks[:NUM_LANDMARKS_55]).all():
                landmarks[:NUM_LANDMARKS_55] = self._fill_55_from_pretrained(
                    crop, Path(name).stem
                )
            return crop, landmarks

        piercing_full = np.array(raw[PIERCING_INDEX], dtype=np.float32)
        crop, meta = self.cropper.crop(image)
        has_gt55 = all(
            isinstance(raw[i], (list, tuple)) and len(raw[i]) == 2
            for i in range(NUM_LANDMARKS_55)
        )
        if has_gt55:
            pts55_full = np.array(raw[:NUM_LANDMARKS_55], dtype=np.float32)
            pts55_crop = remap_points_to_crop(
                pts55_full,
                meta.x0,
                meta.y0,
                meta.side,
                INPUT_SIZE,
                flipped=meta.flipped,
            )
        else:
            pts55_crop = self._fill_55_from_pretrained(crop, Path(name).stem)
        piercing_crop = remap_points_to_crop(
            piercing_full.reshape(1, 2),
            meta.x0,
            meta.y0,
            meta.side,
            INPUT_SIZE,
            flipped=meta.flipped,
        )[0]
        landmarks = np.zeros((NUM_LANDMARKS_56, 2), dtype=np.float32)
        landmarks[:NUM_LANDMARKS_55] = pts55_crop
        landmarks[PIERCING_INDEX] = piercing_crop
        return crop, landmarks

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_path = self.img_dir / name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)

        if self.mode == "yolo":
            crop, landmarks = self._getitem_yolo(name, image)
        else:
            crop, landmarks = self._getitem_pts(name, image)

        if self.augment:
            crop, landmarks = augment_sample(crop, landmarks, self.aug_cfg)

        heatmaps = generate_gaussian_heatmaps(landmarks)
        tensor = preprocess_ear_bgr(crop, INPUT_SIZE).squeeze(0)

        return {
            "image": tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "landmarks": torch.from_numpy(landmarks.astype(np.float32)),
            "name": name,
        }


def discover_annotated(img_dir: Path = DATA_IMAGES, ann_dir: Path = DATA_ANNOTATIONS) -> List[str]:
    del ann_dir
    img_dir = Path(img_dir)
    if is_yolo_pose_images_dir(img_dir):
        labels_dir = labels_dir_for_images(img_dir)
        return [
            p.name
            for p in list_yolo_pose_images(img_dir)
            if get_piercing_px(p, labels_dir) is not None
        ]
    names = []
    for p in list_images(img_dir):
        if has_landmark_56(p):
            names.append(p.name)
    return names


def train_val_split(
    names: List[str], val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    if len(names) <= 1:
        return names, names
    n_val = max(1, int(round(len(names) * val_ratio)))
    n_val = min(n_val, len(names) - 1)
    return names[n_val:], names[:n_val]
