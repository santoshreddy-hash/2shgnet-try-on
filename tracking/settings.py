"""Load jewellery-matched One Euro settings (same values as ear_landmark_live)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train.config import (
    NUM_LANDMARKS_56,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_MAX_STEP_PX,
    ONE_EURO_MIN_CUTOFF,
    ONE_EURO_REST_HOLD_FRAMES,
    ONE_EURO_REST_RELEASE_MULT,
    ONE_EURO_REST_SPEED_PX,
)
from tracking.one_euro import OneEuroLandmarkFilter

_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "one_euro_settings.json"

_DEFAULTS: dict[str, Any] = {
    "min_cutoff": ONE_EURO_MIN_CUTOFF,
    "beta": ONE_EURO_BETA,
    "d_cutoff": ONE_EURO_D_CUTOFF,
    "rest_speed_px": ONE_EURO_REST_SPEED_PX,
    "rest_hold_frames": ONE_EURO_REST_HOLD_FRAMES,
    "rest_release_mult": ONE_EURO_REST_RELEASE_MULT,
    "max_step_px": ONE_EURO_MAX_STEP_PX,
}


def load_one_euro_settings() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if _SETTINGS_PATH.is_file():
        try:
            data = json.loads(_SETTINGS_PATH.read_text())
            for k in _DEFAULTS:
                if k in data:
                    cfg[k] = type(_DEFAULTS[k])(data[k])
        except Exception:
            pass
    return cfg


def make_landmark_filter(num_landmarks: int = NUM_LANDMARKS_56) -> OneEuroLandmarkFilter:
    """Same One Euro as jewellery try-on (56 landmarks including piercing)."""
    s = load_one_euro_settings()
    return OneEuroLandmarkFilter(
        num_landmarks=num_landmarks,
        min_cutoff=float(s["min_cutoff"]),
        beta=float(s["beta"]),
        d_cutoff=float(s["d_cutoff"]),
        rest_speed_px=float(s["rest_speed_px"]),
        rest_hold_frames=int(s["rest_hold_frames"]),
        rest_release_mult=float(s["rest_release_mult"]),
    )


def max_step_px() -> float:
    return float(load_one_euro_settings()["max_step_px"])
