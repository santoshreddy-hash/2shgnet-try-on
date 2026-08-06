"""SHGNet-56 training configuration."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Paths (repo-relative — no machine-specific absolutes)
# ---------------------------------------------------------------------------
DATA_ROOT = ROOT / "data" / "data"
IBUG_ROOT = DATA_ROOT / "ibug_ears"
IBUG_CROPS = DATA_ROOT / "ibug_crops"
EAR_POSE_ROOT = DATA_ROOT / "ear_pose"
# YOLO layout: images/{train,val} + labels/{train,val}
DATA_IMAGES = EAR_POSE_ROOT / "images" / "train"
DATA_LABELS = EAR_POSE_ROOT / "labels" / "train"
DATA_ANNOTATIONS = DATA_LABELS
DEFAULT_ANNOTATE_DIR = DATA_IMAGES
DATA_IBUG = IBUG_ROOT
IBUG_CROP_TRAIN = IBUG_CROPS / "collectiona_train"

# Weights
PRETRAINED_55 = ROOT / "models" / "shgnet" / "hourglass_2stack_best.pth"
PRETRAINED_56 = ROOT / "models" / "shgnet" / "SHGNet-56_final.pth"
YOLO_ONNX = ROOT / "models" / "yolo26n-pose.onnx"
# Prefer models/yolo/ if top-level stub missing
YOLO_ONNX_ALT = ROOT / "models" / "yolo" / "yolo26n-pose.onnx"

OUTPUTS = ROOT / "outputs"
CKPT_DIR = OUTPUTS / "checkpoints"
ONNX_DIR = OUTPUTS / "onnx"
ONNX_EXPORT = ROOT / "models" / "shgnet" / "SHGNet-56.onnx"
ONNX_EXPORT_ALT = ONNX_DIR / "SHGNet-56.onnx"


def resolve_pretrained_56() -> Path | None:
    """First real SHGNet-56 .pth (>1 MB; skips path-stub placeholders)."""
    candidates = (
        PRETRAINED_56,
        ROOT / "SHGNet-56_final.pth",
        CKPT_DIR / "SHGNet-56_final.pth",
        ROOT / "models" / "shgnet" / "SHGNet-56.pth",
        CKPT_DIR / "best_stage3.pth",
        CKPT_DIR / "best_stage2.pth",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


def resolve_onnx_export() -> Path:
    for path in (ONNX_EXPORT, ONNX_EXPORT_ALT):
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return ONNX_EXPORT


def resolve_yolo_onnx() -> Path:
    for path in (YOLO_ONNX, YOLO_ONNX_ALT):
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return YOLO_ONNX

# Model I/O
INPUT_SIZE = 256
HEATMAP_SIZE = 64
NUM_LANDMARKS_55 = 55
NUM_LANDMARKS_56 = 56
PIERCING_INDEX = 55  # 0-based → landmark #56
GAUSSIAN_SIGMA = 2.0

# Crop (jewellery-aligned tip-centered full-ear)
# Live needs ~2.1–2.3× pinna so lobe stays in-frame; 1.65 clipped #56 upward.
CROP_PAD = 2.15
EAR_KEYPOINT_MIN_CONF = 0.30

# Training stages
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 20
STAGE3_EPOCHS = 15
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
BATCH_SIZE = 16
NUM_WORKERS = 0  # keep 0 for CUDA/MPS DataLoader safety
VAL_SPLIT = 0.15
SEED = 42

# Online augmentation ranges (random mix per sample)
AUG_ROTATION_DEG = 15.0
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.1
AUG_TRANSLATE_FRAC = 0.05
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_BLUR_PROB = 0.3
AUG_FLIP_PROB = 0.5

# Validation
PCK_THRESHOLDS = (0.02, 0.05, 0.10)

# Temporal smoothing (One Euro) — stick to ear under head motion
ONE_EURO_MIN_CUTOFF = 1.8
ONE_EURO_BETA = 0.85
ONE_EURO_D_CUTOFF = 1.19
ONE_EURO_REST_SPEED_PX = 1.5
ONE_EURO_REST_HOLD_FRAMES = 2
ONE_EURO_REST_RELEASE_MULT = 1.5
ONE_EURO_MAX_STEP_PX = 42.0

# Live FPS — hard band for desktop + browser (never outside this range)
CAMERA_FPS_MIN = 20
CAMERA_FPS_MAX = 30
CAMERA_FPS = 25  # default setpoint (mid-band)
# Capture / process size (aspect preserved; quality default for desktop)
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
# Sparse inference defaults (dense enough for stable lock; tip-hold fills gaps)
LIVE_YOLO_EVERY = 1
LIVE_SHG_EVERY = 1
