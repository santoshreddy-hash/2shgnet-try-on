"""SHGNet-56 training configuration."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths — primary training data = ear_pose (YOLO images/ + labels/)
DATA_ROOT = ROOT / "data" / "data"
IBUG_ROOT = DATA_ROOT / "ibug_ears"
IBUG_CROPS = DATA_ROOT / "ibug_crops"
EAR_POSE_ROOT = Path(
    "/Users/santoshreddy/Documents/virtual try on/ear_landmark_live/data/ear_pose"
)
# Annotated AudioEar / ear_pose train split (1700 imgs, landmark #56 in .txt)
DATA_IMAGES = EAR_POSE_ROOT / "images" / "train"
DATA_LABELS = EAR_POSE_ROOT / "labels" / "train"
DATA_ANNOTATIONS = DATA_LABELS  # YOLO .txt labels (not sibling .pts)
DEFAULT_ANNOTATE_DIR = DATA_IMAGES
DATA_IBUG = IBUG_ROOT
# Legacy iBUG crop folder (annotator / older runs)
IBUG_CROP_TRAIN = IBUG_CROPS / "collectiona_train"
PRETRAINED_55 = ROOT / "models" / "shgnet" / "hourglass_2stack_best.pth"
YOLO_ONNX = ROOT / "models" / "yolo26n-pose.onnx"
OUTPUTS = ROOT / "outputs"
CKPT_DIR = OUTPUTS / "checkpoints"
ONNX_DIR = OUTPUTS / "onnx"
ONNX_EXPORT = ONNX_DIR / "SHGNet-56.onnx"

# Model I/O
INPUT_SIZE = 256
HEATMAP_SIZE = 64
NUM_LANDMARKS_55 = 55
NUM_LANDMARKS_56 = 56
PIERCING_INDEX = 55  # 0-based → landmark #56
GAUSSIAN_SIGMA = 2.0

# Crop (jewellery-aligned tip-centered full-ear)
CROP_PAD = 1.65
EAR_KEYPOINT_MIN_CONF = 0.30

# Training stages
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 20
STAGE3_EPOCHS = 15
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
BATCH_SIZE = 8
NUM_WORKERS = 0  # MPS / spawn cannot pickle CUDA/MPS models in Dataset
VAL_SPLIT = 0.15
SEED = 42

# Augmentation ranges
AUG_ROTATION_DEG = 15.0
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.1
AUG_TRANSLATE_FRAC = 0.05
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_BLUR_PROB = 0.3
AUG_FLIP_PROB = 0.5

# Validation
PCK_THRESHOLDS = (0.02, 0.05, 0.10)  # fraction of crop diagonal

# Temporal smoothing — same One Euro values as ear jewellery virtual try-on
# (ear_landmark_live/config.py + one_euro_settings.json)
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.25
ONE_EURO_D_CUTOFF = 1.19
ONE_EURO_REST_SPEED_PX = 0.0  # disable rest freeze (was sticking landmarks)
ONE_EURO_REST_HOLD_FRAMES = 3
ONE_EURO_REST_RELEASE_MULT = 2.0
ONE_EURO_MAX_STEP_PX = 20.0

# Live FPS (match ear jewellery virtual try-on)
CAMERA_FPS_MIN = 20  # floor
CAMERA_FPS_MAX = 30  # ceiling
CAMERA_FPS = CAMERA_FPS_MAX  # default pace target
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
