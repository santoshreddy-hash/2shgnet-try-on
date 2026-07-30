# 2shgnet-try-on (SHGNet-56)

Shareable **source-only** drop of the SHGNet-56 / ear-landmark pipeline (train, annotate, live, tracking, web, augmentation export).  
**Datasets, virtualenvs, node_modules, checkpoints, ONNX binaries, and large outputs are not included** — supply those locally.

## What's included

| Path | Purpose |
|------|---------|
| `train/` | Training, eval, crop prep, ONNX export, **augmentation exporter** |
| `annotator/` | Gradio landmark annotator |
| `live/` | Desktop / ONNX live inference |
| `tracking/` | One-Euro filter + landmark stick helpers |
| `web/` | Browser demo (JS/HTML; run `npm install` yourself) |
| `run_pipeline.py` | CLI: `annotate` \| `train` \| `export` \| `all` |
| `requirements.txt` | Python dependencies |
| `one_euro_settings.json` | Tracking filter defaults |
| `models/shgnet/`, `models/yolo/` | Empty stubs (place weights here) |
| `data/`, `outputs/` | Empty placeholders |
| `export_augmented_dataset.py` | Thin wrapper → `train.export_augmented_dataset` |

## What's excluded

- `data/` image/label trees (place your own under `data/data/…` or edit paths in `train/config.py`)
- `.venv/`, `web/node_modules/`, `__pycache__/`, `.DS_Store`
- Heavy outputs: `outputs/augmented/`, `outputs/checkpoints/`, `outputs/onnx/`, caches, logs
- Weight files: `*.pth`, `*.pt`, `*.onnx` (not shipped)

## Install

```bash
cd "."  # this repo root

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

Optional conda:

```bash
conda create -n shgnet56 python=3.11 -y
conda activate shgnet56
pip install -r requirements.txt
```

Web demo (optional):

```bash
cd web && npm install
```

## Data & models (you must supply)

Expected layout after you copy assets (see also `train/config.py`):

```
data/data/
  ear_pose/images/{train,val}/…
  ear_pose/labels/{train,val}/…     # YOLO-pose .txt
  ibug_crops/collectiona_{train,test}/…   # optional

models/
  shgnet/hourglass_2stack_best.pth  # 55-landmark pretrained (example name)
  yolo26n-pose.onnx                 # detector (path in config: models/yolo26n-pose.onnx)
```

`train/config.py` may still contain **machine-specific absolute paths** for `EAR_POSE_ROOT` / defaults — point them at your local data before training.

## Augmentation export (44 variants / image)

Additive families (not a cartesian product): flip 2, scale 5, translate 7, blur 5, noise 6, occlusion 6, smoke_blur 5, brightness 4, contrast 4 → **44**.

Preferred (module):

```bash
# Expects data under data/data/ear_pose and/or data/data/ibug_crops
python -m train.export_augmented_dataset --only all
python -m train.export_augmented_dataset --only ear_pose --limit 1   # smoke test
python -m train.export_augmented_dataset --out outputs/augmented --only collectiona
```

Or via the top-level wrapper:

```bash
python export_augmented_dataset.py --only all
```

Outputs go to `outputs/augmented/<split_name>/` (`images/`, `labels/` when YOLO, `manifest.csv`, `summary.json`). Existing files are skipped (resumable). Missing input dirs are skipped with a message.

## Other entry points

```bash
python run_pipeline.py annotate
python run_pipeline.py train
python run_pipeline.py export          # ONNX (needs trained weights)
python -m train.train                  # direct train
python annotator/app.py
python live/app.py
```

## Notes for recipients

1. This pack is **code only** (~few hundred KB). Do not expect datasets or 16GB+ run outputs.
2. Install Python deps from `requirements.txt`; install web deps with `npm install` under `web/` if needed.
3. Place models under `models/` as described above; place datasets under `data/` (or edit `train/config.py`).
4. The live augmentation job on the original machine is independent of this folder.
