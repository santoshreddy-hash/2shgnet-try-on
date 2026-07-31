# SHGNet-56 — working flow

## Overview

```
Images + labels (#56)
        │
        ▼
   Dataset loader ── online aug (in RAM) ──► Trainer (stage 2→3)
        │                                         │
        │                                         ▼
        │                              outputs/checkpoints/
        │                              SHGNet-56_final.pth
        │                                         │
        │                                         ▼
        │                                    export ONNX
        │                                         │
        └─────────────────────────────────────────┼──────────────┐
                                                  ▼              ▼
                                    live/ (YOLO crop + ONNX)   web/ (WASM)
```

## 1. Prepare folder layout

```
data/data/ear_pose/
  images/train/   *.png|jpg     ← required
  labels/train/   *.txt         ← same stems, 56 keypoints
  images/val/                   ← optional (else 15% split from train)
  labels/val/

models/shgnet/SHGNet-56_final.pth   ← required to fine-tune
models/yolo26n-pose.onnx            ← required for live crop
```

### One-shot wire from Windows pack

```powershell
# From D:\try on proj\2shgnet-try-on
.\scripts\wire_local_windows.ps1
```

Expects:

- `.\SHGNet-56_final.pth`
- `.\dataset annotated\datasetr annotated\images`
- `.\dataset annotated\datasetr annotated\labels`

**Label format:** one line per ear  
`0 cx cy w h  x1 y1 v1 … x56 y56 v56` (coords 0–1).  
Piercing is keypoint **#56** (index 55). Samples without #56 are skipped.

## 2. Annotate (if labels lack piercing)

```bash
python annotator/app.py --folder data/data/ear_pose/images/train --port 7860
```

Click piercing → Save → writes #56 into the matching YOLO `.txt`.

## 3. Train

Default = fine-tune existing **SHGNet-56** (stage 2 + stage 3):

```bash
python -m train.train --device cuda
# equivalents:
python run_pipeline.py train --device cuda
```

| Stage | What trains | Default epochs | LR |
|-------|-------------|----------------|-----|
| 2 | Last hourglass + heads | 20 | 1e-4 |
| 3 | Full network | 15 | 1e-5 |

Useful flags:

```bash
python -m train.train --device cuda --batch-size 16 \
  --stage2-epochs 12 --stage3-epochs 6 \
  --checkpoint models/shgnet/SHGNet-56_final.pth

# Legacy only (expand 55 → 56, then stages 1–3):
python -m train.train --from-55 --device cuda
```

**Outputs**

- `outputs/checkpoints/best_stage2.pth`
- `outputs/checkpoints/best_stage3.pth`
- `outputs/checkpoints/SHGNet-56_final.pth`
- `outputs/train_results.json`

Online augmentation (flip / rotate / scale / …) runs **in memory** — no augmented files on disk.

### On-the-fly 45 variants (recommended for full train)

```bash
python -m train.train --device cuda --variants-per-image 45
```

Effective train size ≈ `N_train × 45` (e.g. 2200 × 45 ≈ 99k samples/epoch).

Smoke (validate only, 1 train step — no full epochs):

```bash
python scripts/smoke_train_45aug.py
```

## 4. Export ONNX

```bash
python -m train.export_onnx \
  --checkpoint outputs/checkpoints/SHGNet-56_final.pth \
  --out models/shgnet/SHGNet-56.onnx
```

## 5. Live inference

Per frame:

1. Camera / image  
2. YOLO pose → ear tip → square tip-centered crop (LEFT flipped)  
3. Resize 256×256 → SHGNet-56 → 56 landmarks  
4. Remap to full frame → One Euro smooth → draw  

```bash
python -m live.desktop_onnx
python live/app.py
cd web && npm install && npm start   # http://127.0.0.1:8765
```

## 6. Optional offline aug export

Writes additive variants to disk (docs / extra data). Prefer online aug for normal training.

```bash
python -m train.export_augmented_dataset --only ear_pose --out outputs/augmented
```

## Checklist before `train`

- [ ] `data/data/ear_pose/images/train` has images  
- [ ] Matching `labels/train/*.txt` with **56** keypoints  
- [ ] `models/shgnet/SHGNet-56_final.pth` is a real binary (~tens of MB)  
- [ ] CUDA torch installed if using GPU  
- [ ] `python -m train.train --device cuda`  

## Config entry points

| Setting | File / symbol |
|---------|----------------|
| Data root | `train/config.py` → `EAR_POSE_ROOT` |
| Train images | `DATA_IMAGES` |
| SHGNet-56 .pth | `PRETRAINED_56` / `resolve_pretrained_56()` |
| YOLO ONNX | `YOLO_ONNX` |
| Checkpoints | `CKPT_DIR` = `outputs/checkpoints` |
