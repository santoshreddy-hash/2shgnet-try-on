# 2shgnet-try-on (SHGNet-56)

Ear landmark + piercing (#56) pipeline: annotate → train → export ONNX → live/web try-on.

## Folder structure

```
2shgnet-try-on/
├── annotator/              # Gradio: click piercing → write landmark #56
├── train/                  # Dataset, model, train, eval, ONNX, aug export
├── live/                   # Desktop Gradio / OpenCV ONNX live
├── tracking/               # One-Euro filter + tip-stick helpers
├── web/                    # Browser WASM demo (onnxruntime-web)
├── data/
│   └── data/
│       ├── ear_pose/       # ★ primary YOLO-pose dataset
│       │   ├── images/{train,val}/   # ear crop images (.png/.jpg)
│       │   └── labels/{train,val}/   # matching .txt (56 keypoints)
│       └── ibug_crops/     # optional collectiona crops + .pts
├── models/
│   ├── shgnet/
│   │   ├── SHGNet-56_final.pth   # ★ train from this (PyTorch)
│   │   └── SHGNet-56.onnx        # live / web inference
│   ├── yolo26n-pose.onnx         # ear tip detector
│   └── yolo/                     # optional alt YOLO path
├── outputs/
│   ├── checkpoints/        # best_stage*.pth, SHGNet-56_final.pth
│   ├── onnx/               # export copies
│   └── augmented/          # optional offline aug dump
├── run_pipeline.py         # annotate | train | export | all
├── requirements.txt
├── WORKFLOW.md             # step-by-step working flow
└── SHGNet56_Pipeline_Guide.pdf
```

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# NVIDIA GPU (example):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Place assets (required for train / live)

| Asset | Path |
|-------|------|
| Train images | `data/data/ear_pose/images/train/` |
| Train labels (56 kpts) | `data/data/ear_pose/labels/train/` |
| Val images/labels | `.../images/val/` + `.../labels/val/` |
| SHGNet-56 weights | `models/shgnet/SHGNet-56_final.pth` |
| YOLO detector | `models/yolo26n-pose.onnx` |

Label line (YOLO-pose): `class cx cy w h` + 56 × `(x y v)` normalized 0–1.

### Windows: wire your existing pack + `.pth` (recommended)

If you already have:

- `dataset annotated\datasetr annotated\images` (+ `labels`)
- `SHGNet-56_final.pth` at the repo root

run (PowerShell, repo root):

```powershell
.\scripts\wire_local_windows.ps1
# or:
python .\scripts\wire_local_dataset.py `
  --pack ".\dataset annotated\datasetr annotated" `
  --checkpoint ".\SHGNet-56_final.pth"
```

This hardlinks (or copies) into `data/data/ear_pose/` and `models/shgnet/`.

If you have `labels.zip` only (train/val `.txt`):

```bash
unzip -qo labels.zip -d /tmp/lb
cp -a /tmp/lb/labels/train/. data/data/ear_pose/labels/train/
cp -a /tmp/lb/labels/val/.   data/data/ear_pose/labels/val/
# still add matching images under images/{train,val}/
```

## Quick commands

```bash
# Annotate piercing #56
python annotator/app.py --folder data/data/ear_pose/images/train

# Fine-tune from SHGNet-56 (stage 2 + 3) — default
python -m train.train --device cuda

# Export ONNX
python -m train.export_onnx

# Live (desktop)
python -m live.desktop_onnx
# or Gradio: python live/app.py

# Browser
cd web && npm install && npm start
```

See **[WORKFLOW.md](WORKFLOW.md)** for the full end-to-end flow.

## Notes

- Default train loads **SHGNet-56** (not 55→56 expand). Use `--from-55` only for legacy.
- Online aug during train is random (in-memory). Offline 44-variant export is optional and writes to disk.
- Datasets and weight binaries are gitignored — supply locally.
