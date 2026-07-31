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

### Wire local assets (recommended)

One command hardlinks/copies into `data/data/ear_pose/` + `models/shgnet/`:

| Source | Expected path |
|--------|----------------|
| Images | `dataset annotated\datasetr annotated\images` |
| Labels | `labels.zip` (repo root) **or** pack `labels/` |
| Weights | `SHGNet-56_final.pth` (repo root) |

```powershell
# Windows (PowerShell, repo root)
.\scripts\wire_local_windows.ps1

# or any OS:
python scripts/wire_local_dataset.py
python scripts/wire_local_dataset.py `
  --images "dataset annotated/datasetr annotated/images" `
  --checkpoint SHGNet-56_final.pth `
  --labels-zip labels.zip
```

The script prints a **readiness** summary (`paired` image/label count + checkpoint). Train only when it says `READY`.

If Cursor fails to switch branches with `git stash --include-untracked` / `could not write index`, your local dataset pack is too large to stash. In PowerShell from the repo root:

```powershell
# clear a stuck lock if present
Remove-Item -Force .git\index.lock -ErrorAction SilentlyContinue
git fetch origin
git checkout cursor/wire-local-assets-e1d3
# if checkout still refuses, keep local files and force the branch tip:
git switch -C cursor/wire-local-assets-e1d3 origin/cursor/wire-local-assets-e1d3 --discard-changes
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
