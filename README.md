# 2shgnet-try-on (SHGNet-56)

Ear landmark + piercing (#56) pipeline: annotate → train → export ONNX → live/web try-on.

## Folder structure

```
2shgnet-try-on/
├── annotator/                 # Gradio: click piercing → landmark #56
├── train/                     # Dataset, model, train, eval, ONNX export
├── live/                      # Desktop Gradio / OpenCV ONNX live
├── tracking/                  # One-Euro filter helpers
├── web/                       # Browser WASM demo
├── scripts/                   # wire_local_*, smoke_train_*, build_training_report
├── docs/                      # Training report PDF/MD + pipeline guide
├── data/data/ear_pose/        # ★ primary YOLO-pose dataset
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── models/
│   ├── shgnet/
│   │   ├── SHGNet-56_final.pth   # ★ trained PyTorch weights
│   │   └── SHGNet-56.onnx        # ★ live / web inference
│   └── yolo/                     # yolo26n-pose.onnx (ear tip)
├── outputs/
│   ├── checkpoints/           # best_stage*.pth, SHGNet-56_final.pth
│   ├── onnx/                  # export copy
│   ├── logs/                  # train logs (gitignored)
│   └── train_results.json
├── local_assets/              # zips, old init .pth, samples (gitignored)
├── run_pipeline.py
├── requirements.txt
├── WORKFLOW.md
└── one_euro_settings.json
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
| Images | ear crops already under `data/data/ear_pose/images/` |
| Labels | `local_assets/labels.zip` **or** pack `labels/` |
| Weights | `models/shgnet/SHGNet-56_final.pth` (trained) |

```powershell
# Windows (PowerShell, repo root)
.\scripts\wire_local_windows.ps1

# or any OS:
python scripts/wire_local_dataset.py
python scripts/wire_local_dataset.py `
  --checkpoint models/shgnet/SHGNet-56_final.pth `
  --labels-zip local_assets/labels.zip `
  --skip-images
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
