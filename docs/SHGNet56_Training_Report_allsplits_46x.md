# SHGNet-56 Training Report

**Run name:** `allsplits_46x_s123`  
**Generated:** 2026-08-01 21:02  
**Project:** 2shgnet-try-on (ear landmarks + piercing #56)  
**Status:** Training **completed** (stages 1 → 2 → 3)

---

## 1. Executive summary

This run fine-tuned a pretrained **SHGNet-56** hourglass on the full ear-pose dataset with **on-the-fly augmentation** (1 original + 45 structured variants = **46 samples per image per epoch**), using a **3-stage** unfreeze schedule.

| Metric | Start (Stage 1 ep1) | Best Stage 1 | Best Stage 2 | Best Stage 3 (final) |
|--------|---------------------|--------------|--------------|----------------------|
| Pierce error (px) | 4.13 | 3.98 | 1.24 | **1.14** |
| Landmark NME | 0.0148 | 0.0143 | 0.0051 | **0.0048** |
| PCK@0.05 | 0.979 | 0.983 | 0.997 | **0.998** |
| Train loss | 0.00065 | — | — | 0.00005 |

**Headline result:** piercing localization improved from ~**4.1 px** (Stage 1) to ~**1.14 px** (best Stage 3), with PCK@0.05 ≈ **0.998**.

Final weights: `outputs/checkpoints/SHGNet-56_final.pth` (from `best_stage3.pth`).

---

## 2. Training command

```powershell
cd "D:\try on proj\2shgnet-try-on"
.\.venv\Scripts\python.exe -u -m train.train `
  --device cuda `
  --variants-per-image 45 `
  --include-original `
  --all-splits `
  --train-on-all `
  --full-stages `
  --batch-size 16 `
  --run-name allsplits_46x_s123
```

| Flag | Effect |
|------|--------|
| `--device cuda` | Train on NVIDIA GPU (RTX 4090) |
| `--variants-per-image 45` | On-the-fly additive variants from `train/online_variants.py` |
| `--include-original` | Also train the unaugmented image → **46** samples/image |
| `--all-splits` | Pool `images/train` + `images/val` (+ `test` if present) |
| `--train-on-all` | No hold-out from training; val metrics on originals of the full set |
| `--full-stages` | Stage 1 (heads) → 2 (last hourglass) → 3 (full net) from SHGNet-56 |
| `--batch-size 16` | Mini-batch size |

---

## 3. Dataset

| Split folder | Annotated images |
|--------------|-----------------:|
| `data/data/ear_pose/images/train` | 2,200 |
| `data/data/ear_pose/images/val` | 300 |
| `test` | (none) |
| **Total** | **2,500** |

**Label format (YOLO-pose):**  
`class cx cy w h` + 56 × `(x y v)` normalized to [0, 1].  
Piercing is landmark **#56** (0-based index **55**).

**Per-epoch sample count:**

| Item | Count |
|------|------:|
| Base images | 2,500 |
| Samples / image | 46 (1 original + 45 augs) |
| Train set length | **115,000** |
| Steps / epoch (bs=16) | **7,188** |
| Val set (originals only) | 2,500 |

### Online augmentation families (45 additive variants)

| Family | Count | Notes |
|--------|------:|-------|
| Flip | 2 | incl. identity / horizontal |
| Rotation | 7 | ≈ ±15° |
| Scale | 5 | ≈ 0.85–1.15 |
| Translate | 9 | small shifts |
| Brightness | 5 | |
| Contrast | 5 | |
| Blur | 3 | kernels 3/5/7 |
| Noise / JPEG | 4 | |
| Occlusion (hair-like) | 5 | |
| **Total** | **45** | + original = **46** |

Variants are applied **in RAM** (no disk dump). Validation uses **originals only**.

---

## 4. Model and stages

**Architecture:** LDNet / stacked hourglass, **2 stacks**, input **256×256**, heatmaps **64×64**, **56** landmark channels.

**Init checkpoint:** `models/shgnet/SHGNet-56_final.pth` (pretrained before this run).

| Stage | What trains | Epochs | LR | Trainable params (this run) |
|-------|-------------|-------:|-----:|----------------------------:|
| 1 | Heatmap output heads only | 30 | 1e-3 | 28,784 |
| 2 | Last hourglass + features + heads | 20 | 1e-4 | 4,019,568 |
| 3 | Full network | 15 | 1e-5 | 8,459,888 |

**Loss:** heatmap MSE (Gaussian targets, σ=2).  
**Best checkpoint selection:** lowest `piercing_point_error_px` on the val originals set.  
**No early stopping / patience** — fixed epoch counts.

**Hardware:** NVIDIA GeForce RTX 4090 (~24 GB), CUDA; observed ~4.5–5.7 it/s; ~26–30 min per epoch (train + val). Peak VRAM use was modest (~few GB) because the model is small and Stage 1 only trains heads.

---

## 5. Best metrics by stage

| Stage | Best ep | Train loss | NME | Pierce px | PCK@0.05 |
|-------|--------:|-----------:|----:|----------:|---------:|
| stage1 | 27 | 0.000619 | 0.0143 | 3.979 | 0.9827 |
| stage2 | 19 | 0.000074 | 0.0051 | 1.242 | 0.9974 |
| stage3 | 14 | 0.000046 | 0.0048 | 1.144 | 0.9975 |

### Improvement vs Stage 1 start

- Pierce px: 4.13 → **1.14** (~72.3% relative reduction)
- NME: 0.0148 → **0.0048**

---

## 6. Per-epoch logs

### Stage 1 (heads)

| Ep | Train loss | Val HM loss | NME | Pierce px | PCK@0.05 |
|---:|----------:|------------:|----:|----------:|---------:|
| 1 | 0.000650 | 0.000601 | 0.0148 | 4.130 | 0.9790 |
| 2 | 0.000639 | 0.000599 | 0.0147 | 4.179 | 0.9797 |
| 3 | 0.000636 | 0.000591 | 0.0146 | 4.130 | 0.9802 |
| 4 | 0.000633 | 0.000589 | 0.0146 | 4.057 | 0.9803 |
| 5 | 0.000631 | 0.000590 | 0.0145 | 4.104 | 0.9812 |
| 6 | 0.000630 | 0.000588 | 0.0145 | 3.986 | 0.9811 |
| 7 | 0.000629 | 0.000586 | 0.0145 | 4.079 | 0.9814 |
| 8 | 0.000628 | 0.000583 | 0.0145 | 4.080 | 0.9813 |
| 9 | 0.000627 | 0.000581 | 0.0144 | 4.006 | 0.9816 |
| 10 | 0.000626 | 0.000582 | 0.0144 | 4.110 | 0.9818 |
| 11 | 0.000626 | 0.000580 | 0.0144 | 4.002 | 0.9818 |
| 12 | 0.000625 | 0.000605 | 0.0144 | 4.099 | 0.9816 |
| 13 | 0.000624 | 0.000578 | 0.0144 | 4.091 | 0.9815 |
| 14 | 0.000624 | 0.000581 | 0.0144 | 4.050 | 0.9818 |
| 15 | 0.000623 | 0.000579 | 0.0144 | 4.074 | 0.9820 |
| 16 | 0.000623 | 0.000577 | 0.0143 | 4.087 | 0.9821 |
| 17 | 0.000623 | 0.000583 | 0.0143 | 4.119 | 0.9821 |
| 18 | 0.000622 | 0.000578 | 0.0143 | 4.078 | 0.9824 |
| 19 | 0.000622 | 0.000579 | 0.0143 | 4.108 | 0.9824 |
| 20 | 0.000621 | 0.000583 | 0.0143 | 4.058 | 0.9824 |
| 21 | 0.000621 | 0.000578 | 0.0143 | 4.068 | 0.9822 |
| 22 | 0.000621 | 0.000575 | 0.0143 | 3.991 | 0.9826 |
| 23 | 0.000621 | 0.000576 | 0.0143 | 4.039 | 0.9826 |
| 24 | 0.000621 | 0.000576 | 0.0143 | 4.039 | 0.9827 |
| 25 | 0.000620 | 0.000573 | 0.0142 | 4.101 | 0.9826 |
| 26 | 0.000620 | 0.000571 | 0.0143 | 4.059 | 0.9829 |
| 27 | 0.000619 | 0.000576 | 0.0143 | 3.979 | 0.9827 |
| 28 | 0.000619 | 0.000571 | 0.0142 | 4.015 | 0.9828 |
| 29 | 0.000619 | 0.000575 | 0.0143 | 4.140 | 0.9827 |
| 30 | 0.000619 | 0.000579 | 0.0143 | 4.122 | 0.9825 |

### Stage 2 (last hourglass)

| Ep | Train loss | Val HM loss | NME | Pierce px | PCK@0.05 |
|---:|----------:|------------:|----:|----------:|---------:|
| 1 | 0.000441 | 0.000265 | 0.0097 | 2.939 | 0.9934 |
| 2 | 0.000280 | 0.000158 | 0.0077 | 2.536 | 0.9951 |
| 3 | 0.000214 | 0.000124 | 0.0069 | 2.235 | 0.9959 |
| 4 | 0.000179 | 0.000102 | 0.0065 | 1.996 | 0.9964 |
| 5 | 0.000156 | 0.000084 | 0.0061 | 1.765 | 0.9968 |
| 6 | 0.000139 | 0.000076 | 0.0060 | 1.613 | 0.9969 |
| 7 | 0.000128 | 0.000066 | 0.0056 | 1.524 | 0.9971 |
| 8 | 0.000118 | 0.000064 | 0.0057 | 1.517 | 0.9970 |
| 9 | 0.000110 | 0.000057 | 0.0055 | 1.432 | 0.9972 |
| 10 | 0.000104 | 0.000055 | 0.0055 | 1.389 | 0.9972 |
| 11 | 0.000099 | 0.000055 | 0.0055 | 1.454 | 0.9974 |
| 12 | 0.000094 | 0.000049 | 0.0054 | 1.341 | 0.9972 |
| 13 | 0.000090 | 0.000046 | 0.0053 | 1.325 | 0.9973 |
| 14 | 0.000087 | 0.000044 | 0.0052 | 1.318 | 0.9974 |
| 15 | 0.000084 | 0.000044 | 0.0053 | 1.280 | 0.9974 |
| 16 | 0.000081 | 0.000042 | 0.0052 | 1.272 | 0.9975 |
| 17 | 0.000079 | 0.000041 | 0.0053 | 1.295 | 0.9974 |
| 18 | 0.000076 | 0.000038 | 0.0052 | 1.257 | 0.9974 |
| 19 | 0.000074 | 0.000037 | 0.0051 | 1.242 | 0.9974 |
| 20 | 0.000072 | 0.000037 | 0.0049 | 1.243 | 0.9979 |

### Stage 3 (full network)

| Ep | Train loss | Val HM loss | NME | Pierce px | PCK@0.05 |
|---:|----------:|------------:|----:|----------:|---------:|
| 1 | 0.000060 | 0.000025 | 0.0050 | 1.188 | 0.9974 |
| 2 | 0.000056 | 0.000025 | 0.0049 | 1.177 | 0.9975 |
| 3 | 0.000054 | 0.000024 | 0.0049 | 1.165 | 0.9974 |
| 4 | 0.000053 | 0.000023 | 0.0049 | 1.161 | 0.9974 |
| 5 | 0.000052 | 0.000023 | 0.0049 | 1.153 | 0.9974 |
| 6 | 0.000051 | 0.000022 | 0.0049 | 1.150 | 0.9974 |
| 7 | 0.000050 | 0.000022 | 0.0048 | 1.155 | 0.9975 |
| 8 | 0.000050 | 0.000022 | 0.0049 | 1.150 | 0.9974 |
| 9 | 0.000049 | 0.000022 | 0.0049 | 1.148 | 0.9975 |
| 10 | 0.000048 | 0.000021 | 0.0049 | 1.152 | 0.9975 |
| 11 | 0.000048 | 0.000021 | 0.0047 | 1.149 | 0.9977 |
| 12 | 0.000047 | 0.000021 | 0.0048 | 1.149 | 0.9974 |
| 13 | 0.000047 | 0.000021 | 0.0048 | 1.150 | 0.9974 |
| 14 | 0.000046 | 0.000020 | 0.0048 | 1.144 | 0.9975 |
| 15 | 0.000046 | 0.000021 | 0.0049 | 1.153 | 0.9974 |

---

## 7. Artifacts

| File | Role | Size (MB) |
|------|------|----------:|
| `outputs/checkpoints/best_stage1.pth` | checkpoint | 32.7 |
| `outputs/checkpoints/best_stage2.pth` | checkpoint | 32.7 |
| `outputs/checkpoints/best_stage3.pth` | checkpoint | 32.7 |
| `outputs/checkpoints/SHGNet-56_final.pth` | checkpoint | 32.7 |
| `outputs/checkpoints/last_stage1.pth` | checkpoint | 32.7 |
| `outputs/checkpoints/last_stage2.pth` | checkpoint | 32.7 |
| `outputs/checkpoints/last_stage3.pth` | checkpoint | 32.7 |
| `outputs/train_results.json` | full history + best dicts | — |
| `models/shgnet/SHGNet-56_final.pth` | (pre-run init; may be overwritten if re-wired) | — |

**Recommended for export / live:**

```powershell
.\.venv\Scripts\python.exe -m train.export_onnx `
  --checkpoint outputs/checkpoints/SHGNet-56_final.pth `
  --out models/shgnet/SHGNet-56.onnx
```

Then:

```powershell
.\.venv\Scripts\python.exe -m live.desktop_onnx
```

---

## 8. Pipeline diagram

```
Ear crops + YOLO labels (#56)
        │
        ▼
Pool train+val (2,500) ──► Piercing56Dataset
        │                     1 original + 45 online variants
        │                     → 115,000 samples / epoch
        ▼
Stage 1 (30 ep, lr 1e-3)  heads only
        ▼
Stage 2 (20 ep, lr 1e-4)  last hourglass + heads
        ▼
Stage 3 (15 ep, lr 1e-5)  full network
        ▼
best_stage3.pth → SHGNet-56_final.pth → ONNX → live / web try-on
```

---

## 9. Code entry points

| Topic | Path |
|-------|------|
| Train CLI | `train/train.py` |
| Dataset + 46× expansion | `train/dataset.py` |
| Online variants | `train/online_variants.py` |
| Config (epochs, LR, paths) | `train/config.py` |
| Freeze / unfreeze stages | `train/model.py` |
| Workflow notes | `WORKFLOW.md` |
| Results JSON | `outputs/train_results.json` |

---

## 10. Notes and caveats

1. **`--train-on-all`:** validation metrics are computed on originals of images that also appear in training (with augs). Absolute numbers are optimistic vs a strict held-out test set; relative improvement across stages remains meaningful.
2. **Stage 1 looked flat** because the network was already strong and only heads were trainable; **Stage 2** delivered the large pierce-error drop.
3. **No patience / early stop** — all 65 epochs ran to completion.
4. Re-running with a larger `--batch-size` (e.g. 32–64) may improve throughput on the 4090; this run used 16.

---

## 11. Conclusion

The `allsplits_46x_s123` run successfully completed 3-stage fine-tuning of SHGNet-56 on 2,500 ear images with 46× on-the-fly sampling. Piercing error improved to **~1.14 px** (best Stage 3) with **PCK@0.05 ≈ 0.998**. Artifacts are ready for ONNX export and live try-on.

*Report auto-generated from `outputs/train_results.json` by `scripts/build_training_report.py`.*
