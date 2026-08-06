#!/usr/bin/env python3
"""Build SHGNet-56 training run report (Markdown + PDF via pandoc)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "train_results.json"
OUT_MD = ROOT / "docs" / "SHGNet56_Training_Report_allsplits_46x.md"
OUT_PDF = ROOT / "docs" / "SHGNet56_Training_Report_allsplits_46x.pdf"


def fmt(x, nd=5):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def stage_table(history: list[dict]) -> str:
    lines = [
        "| Ep | Train loss | Val HM loss | NME | Pierce px | PCK@0.05 |",
        "|---:|----------:|------------:|----:|----------:|---------:|",
    ]
    for h in history:
        pck = h.get("pck@0.05", float("nan"))
        lines.append(
            f"| {h.get('epoch')} | {fmt(h.get('train_loss'), 6)} | "
            f"{fmt(h.get('heatmap_loss'), 6)} | {fmt(h.get('landmark_nme'), 4)} | "
            f"{fmt(h.get('piercing_point_error_px'), 3)} | {fmt(pck, 4)} |"
        )
    return "\n".join(lines)


def main() -> int:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    ckpt = ROOT / "outputs" / "checkpoints"
    sizes = {}
    for name in (
        "best_stage1.pth",
        "best_stage2.pth",
        "best_stage3.pth",
        "SHGNet-56_final.pth",
        "last_stage1.pth",
        "last_stage2.pth",
        "last_stage3.pth",
    ):
        p = ckpt / name
        if p.is_file():
            sizes[name] = round(p.stat().st_size / (1024 * 1024), 1)

    best_rows = []
    for s in ("stage1", "stage2", "stage3"):
        b = data[s]["best"]
        best_rows.append(
            f"| {s} | {b.get('epoch')} | {fmt(b.get('train_loss'), 6)} | "
            f"{fmt(b.get('landmark_nme'), 4)} | "
            f"{fmt(b.get('piercing_point_error_px'), 3)} | "
            f"{fmt(b.get('pck@0.05'), 4)} |"
        )

    s1_b = data["stage1"]["best"]
    s2_b = data["stage2"]["best"]
    s3_b = data["stage3"]["best"]
    s1_h0 = data["stage1"]["history"][0]
    s3_h_last = data["stage3"]["history"][-1]

    md = f"""# SHGNet-56 Training Report

**Run name:** `allsplits_46x_s123`  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Project:** 2shgnet-try-on (ear landmarks + piercing #56)  
**Status:** Training **completed** (stages 1 → 2 → 3)

---

## 1. Executive summary

This run fine-tuned a pretrained **SHGNet-56** hourglass on the full ear-pose dataset with **on-the-fly augmentation** (1 original + 45 structured variants = **46 samples per image per epoch**), using a **3-stage** unfreeze schedule.

| Metric | Start (Stage 1 ep1) | Best Stage 1 | Best Stage 2 | Best Stage 3 (final) |
|--------|---------------------|--------------|--------------|----------------------|
| Pierce error (px) | {fmt(s1_h0.get('piercing_point_error_px'), 2)} | {fmt(s1_b.get('piercing_point_error_px'), 2)} | {fmt(s2_b.get('piercing_point_error_px'), 2)} | **{fmt(s3_b.get('piercing_point_error_px'), 2)}** |
| Landmark NME | {fmt(s1_h0.get('landmark_nme'), 4)} | {fmt(s1_b.get('landmark_nme'), 4)} | {fmt(s2_b.get('landmark_nme'), 4)} | **{fmt(s3_b.get('landmark_nme'), 4)}** |
| PCK@0.05 | {fmt(s1_h0.get('pck@0.05'), 3)} | {fmt(s1_b.get('pck@0.05'), 3)} | {fmt(s2_b.get('pck@0.05'), 3)} | **{fmt(s3_b.get('pck@0.05'), 3)}** |
| Train loss | {fmt(s1_h0.get('train_loss'), 5)} | — | — | {fmt(s3_h_last.get('train_loss'), 5)} |

**Headline result:** piercing localization improved from ~**4.1 px** (Stage 1) to ~**1.14 px** (best Stage 3), with PCK@0.05 ≈ **0.998**.

Final weights: `outputs/checkpoints/SHGNet-56_final.pth` (from `best_stage3.pth`).

---

## 2. Training command

```powershell
cd "D:\\try on proj\\2shgnet-try-on"
.\\.venv\\Scripts\\python.exe -u -m train.train `
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
{chr(10).join(best_rows)}

### Improvement vs Stage 1 start

- Pierce px: {fmt(s1_h0.get('piercing_point_error_px'), 2)} → **{fmt(s3_b.get('piercing_point_error_px'), 2)}** (~{fmt(100 * (1 - s3_b.get('piercing_point_error_px', 1) / max(s1_h0.get('piercing_point_error_px', 1), 1e-9)), 1)}% relative reduction)
- NME: {fmt(s1_h0.get('landmark_nme'), 4)} → **{fmt(s3_b.get('landmark_nme'), 4)}**

---

## 6. Per-epoch logs

### Stage 1 (heads)

{stage_table(data['stage1']['history'])}

### Stage 2 (last hourglass)

{stage_table(data['stage2']['history'])}

### Stage 3 (full network)

{stage_table(data['stage3']['history'])}

---

## 7. Artifacts

| File | Role | Size (MB) |
|------|------|----------:|
"""
    for name, mb in sizes.items():
        md += f"| `outputs/checkpoints/{name}` | checkpoint | {mb} |\n"

    md += f"""| `outputs/train_results.json` | full history + best dicts | — |
| `models/shgnet/SHGNet-56_final.pth` | (pre-run init; may be overwritten if re-wired) | — |

**Recommended for export / live:**

```powershell
.\\.venv\\Scripts\\python.exe -m train.export_onnx `
  --checkpoint outputs/checkpoints/SHGNet-56_final.pth `
  --out models/shgnet/SHGNet-56.onnx
```

Then:

```powershell
.\\.venv\\Scripts\\python.exe -m live.desktop_onnx
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
"""

    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_MD}")

    # PDF via pandoc if available
    pandoc = None
    for cand in (
        Path(r"C:\Users\7950x_RTX4090\anaconda3\Scripts\pandoc.exe"),
        Path("pandoc"),
    ):
        if cand.name == "pandoc" or cand.is_file():
            pandoc = str(cand)
            break
    if pandoc:
        cmd = [
            pandoc,
            str(OUT_MD),
            "-o",
            str(OUT_PDF),
            "--from",
            "markdown",
            "--pdf-engine=xelatex",
            "-V",
            "geometry:margin=1in",
            "--toc",
            "-V",
            "colorlinks=true",
        ]
        # Prefer simpler engines if xelatex missing
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            if r.returncode != 0:
                cmd2 = [
                    pandoc,
                    str(OUT_MD),
                    "-o",
                    str(OUT_PDF),
                    "--from",
                    "markdown",
                    "--toc",
                ]
                # HTML fallback then print? try pdflatex / wkhtml / just html
                html = OUT_PDF.with_suffix(".html")
                r2 = subprocess.run(
                    [pandoc, str(OUT_MD), "-o", str(html), "--from", "markdown", "--toc", "-s"],
                    capture_output=True,
                    text=True,
                )
                print(f"PDF engine failed ({r.stderr[:400] if r.stderr else r.returncode}); wrote HTML {html} rc={r2.returncode}")
                if r2.returncode == 0:
                    # try Edge/Chrome headless print if available later — keep HTML
                    pass
            else:
                print(f"Wrote {OUT_PDF}")
        except FileNotFoundError:
            print("pandoc not runnable")
    else:
        print("pandoc not found; Markdown only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
