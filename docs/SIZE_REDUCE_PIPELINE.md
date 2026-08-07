# SHGNet-56 size-reduce pipeline

Follows the flowchart: **FP32 master → keep original → profile → bottlenecks → Conv+BN fuse → benchmark → FP16 / INT8 export → compare → deploy pick**.

## Master (step 1) — untouched

| File | Role |
|------|------|
| `models/shgnet/SHGNet-56_final.pth` | Live FP32 master (**not overwritten**) |
| `outputs/size_reduce/SHGNet-56_final_fp32_master.pth` | Immutable backup (~32.7 MB) |

## Profile (step 2)

| Metric | Value |
|--------|-------|
| Params | **8.46 M** |
| FP32 weight size | ~32.3 MB |
| FLOPs | **8.55 GFLOPs** (1×3×256×256) |
| Eager PyTorch CPU | ~327 ms / ~3 FPS (cold-ish bench) |

## Bottlenecks (step 3)

Conv layers dominate (hourglass / residual `conv2`, `pre.*`, `merge_feature`).  
**Action:** fuse `Conv` + following `BatchNorm2d` (65 pairs) before ReLU.

## Fused FP32 (step 4)

| Item | Result |
|------|--------|
| Fused Conv+BN pairs | **65** |
| Heatmap Δ vs unfused | max **1.7e-6** (numerically same) |
| Fused ONNX | **`outputs/size_reduce/SHGNet-56_fused_fp32.onnx` (~25.3 MB)** |

> Note: eager PyTorch timing can look slower after fuse on CPU; deploy via **ONNX Runtime** (fewer BN ops, constant-folded graph).

## Export + bench (steps 5–6)

CPU ONNX Runtime, 20 runs, landmark Δ vs fused FP32 on `outputs/smoke/image_pipeline/04_ear_crop_256.png`:

| Model | ONNX MB | Infer ms | FPS est | Mean Δpx | Pierce Δpx |
|-------|---------|----------|---------|----------|------------|
| **FP32 fused** | 25.3 | 240 | 4.2 | 0.00 | 0.00 |
| **FP16** | **12.7** | 282 | 3.6 | **0.07** | **0.00** |
| **INT8** (dynamic) | **6.8** | 167 | 6.0 | **10.4** | **17.9** |

## Recommendations

| Target | Pick | Why |
|--------|------|-----|
| **Desktop** | FP32 fused | Best accuracy; ORT + AVX |
| **Browser (WASM)** | **FP16** | ~2× smaller than FP32, ~0.07 px drift — INT8 too lossy without calibration |
| **Mobile** | **FP16** (for now) | Same; next: *static* INT8 with ear-crop calibration set |

## Artifacts (local; `*.onnx` / `*.pth` gitignored)

```
outputs/size_reduce/
  SHGNet-56_final_fp32_master.pth
  SHGNet-56_fused_fp32.pth
  SHGNet-56_fused_fp32.onnx
  SHGNet-56_fp16.onnx
  SHGNet-56_int8.onnx
  01_…06 JSON benches
```

Tracked summaries: `docs/size_reduce/*.json`, `docs/size_reduce_compare.json`.

## Reproduce

```bash
python scripts/size_reduce_pipeline.py --all --runs 30
```
