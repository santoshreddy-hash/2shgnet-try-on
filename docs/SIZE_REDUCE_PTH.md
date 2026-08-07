# SHGNet-56 size-reduce on **.pth**

Order: keep FP32 → profile → fuse FP32 → **FP16←FP32** → **INT8←FP32** → compare all three.

FP16 and INT8 are **separate** conversions from FP32 (not FP32→FP16→INT8).

Live master `models/shgnet/SHGNet-56_final.pth` is never overwritten.

| Model | .pth MB | Infer ms | Mean Δpx vs FP32 | From |
|-------|---------|----------|------------------|------|
| FP32 | 32.427 | 221.289 | 0.0 | FP32 |
| FP16 | 16.429 | 262.44 | 0.0714 | FP32 |
| INT8 | 6.671 | 271.575 | 75.4639 | FP32 |

- **Desktop:** FP32
- **Browser:** FP16
- **Mobile:** FP16

```bash
python scripts/size_reduce_pth.py --all
```

