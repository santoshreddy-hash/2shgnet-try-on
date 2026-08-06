# EARRING Try On (browser)

Virtual earring try-on using **SHGNet-56** + **YOLO pose** in the browser (`onnxruntime-web` WASM).

- Upload your own earring image (PNG preferred; white backgrounds keyed out)
- Stud locks to piercing landmark **#56** (index 55) — landmarks are hidden in the UI
- Drag the dangling part to swing it; release to settle with improved pendulum physics
- Modes: Classic / Subtle + Freedom slider

## Place ONNX models

From the repo root:

```
models/shgnet/SHGNet-56.onnx
models/yolo/yolo26n-pose.onnx
```

## Run

```powershell
cd web
npm install
npm start
```

Open: **http://127.0.0.1:8765/earring/**

## Browser tips

- Chrome or Edge (desktop), allow camera
- Clear **side profile** of one ear
- Product PNG with transparent (or white) background works best; **top of image = stud**
- Drag the swinging part of the earring on the live canvas

## Studio flow

1. Upload earring image → **TRY ON**
2. **Load models** → **Start camera**
3. Drag dangle to swing; tweak Classic/Subtle + Freedom