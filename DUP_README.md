# dup fol try on

Duplicate of `2shgnet-try-on` with required runtime/training files.

## Quick start
```powershell
cd "D:\try on proj\dup fol try on"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install onnxruntime ultralytics

# Desktop live
python -m live.desktop_onnx --camera 0

# Browser demo
cd web
npm install
npm start
# -> http://127.0.0.1:8765
```

## Key model files
- `models/shgnet/SHGNet-56_final.pth`
- `models/shgnet/SHGNet-56.onnx`
- `models/yolo/yolo26n-pose.onnx`
- `one_euro_settings.json`

Copied from: `2shgnet-try-on` on 2026-08-03 18:31
