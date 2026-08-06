# Master Copy — SHGNet-56 Ear Pipeline

Lightweight snapshot of the project: **source code + configs + ONNX/PTH models**.

**Excluded:** `.venv`, `node_modules`, `web/vendor`, datasets (images/labels), training outputs, zip packs.

**Included models:**
- `models/shgnet/SHGNet-56.onnx`
- `models/shgnet/SHGNet-56_final.pth`
- `models/yolo/yolo26n-pose.onnx`
- `models/yolo/yolo11n-pose.onnx`

## Setup

```bash
cd "/Users/santoshreddy/Desktop/master copy"

# Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Browser
cd web && npm install && cd ..
```

## Run

```bash
# Desktop live
python -m live.desktop_onnx

# Gradio live
python live/app.py

# Browser
npm start
# -> http://127.0.0.1:8765
```

Created from: `/Users/santoshreddy/Desktop/fps`
