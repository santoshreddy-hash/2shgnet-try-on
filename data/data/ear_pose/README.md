# ear_pose dataset

Primary YOLO-pose layout for SHGNet-56 training.

```
ear_pose/
  images/train/   ear crops (.png / .jpg)
  labels/train/   matching .txt (56 keypoints, piercing = #56)
  images/val/     optional
  labels/val/     optional
```

Label line: `class cx cy w h` + `(x y v) * 56` (normalized 0–1).

Populate via the wire script (preferred):

```bash
python scripts/wire_local_dataset.py
# extracts labels.zip + links local images + SHGNet-56_final.pth
```

Or manually: put train/val images here and extract `labels.zip` into
`labels/{train,val}/` (stems must match image names).
