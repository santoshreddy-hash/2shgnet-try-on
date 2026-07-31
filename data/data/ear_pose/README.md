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

Populate from a flat pack by putting train/val images here and extracting
`labels.zip` into `labels/{train,val}/` (stems must match image names).
