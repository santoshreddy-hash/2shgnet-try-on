#!/usr/bin/env python3
"""Batch-test SHGNet-56 on EarVN1.0 ear crops (~500 images).

EarVN images are already tight ear crops (no full-face YOLO needed).
Each image → resize 256 → SHGNet (flip + no-flip, keep higher score) → overlay.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CKPT_DIR, INPUT_SIZE, PIERCING_INDEX, resolve_pretrained_56
from train.model import build_ldnet56
from train.shgnet_base import heatmaps_to_points, preprocess_ear_bgr, select_device


def load_shgnet(ckpt: Path, device: torch.device):
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    arch = blob.get("arch") or {}
    model = build_ldnet56(
        nstack=int(arch.get("nstack", 2)),
        layer=int(arch.get("layer", 4)),
        in_channel=int(arch.get("in_channel", 256)),
    )
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def peak_score(hm: torch.Tensor) -> float:
    arr = hm.detach().float()
    if arr.ndim == 4:
        arr = arr[0]
    # last stack heatmaps if list-like not needed — model returns tensor
    flat = arr.reshape(arr.shape[0], -1)
    return float(flat.max(dim=1).values.mean().item())


@torch.inference_mode()
def infer_best(model, ear_bgr: np.ndarray, device: torch.device):
    """Run SHGNet on crop; try flip; return pts256, score, flipped, crop256 used."""
    crop = cv2.resize(ear_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    best = None
    for flip in (False, True):
        img = cv2.flip(crop, 1) if flip else crop
        t = preprocess_ear_bgr(img, INPUT_SIZE).to(device)
        hm = model(t)
        if isinstance(hm, (list, tuple)):
            hm = hm[-1]
        score = peak_score(hm)
        pts = heatmaps_to_points(hm, INPUT_SIZE)
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts[0]
        if flip:
            pts[:, 0] = float(INPUT_SIZE - 1) - pts[:, 0]
        if best is None or score > best[1]:
            best = (pts, score, flip, crop)
    assert best is not None
    return best


def draw_overlay(crop256: np.ndarray, pts: np.ndarray, score: float, label: str) -> np.ndarray:
    vis = crop256.copy()
    for i in range(55):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        if 0 <= x < 256 and 0 <= y < 256:
            cv2.circle(vis, (x, y), 2, (0, 220, 255), -1)
    x, y = int(round(pts[PIERCING_INDEX, 0])), int(round(pts[PIERCING_INDEX, 1]))
    cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.circle(vis, (x, y), 7, (0, 0, 255), 2)
    cv2.putText(
        vis,
        f"{label} s={score:.2f}",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def make_contact_sheet(
    tiles: list[np.ndarray], cols: int = 10, cell: int = 128
) -> np.ndarray:
    if not tiles:
        return np.zeros((cell, cell, 3), np.uint8)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 24, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        thumb = cv2.resize(t, (cell, cell), interpolation=cv2.INTER_AREA)
        sheet[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = thumb
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch test SHGNet-56 on EarVN1.0")
    ap.add_argument(
        "--images-dir",
        default=r"D:\try on proj\EarVN1.0 dataset\EarVN1.0 dataset\Images",
    )
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "earvn_test_500"),
    )
    ap.add_argument("--save-every", type=int, default=1, help="save overlay every Nth")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Missing images dir: {images_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {images_dir} …")
    all_imgs = sorted(images_dir.rglob("*.jpg")) + sorted(images_dir.rglob("*.png"))
    print(f"Found {len(all_imgs)} images")
    if not all_imgs:
        return 1

    rng = random.Random(args.seed)
    n = min(args.n, len(all_imgs))
    picked = rng.sample(all_imgs, n)

    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_pretrained_56()
    if ckpt is None or not Path(ckpt).is_file():
        ckpt = CKPT_DIR / "SHGNet-56_final.pth"
    ckpt = Path(ckpt)
    device = select_device(args.device)
    print(f"Checkpoint: {ckpt}")
    print(f"Device: {device} · testing {n} images (seed={args.seed})")

    model = load_shgnet(ckpt, device)
    out = Path(args.out_dir)
    overlays_dir = out / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tiles: list[np.ndarray] = []
    scores: list[float] = []
    fails = 0
    t0 = time.perf_counter()

    for i, path in enumerate(picked):
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            fails += 1
            rows.append({"path": str(path), "ok": False, "error": "read_fail"})
            continue
        try:
            pts, score, flipped, crop256 = infer_best(model, img, device)
            pierce = pts[PIERCING_INDEX]
            rec = {
                "path": str(path),
                "ok": True,
                "identity": path.parent.name,
                "shape": list(img.shape),
                "score": float(score),
                "flipped": bool(flipped),
                "piercing_256": [float(pierce[0]), float(pierce[1])],
            }
            rows.append(rec)
            scores.append(float(score))
            label = path.parent.name[:12]
            vis = draw_overlay(crop256, pts, score, label)
            tiles.append(vis)
            if i % max(1, args.save_every) == 0:
                # keep relative-ish name
                safe = f"{i:04d}_{path.parent.name}_{path.stem}".replace(" ", "_")
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe)[
                    :120
                ]
                cv2.imwrite(str(overlays_dir / f"{safe}.jpg"), vis)
        except Exception as exc:  # noqa: BLE001
            fails += 1
            rows.append({"path": str(path), "ok": False, "error": str(exc)})

        if (i + 1) % 50 == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"  [{i + 1}/{n}] ok={len(scores)} fail={fails} "
                f"mean_score={np.mean(scores) if scores else 0:.3f} "
                f"{rate:.1f} img/s"
            )

    elapsed = time.perf_counter() - t0
    summary = {
        "images_dir": str(images_dir),
        "n_requested": args.n,
        "n_tested": n,
        "n_ok": len(scores),
        "n_fail": fails,
        "seed": args.seed,
        "checkpoint": str(ckpt),
        "device": str(device),
        "elapsed_sec": round(elapsed, 2),
        "imgs_per_sec": round(n / max(elapsed, 1e-6), 2),
        "score_mean": float(np.mean(scores)) if scores else None,
        "score_median": float(np.median(scores)) if scores else None,
        "score_p10": float(np.percentile(scores, 10)) if scores else None,
        "score_p90": float(np.percentile(scores, 90)) if scores else None,
        "flip_rate": float(np.mean([r["flipped"] for r in rows if r.get("ok")]))
        if any(r.get("ok") for r in rows)
        else None,
    }

    # Contact sheets (chunks of 100)
    sheets_dir = out / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    chunk = 100
    for s in range(0, len(tiles), chunk):
        sheet = make_contact_sheet(tiles[s : s + chunk], cols=10, cell=128)
        cv2.imwrite(str(sheets_dir / f"sheet_{s // chunk:02d}.jpg"), sheet)

    # Preview montage of first 40
    preview = make_contact_sheet(tiles[:40], cols=8, cell=160)
    cv2.imwrite(str(out / "00_preview_40.jpg"), preview)

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n=== EarVN1 batch summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOutputs → {out}")
    print(f"  preview: {out / '00_preview_40.jpg'}")
    print(f"  sheets:  {sheets_dir}")
    print(f"  overlays:{overlays_dir} ({len(list(overlays_dir.glob('*.jpg')))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
