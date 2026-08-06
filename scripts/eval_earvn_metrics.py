#!/usr/bin/env python3
"""Compute metrics for EarVN1 batch (no landmark GT → proxy + stability metrics).

EarVN1.0 is an identity dataset — it has no 56-landmark ground truth, so classic
NME / PCK / piercing-px cannot be computed. This script reports:

  A) Inference success & confidence (heatmap peak score)
  B) Geometric validity of the 56-point cloud on the ear crop
  C) Piercing anatomical prior (lobe / lower ear)
  D) Flip-consistency NME (self-agreement under horizontal flip)
  E) Reference GT metrics from training val (ear_pose), for comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import INPUT_SIZE, NUM_LANDMARKS_55, PIERCING_INDEX, resolve_pretrained_56, CKPT_DIR
from train.metrics import nme, pck
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
    flat = arr.reshape(arr.shape[0], -1)
    return float(flat.max(dim=1).values.mean().item())


def to_pts(hm: torch.Tensor) -> np.ndarray:
    pts = heatmaps_to_points(hm, INPUT_SIZE)
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 3:
        pts = pts[0]
    return pts


@torch.inference_mode()
def predict_both(model, ear_bgr: np.ndarray, device: torch.device):
    crop = cv2.resize(ear_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    outs = {}
    for flip in (False, True):
        img = cv2.flip(crop, 1) if flip else crop
        t = preprocess_ear_bgr(img, INPUT_SIZE).to(device)
        hm = model(t)
        if isinstance(hm, (list, tuple)):
            hm = hm[-1]
        score = peak_score(hm)
        pts = to_pts(hm)
        if flip:
            pts = pts.copy()
            pts[:, 0] = float(INPUT_SIZE - 1) - pts[:, 0]
        outs[flip] = (pts, score)
    # best by score
    best_flip = max(outs.keys(), key=lambda f: outs[f][1])
    return crop, outs[False], outs[True], best_flip


def geometric_ok(pts: np.ndarray) -> dict:
    """Crop-relative geometric gates (side_px = 256)."""
    p = pts[:NUM_LANDMARKS_55]
    x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
    x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    span = max(bw, bh)
    side = float(INPUT_SIZE)
    ratio = span / side
    aspect_ok = min(bw, bh) >= span * 0.28
    span_ok = 0.35 <= ratio <= 0.95
    # piercing should sit near lower portion of ear cloud
    pierce = pts[PIERCING_INDEX]
    y_rel = (float(pierce[1]) - y0) / max(bh, 1.0)
    pierce_lower = y_rel >= 0.45
    in_box = (x0 - 0.1 * bw) <= pierce[0] <= (x1 + 0.1 * bw) and (
        y0 - 0.1 * bh
    ) <= pierce[1] <= (y1 + 0.15 * bh)
    inside_frame = (
        (pts[:, 0] >= -8).all()
        and (pts[:, 0] <= INPUT_SIZE + 8).all()
        and (pts[:, 1] >= -8).all()
        and (pts[:, 1] <= INPUT_SIZE + 8).all()
    )
    ok = bool(span_ok and aspect_ok and pierce_lower and in_box and inside_frame)
    return {
        "ok": ok,
        "span_ratio": float(ratio),
        "aspect_ok": bool(aspect_ok),
        "span_ok": bool(span_ok),
        "pierce_lower": bool(pierce_lower),
        "pierce_in_box": bool(in_box),
        "inside_frame": bool(inside_frame),
        "pierce_y_rel": float(y_rel),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-jsonl", default=str(ROOT / "outputs" / "earvn_test_500" / "results.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "earvn_test_500"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows in jsonl")
    args = ap.parse_args()

    results_path = Path(args.results_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[: args.limit]

    paths = [Path(r["path"]) for r in rows if r.get("path")]
    print(f"Evaluating {len(paths)} EarVN images from {results_path}")

    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_pretrained_56()
    if ckpt is None or not Path(ckpt).is_file():
        ckpt = CKPT_DIR / "SHGNet-56_final.pth"
    ckpt = Path(ckpt)
    device = select_device(args.device)
    model = load_shgnet(ckpt, device)

    scores = []
    geo_flags = []
    pierce_lower = []
    flip_nmes = []
    flip_pck02 = []
    flip_pck05 = []
    best_flip_rate = 0
    per_image = []
    fails = 0

    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            fails += 1
            continue
        try:
            crop, noflip, yesflip, best_flip = predict_both(model, img, device)
            pts_nf, sc_nf = noflip
            pts_yf, sc_yf = yesflip
            pts, score = (pts_yf, sc_yf) if best_flip else (pts_nf, sc_nf)
            if best_flip:
                best_flip_rate += 1

            geo = geometric_ok(pts)
            # Flip consistency: compare mirrored predictions in same space
            # pts_yf already unflipped to original crop coords
            fnme = nme(pts_nf, pts_yf)
            fpck = pck(pts_nf, pts_yf)

            scores.append(score)
            geo_flags.append(geo["ok"])
            pierce_lower.append(geo["pierce_lower"])
            flip_nmes.append(fnme)
            flip_pck02.append(fpck["pck@0.02"])
            flip_pck05.append(fpck["pck@0.05"])

            per_image.append(
                {
                    "path": str(path),
                    "score": score,
                    "best_flip": bool(best_flip),
                    "geo_ok": geo["ok"],
                    "pierce_lower": geo["pierce_lower"],
                    "flip_nme": fnme,
                    "flip_pck@0.02": fpck["pck@0.02"],
                    "flip_pck@0.05": fpck["pck@0.05"],
                    "span_ratio": geo["span_ratio"],
                    "pierce_y_rel": geo["pierce_y_rel"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            fails += 1
            per_image.append({"path": str(path), "ok": False, "error": str(exc)})

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(paths)}]")

    n = len(scores)
    scores_a = np.asarray(scores, dtype=np.float64) if scores else np.array([])

    def rate(mask) -> float:
        return float(np.mean(mask)) if len(mask) else float("nan")

    conf_thresholds = [0.3, 0.5, 0.6, 0.7, 0.8]
    conf_acc = {
        f"frac_score_ge_{t}": float((scores_a >= t).mean()) if n else None
        for t in conf_thresholds
    }

    # Proxy "accuracy": geometric validity among predictions
    metrics = {
        "dataset": "EarVN1.0 (no landmark GT)",
        "n_images": len(paths),
        "n_ok": n,
        "n_fail": fails,
        "checkpoint": str(ckpt),
        "device": str(device),
        "note": (
            "EarVN1.0 has no 56-landmark annotations. "
            "True NME/PCK vs GT cannot be computed. "
            "Reported values are confidence, geometric validity, and flip self-consistency."
        ),
        "A_confidence": {
            "mean": float(scores_a.mean()) if n else None,
            "median": float(np.median(scores_a)) if n else None,
            "std": float(scores_a.std()) if n else None,
            "p10": float(np.percentile(scores_a, 10)) if n else None,
            "p90": float(np.percentile(scores_a, 90)) if n else None,
            **conf_acc,
        },
        "B_geometric_validity": {
            "accuracy_geo_ok": rate(geo_flags),
            "pierce_lower_ear_rate": rate(pierce_lower),
            "definition": (
                "geo_ok = landmark span 35–95% of 256 crop, aspect ok, "
                "piercing in lower ear cloud / bbox"
            ),
        },
        "C_flip_consistency": {
            "mean_nme": float(np.mean(flip_nmes)) if flip_nmes else None,
            "median_nme": float(np.median(flip_nmes)) if flip_nmes else None,
            "pck@0.02": float(np.mean(flip_pck02)) if flip_pck02 else None,
            "pck@0.05": float(np.mean(flip_pck05)) if flip_pck05 else None,
            "flip_selected_rate": best_flip_rate / max(n, 1),
            "definition": (
                "Compare no-flip vs flip→unflip predictions on same crop "
                "(self-agreement; not GT accuracy)"
            ),
        },
        "D_proxy_accuracy_summary": {
            "inference_success_rate": n / max(len(paths), 1),
            "high_conf_accuracy_score_ge_0.5": conf_acc.get("frac_score_ge_0.5"),
            "geometric_accuracy": rate(geo_flags),
            "combined_geo_and_score_ge_0.5": float(
                np.mean(
                    [
                        g and s >= 0.5
                        for g, s in zip(geo_flags, scores)
                    ]
                )
            )
            if n
            else None,
        },
    }

    # Reference GT metrics from training (annotated ear_pose)
    train_res = ROOT / "outputs" / "train_results.json"
    if train_res.is_file():
        tr = json.loads(train_res.read_text(encoding="utf-8"))
        best = tr.get("stage3", {}).get("best") or {}
        metrics["E_reference_gt_ear_pose_val"] = {
            "source": "outputs/train_results.json stage3 best (annotated ear_pose)",
            "landmark_nme": best.get("landmark_nme"),
            "piercing_point_error_px": best.get("piercing_point_error_px"),
            "pck@0.02": best.get("pck@0.02"),
            "pck@0.05": best.get("pck@0.05"),
            "pck@0.1": best.get("pck@0.1"),
            "caveat": "GT metrics from training validation, NOT EarVN1.0",
        }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "metrics_per_image.jsonl").open("w", encoding="utf-8") as f:
        for r in per_image:
            f.write(json.dumps(r) + "\n")

    # Human-readable report
    lines = [
        "# EarVN1.0 Test Metrics (500 images)",
        "",
        "> **Important:** EarVN1.0 has **no landmark ground truth**. "
        "NME / PCK / piercing error vs GT are **not available** on this set.",
        "",
        f"- Images evaluated: **{n}/{len(paths)}** (fail={fails})",
        f"- Checkpoint: `{ckpt}`",
        "",
        "## A. Confidence (heatmap peak score)",
        f"- Mean / median: **{metrics['A_confidence']['mean']:.3f}** / **{metrics['A_confidence']['median']:.3f}**",
        f"- P10 / P90: {metrics['A_confidence']['p10']:.3f} / {metrics['A_confidence']['p90']:.3f}",
    ]
    for t in conf_thresholds:
        key = f"frac_score_ge_{t}"
        lines.append(f"- Fraction score >= {t}: **{100 * metrics['A_confidence'][key]:.1f}%**")
    lines += [
        "",
        "## B. Geometric validity (proxy accuracy)",
        f"- **Geometric accuracy:** **{100 * metrics['B_geometric_validity']['accuracy_geo_ok']:.1f}%**",
        f"- Piercing on lower ear: **{100 * metrics['B_geometric_validity']['pierce_lower_ear_rate']:.1f}%**",
        "",
        "## C. Flip self-consistency",
        f"- Mean flip-NME: **{metrics['C_flip_consistency']['mean_nme']:.4f}**",
        f"- Flip PCK@0.05: **{100 * metrics['C_flip_consistency']['pck@0.05']:.1f}%**",
        f"- Flip selected: **{100 * metrics['C_flip_consistency']['flip_selected_rate']:.1f}%**",
        "",
        "## D. Proxy accuracy summary",
        f"- Inference success: **{100 * metrics['D_proxy_accuracy_summary']['inference_success_rate']:.1f}%**",
        f"- High-conf (score>=0.5): **{100 * metrics['D_proxy_accuracy_summary']['high_conf_accuracy_score_ge_0.5']:.1f}%**",
        f"- Geo OK: **{100 * metrics['D_proxy_accuracy_summary']['geometric_accuracy']:.1f}%**",
        f"- Combined (geo OK AND score>=0.5): **{100 * metrics['D_proxy_accuracy_summary']['combined_geo_and_score_ge_0.5']:.1f}%**",
        "",
    ]
    if "E_reference_gt_ear_pose_val" in metrics:
        e = metrics["E_reference_gt_ear_pose_val"]
        lines += [
            "## E. Reference GT metrics (annotated ear_pose val — not EarVN)",
            f"- NME: **{e['landmark_nme']:.4f}**",
            f"- Piercing error: **{e['piercing_point_error_px']:.3f} px**",
            f"- PCK@0.02 / @0.05 / @0.1: "
            f"**{100*e['pck@0.02']:.2f}%** / **{100*e['pck@0.05']:.2f}%** / **{100*e['pck@0.1']:.2f}%**",
            "",
        ]
    report = "\n".join(lines)
    (out_dir / "METRICS.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'METRICS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
