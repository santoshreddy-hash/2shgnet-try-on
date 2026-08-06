#!/usr/bin/env python3
"""3-stage fine-tune SHGNet-56 and validate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import (
    BATCH_SIZE,
    CKPT_DIR,
    DATA_IMAGES,
    EAR_POSE_ROOT,
    NUM_WORKERS,
    OUTPUTS,
    SEED,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    STAGE3_EPOCHS,
    STAGE3_LR,
    VAL_SPLIT,
    resolve_pretrained_56,
)
from train.dataset import (
    Piercing56Dataset,
    discover_annotated,
    discover_annotated_all_splits,
    train_val_split,
)
from train.yolo_pose_labels import labels_dir_for_images
from train.metrics import (
    decode_heatmaps,
    heatmap_mse,
    landmark_nme_55,
    nme,
    pck,
    piercing_point_error,
)
from train.model import (
    load_pretrained_expand_to_56,
    trainable_param_count,
    unfreeze_all,
    unfreeze_final_layer,
    unfreeze_last_hourglass,
)
from train.shgnet_base import select_device


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], 0),
        "heatmaps": torch.stack([b["heatmaps"] for b in batch], 0),
        "landmarks": torch.stack([b["landmarks"] for b in batch], 0),
        "name": [b["name"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    losses, nmes, nmes55, pierce_px, pierce_n = [], [], [], [], []
    pck_acc = {}
    for batch in loader:
        imgs = batch["image"].to(device)
        tgt = batch["heatmaps"].to(device)
        gt = batch["landmarks"].numpy()
        pred_hm = model(imgs)
        losses.append(float(heatmap_mse(pred_hm, tgt).item()))
        pred_pts = decode_heatmaps(pred_hm)
        if pred_pts.ndim == 2:
            pred_pts = pred_pts[np.newaxis, ...]
        for i in range(pred_pts.shape[0]):
            nmes.append(nme(pred_pts[i], gt[i]))
            nmes55.append(landmark_nme_55(pred_pts[i], gt[i]))
            px, pn = piercing_point_error(pred_pts[i], gt[i])
            pierce_px.append(px)
            pierce_n.append(pn)
            for k, v in pck(pred_pts[i], gt[i]).items():
                pck_acc.setdefault(k, []).append(v)
    out = {
        "heatmap_loss": float(np.nanmean(losses)),
        "landmark_nme": float(np.nanmean(nmes)),
        "landmark_nme_55": float(np.nanmean(nmes55)),
        "piercing_point_error_px": float(np.nanmean(pierce_px)),
        "piercing_point_error_norm": float(np.nanmean(pierce_n)),
    }
    for k, vals in pck_acc.items():
        out[k] = float(np.nanmean(vals))
    return out


def run_stage(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    stage_name: str,
    ckpt_dir: Path,
) -> dict:
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    best = {"piercing_point_error_px": float("inf")}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f"{stage_name} ep{epoch}/{epochs}", leave=False)
        for batch in pbar:
            imgs = batch["image"].to(device)
            tgt = batch["heatmaps"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(imgs)
            loss = heatmap_mse(pred, tgt)
            loss.backward()
            opt.step()
            running.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(running[-20:]))
        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = float(np.mean(running)) if running else float("nan")
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"[{stage_name}] epoch {epoch}: "
            f"loss={metrics['train_loss']:.5f} "
            f"val_hm={metrics['heatmap_loss']:.5f} "
            f"nme={metrics['landmark_nme']:.4f} "
            f"pierce_px={metrics['piercing_point_error_px']:.2f} "
            f"pck@0.05={metrics.get('pck@0.05', float('nan')):.3f}"
        )
        if metrics["piercing_point_error_px"] <= best["piercing_point_error_px"]:
            best = metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": {
                        "nstack": model.nstack,
                        "layer": model.layer,
                        "in_channel": model.in_channel,
                        "out_channel": model.out_channel,
                    },
                    "stage": stage_name,
                    "metrics": metrics,
                },
                ckpt_dir / f"best_{stage_name}.pth",
            )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": {
                "nstack": model.nstack,
                "layer": model.layer,
                "in_channel": model.in_channel,
                "out_channel": model.out_channel,
            },
            "stage": stage_name,
            "history": history,
        },
        ckpt_dir / f"last_{stage_name}.pth",
    )
    return {"history": history, "best": best}


def load_checkpoint_into_56(path: Path, device):
    from train.model import build_ldnet56

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch") or {}
    model = build_ldnet56(
        nstack=int(arch.get("nstack", 2)),
        layer=int(arch.get("layer", 4)),
        in_channel=int(arch.get("in_channel", 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    meta = {"arch": {
        "nstack": model.nstack,
        "layer": model.layer,
        "in_channel": model.in_channel,
        "out_channel": model.out_channel,
    }, "source_checkpoint": str(path)}
    return model, meta


def main() -> int:
    p = argparse.ArgumentParser(description="Train SHGNet-56 (piercing = landmark #56)")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    p.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    p.add_argument("--stage3-epochs", type=int, default=STAGE3_EPOCHS)
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument(
        "--stage3-only",
        action="store_true",
        help="Skip stages 1–2; load --checkpoint and run stage 3 only",
    )
    p.add_argument(
        "--from-stage2",
        action="store_true",
        default=True,
        help="Default: fine-tune SHGNet-56 (stage 2 + 3). Does not expand from 55.",
    )
    p.add_argument(
        "--full-stages",
        action="store_true",
        help="Load SHGNet-56 and run stage 1 (heads) → 2 → 3 (no 55 expand)",
    )
    p.add_argument(
        "--from-55",
        action="store_true",
        help="Legacy: expand pretrained 55-LM hourglass → 56, then stages 1–3",
    )
    p.add_argument(
        "--variants-per-image",
        type=int,
        default=0,
        help="On-the-fly additive variants per image (45 = online_variants). "
        "0 = legacy random augment_sample only.",
    )
    p.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --variants-per-image: also train the unaugmented original "
        "(default: yes → 1+N samples/image). Use --no-include-original for N only.",
    )
    p.add_argument(
        "--all-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pool images/train+val+test (default). Use --no-all-splits for a "
        "single --images-dir with an internal val split.",
    )
    p.add_argument(
        "--train-on-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --all-splits: train on every image (no holdout). Val metrics "
        "use originals of all images (default).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="SHGNet-56 .pth (default: models/shgnet/SHGNet-56_final.pth "
        "or outputs/checkpoints/SHGNet-56_final.pth)",
    )
    p.add_argument(
        "--images-dir",
        default=None,
        help="Override training image folder (YOLO images/train or iBUG crops with .pts)",
    )
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help="Directory for stage checkpoints / final .pth (default: outputs/checkpoints)",
    )
    p.add_argument(
        "--results-json",
        default=None,
        help="Where to write train_results.json (default: outputs/train_results.json)",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional tag stored in results / final checkpoint metadata",
    )
    args = p.parse_args()

    device = select_device(args.device)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_json = (
        Path(args.results_json)
        if args.results_json
        else (OUTPUTS / "train_results.json")
    )
    cache_dir = OUTPUTS / "lm55_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.images_dir) if args.images_dir else DATA_IMAGES
    # YOLO: labels sibling; .pts: annotations next to images
    try:
        ann_dir = labels_dir_for_images(img_dir)
    except Exception:
        ann_dir = img_dir

    variants_n = max(0, int(args.variants_per_image))
    include_orig = bool(args.include_original)

    train_paths = None
    val_paths = None
    train_names = None
    val_names = None

    if args.all_splits and args.images_dir is None:
        all_paths = discover_annotated_all_splits(EAR_POSE_ROOT)
        if not all_paths:
            print(
                "No annotated images under "
                f"{EAR_POSE_ROOT / 'images'}/{{train,val,test}}",
                file=sys.stderr,
            )
            return 1
        by_split: dict[str, int] = {}
        for p in all_paths:
            by_split[p.parent.name] = by_split.get(p.parent.name, 0) + 1
        print(f"All splits: {dict(sorted(by_split.items()))} total={len(all_paths)}")
        if args.train_on_all:
            train_paths = all_paths
            val_paths = all_paths  # originals-only metrics on full set
            print(
                f"Train on ALL {len(train_paths)} images "
                f"(val metrics = originals of same set)"
            )
        else:
            # Keep official val/test for metrics; train on train only + optional merge
            train_paths = [p for p in all_paths if p.parent.name == "train"]
            val_paths = [p for p in all_paths if p.parent.name in ("val", "test")]
            if not train_paths:
                train_paths = all_paths
            if not val_paths:
                from train.dataset import train_val_split_paths

                train_paths, val_paths = train_val_split_paths(
                    all_paths, VAL_SPLIT, SEED
                )
            print(
                f"Train paths={len(train_paths)} | val/test paths={len(val_paths)}"
            )
    else:
        names = discover_annotated(img_dir, ann_dir)
        if not names:
            print(
                "No annotated images found.\n"
                f"images-dir={img_dir}\n"
                "1) Run: python annotator/app.py\n"
                "2) Upload images, click piercing, Save (writes landmark #56)\n"
                "3) Re-run this trainer.",
                file=sys.stderr,
            )
            return 1
        train_names, val_names = train_val_split(names, VAL_SPLIT, SEED)
        print(
            f"Annotated: {len(names)} | train={len(train_names)} val={len(val_names)}"
        )

    print(f"Images dir: {img_dir if not args.all_splits else EAR_POSE_ROOT / 'images'}")
    print(f"Ckpt dir: {ckpt_dir}")
    print(f"Device: {device}")

    if train_paths is not None:
        train_ds = Piercing56Dataset(
            image_paths=train_paths,
            augment=variants_n <= 0,
            cache_dir=cache_dir,
            fill_55_with_pretrained=False,
            device="cpu",
            variants_per_image=variants_n,
            include_original=include_orig,
        )
        val_ds = Piercing56Dataset(
            image_paths=val_paths,
            augment=False,
            cache_dir=cache_dir,
            fill_55_with_pretrained=False,
            device="cpu",
            variants_per_image=0,
            include_original=True,
        )
    else:
        train_ds = Piercing56Dataset(
            train_names,
            img_dir=img_dir,
            ann_dir=ann_dir,
            augment=variants_n <= 0,
            cache_dir=cache_dir,
            fill_55_with_pretrained=False,
            device="cpu",
            variants_per_image=variants_n,
            include_original=include_orig,
        )
        val_ds = Piercing56Dataset(
            val_names,
            img_dir=img_dir,
            ann_dir=ann_dir,
            augment=False,
            cache_dir=cache_dir,
            fill_55_with_pretrained=False,
            device="cpu",
            variants_per_image=0,
            include_original=True,
        )
    if variants_n > 0:
        print(
            f"Online variants: {variants_n}/image"
            f"{' + original' if include_orig else ''} "
            f"→ {train_ds.samples_per_image} samples/image | "
            f"train_len={len(train_ds)} val_len={len(val_ds)}"
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    results = {}

    use_from_55 = bool(args.from_55)
    use_full_stages = bool(args.full_stages) and not use_from_55 and not args.stage3_only
    use_from_stage2 = (
        bool(args.from_stage2)
        and not use_from_55
        and not args.stage3_only
        and not use_full_stages
    )

    if args.stage3_only and use_from_55:
        print("Use only one of --stage3-only / --from-55", file=sys.stderr)
        return 1
    if args.full_stages and use_from_55:
        print("Use only one of --full-stages / --from-55", file=sys.stderr)
        return 1

    def _resolve_56_ckpt() -> Path | None:
        if args.checkpoint:
            pth = Path(args.checkpoint)
            if pth.is_file() and pth.stat().st_size > 1_000_000:
                return pth
            print(
                f"Checkpoint missing or not a real .pth (stub/tiny?): {pth}",
                file=sys.stderr,
            )
            return None
        return resolve_pretrained_56()

    if args.stage3_only:
        ckpt_path = _resolve_56_ckpt()
        if ckpt_path is None:
            print(
                "Need SHGNet-56 .pth at models/shgnet/SHGNet-56_final.pth "
                "or --checkpoint path",
                file=sys.stderr,
            )
            return 1
        model, meta = load_checkpoint_into_56(ckpt_path, device)
        print(f"Resuming Stage 3 from {ckpt_path}")
        unfreeze_all(model)
        print(f"Stage 3 trainable params: {trainable_param_count(model):,}")
        results["stage3"] = run_stage(
            model, train_loader, val_loader, device,
            args.stage3_epochs, STAGE3_LR, "stage3", ckpt_dir,
        )
    elif use_full_stages:
        ckpt_path = _resolve_56_ckpt()
        if ckpt_path is None:
            print(
                "Need a real SHGNet-56 .pth for --full-stages.\n"
                "Place at models/shgnet/SHGNet-56_final.pth\n"
                "or pass --checkpoint path",
                file=sys.stderr,
            )
            return 1
        model, meta = load_checkpoint_into_56(ckpt_path, device)
        print(f"Full stages 1→2→3 from SHGNet-56 {ckpt_path}")
        unfreeze_final_layer(model)
        print(f"Stage 1 trainable params: {trainable_param_count(model):,}")
        results["stage1"] = run_stage(
            model, train_loader, val_loader, device,
            args.stage1_epochs, STAGE1_LR, "stage1", ckpt_dir,
        )
        unfreeze_last_hourglass(model)
        print(f"Stage 2 trainable params: {trainable_param_count(model):,}")
        results["stage2"] = run_stage(
            model, train_loader, val_loader, device,
            args.stage2_epochs, STAGE2_LR, "stage2", ckpt_dir,
        )
        if not args.skip_stage3:
            unfreeze_all(model)
            print(f"Stage 3 trainable params: {trainable_param_count(model):,}")
            results["stage3"] = run_stage(
                model, train_loader, val_loader, device,
                args.stage3_epochs, STAGE3_LR, "stage3", ckpt_dir,
            )
    elif use_from_stage2:
        ckpt_path = _resolve_56_ckpt()
        if ckpt_path is None:
            print(
                "Need a real SHGNet-56 .pth to fine-tune.\n"
                "Place at models/shgnet/SHGNet-56_final.pth\n"
                "or pass --checkpoint path/to/SHGNet-56_final.pth\n"
                "Legacy only: --from-55",
                file=sys.stderr,
            )
            return 1
        model, meta = load_checkpoint_into_56(ckpt_path, device)
        print(f"Fine-tuning SHGNet-56 from {ckpt_path} (stage 2 + 3)")
        unfreeze_last_hourglass(model)
        print(f"Stage 2 trainable params: {trainable_param_count(model):,}")
        results["stage2"] = run_stage(
            model, train_loader, val_loader, device,
            args.stage2_epochs, STAGE2_LR, "stage2", ckpt_dir,
        )
        if not args.skip_stage3:
            unfreeze_all(model)
            print(f"Stage 3 trainable params: {trainable_param_count(model):,}")
            results["stage3"] = run_stage(
                model, train_loader, val_loader, device,
                args.stage3_epochs, STAGE3_LR, "stage3", ckpt_dir,
            )
    else:
        model, meta = load_pretrained_expand_to_56(device=device)
        print(f"Expanded 55→56 from {meta['source_checkpoint']}")

        unfreeze_final_layer(model)
        print(f"Stage 1 trainable params: {trainable_param_count(model):,}")
        results["stage1"] = run_stage(
            model, train_loader, val_loader, device,
            args.stage1_epochs, STAGE1_LR, "stage1", ckpt_dir,
        )

        unfreeze_last_hourglass(model)
        print(f"Stage 2 trainable params: {trainable_param_count(model):,}")
        results["stage2"] = run_stage(
            model, train_loader, val_loader, device,
            args.stage2_epochs, STAGE2_LR, "stage2", ckpt_dir,
        )

        if not args.skip_stage3:
            unfreeze_all(model)
            print(f"Stage 3 trainable params: {trainable_param_count(model):,}")
            results["stage3"] = run_stage(
                model, train_loader, val_loader, device,
                args.stage3_epochs, STAGE3_LR, "stage3", ckpt_dir,
            )

    # Prefer best weights from stages completed in *this* run
    if "stage3" in results and (ckpt_dir / "best_stage3.pth").is_file():
        best3 = ckpt_dir / "best_stage3.pth"
        model, meta = load_checkpoint_into_56(best3, device)
        print(f"Final export weights from {best3}")
    elif "stage2" in results and (ckpt_dir / "best_stage2.pth").is_file():
        best2 = ckpt_dir / "best_stage2.pth"
        model, meta = load_checkpoint_into_56(best2, device)
        print(f"Final export weights from {best2}")
    elif "stage1" in results and (ckpt_dir / "best_stage1.pth").is_file():
        best1 = ckpt_dir / "best_stage1.pth"
        model, meta = load_checkpoint_into_56(best1, device)
        print(f"Final export weights from {best1}")

    final_path = ckpt_dir / "SHGNet-56_final.pth"
    n_imgs = len(train_ds.paths)
    payload = {
        "model_state_dict": model.state_dict(),
        "arch": meta["arch"],
        "results": results,
        "images_dir": str(
            EAR_POSE_ROOT / "images" if args.all_splits and args.images_dir is None else img_dir
        ),
        "n_annotated": n_imgs,
        "samples_per_image": train_ds.samples_per_image,
        "variants_per_image": variants_n,
        "include_original": include_orig,
        "all_splits": bool(args.all_splits and args.images_dir is None),
        "train_on_all": bool(args.train_on_all),
    }
    if args.run_name:
        payload["run_name"] = args.run_name
    torch.save(payload, final_path)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {final_path}")
    print(f"Saved {results_json}")
    print("Next: python -m train.test_model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
