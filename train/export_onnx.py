#!/usr/bin/env python3
"""Export trained SHGNet-56 to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CKPT_DIR, NUM_LANDMARKS_56, ONNX_EXPORT
from train.model import build_ldnet56


def load_model(checkpoint: Path):
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch") or {}
    model = build_ldnet56(
        nstack=int(arch.get("nstack", 2)),
        layer=int(arch.get("layer", 4)),
        in_channel=int(arch.get("in_channel", 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def consolidate_to_single_file(onnx_path: Path) -> Path:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    single = onnx_path.with_name(onnx_path.stem + "_single.onnx")
    onnx.save_model(model, str(single), save_as_external_data=False)
    return single


def main() -> int:
    p = argparse.ArgumentParser(description="Export SHGNet-56.onnx")
    p.add_argument(
        "--checkpoint",
        default=str(CKPT_DIR / "SHGNet-56_final.pth"),
    )
    p.add_argument("--out", default=str(ONNX_EXPORT))
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    if not ckpt_path.is_file():
        # fallback to best stage3 / stage2 / stage1
        for name in ("best_stage3.pth", "best_stage2.pth", "best_stage1.pth"):
            alt = CKPT_DIR / name
            if alt.is_file():
                ckpt_path = alt
                break
        else:
            print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1

    model = load_model(ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["ear_crop"],
        output_names=["heatmaps"],
        opset_version=18,
        do_constant_folding=True,
    )
    try:
        import shutil

        single = consolidate_to_single_file(out_path)
        shutil.copy2(single, out_path)
        data_path = out_path.with_suffix(out_path.suffix + ".data")
        if data_path.is_file():
            data_path.unlink()
        single.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Consolidate warning: {exc}")

    print(f"Exported {out_path.resolve()}")
    print(f"Input  : 1×3×256×256")
    print(f"Output : 1×{NUM_LANDMARKS_56}×64×64  (55 landmarks + piercing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
