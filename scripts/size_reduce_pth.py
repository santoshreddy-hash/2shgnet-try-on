#!/usr/bin/env python3
"""
Size-reduce on SHGNet-56 **.pth** — FP16 and INT8 each come **from FP32**, not chained.

Order:
  1) Keep original FP32 master (.pth backup; never overwrite live)
  2) Profile FP32
  3) Bottlenecks → Conv+BN fuse → fused FP32 .pth + bench
  4) Export FP16 .pth  ← from FP32 (separate)
  5) Export INT8 .pth  ← from FP32 (separate)
  6) Benchmark + compare all three (FP32 / FP16 / INT8)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import INPUT_SIZE, PRETRAINED_56
from train.export_onnx import load_model
from train.model import build_ldnet56
from train.shgnet_base import Conv

OUT = ROOT / "outputs" / "size_reduce" / "pth"
MASTER = "SHGNet-56_final_fp32_master.pth"
FUSED = "SHGNet-56_fused_fp32.pth"
FP16 = "SHGNet-56_fp16.pth"  # from FP32
INT8 = "SHGNet-56_int8.pth"  # from FP32


def _mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024) if p.is_file() else 0.0


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _arch(ckpt: Path | dict) -> dict:
    if isinstance(ckpt, dict):
        return dict(ckpt.get("arch") or {})
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    return dict(blob.get("arch") or {})


def _build(arch: dict) -> nn.Module:
    return build_ldnet56(
        nstack=int(arch.get("nstack", 2)),
        layer=int(arch.get("layer", 4)),
        in_channel=int(arch.get("in_channel", 256)),
    )


def _state_mb(sd: dict) -> float:
    n = 0
    for v in sd.values():
        if torch.is_tensor(v):
            n += v.numel() * v.element_size()
    return n / (1024 * 1024)


def fuse_conv_bn(model: nn.Module) -> int:
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    n = 0
    for mod in model.modules():
        if isinstance(mod, Conv) and mod.bn is not None and isinstance(mod.conv, nn.Conv2d):
            mod.conv = fuse_conv_bn_eval(mod.conv, mod.bn)
            mod.bn = None
            n += 1
    model.eval()
    return n


def _bench(model: nn.Module, x: torch.Tensor, runs: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ms))


def _peaks(hm: torch.Tensor) -> torch.Tensor:
    _, c, h, w = hm.shape
    flat = hm.reshape(c, -1)
    idx = flat.argmax(dim=1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    scale = INPUT_SIZE / float(w)
    return torch.stack([xs * scale, ys * scale], dim=1)


def _acc(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    d = (_peaks(ref) - _peaks(cand)).norm(dim=1)
    return {
        "mean_px": round(float(d.mean()), 4),
        "max_px": round(float(d.max()), 4),
        "pierce_px": round(float(d[min(55, d.numel() - 1)]), 4),
        "heatmap_max_abs": round(float((ref.float() - cand.float()).abs().max()), 6),
    }


def _load_crop() -> torch.Tensor:
    crop = ROOT / "outputs" / "smoke" / "image_pipeline" / "04_ear_crop_256.png"
    if crop.is_file():
        import cv2

        img = cv2.imread(str(crop))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)


def _fp32_source_path() -> Path:
    """Prefer fused FP32 if present (same accuracy, flowchart order); else master."""
    fused = OUT / FUSED
    if fused.is_file():
        return fused
    return OUT / MASTER


def _load_fp32_model(path: Path) -> tuple[nn.Module, dict]:
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = blob.get("arch") or {}
    model = _build(arch)
    model.load_state_dict(blob["model_state_dict"], strict=False)
    model.eval()
    return model, blob


# ---------------------------------------------------------------------------
# 1) Keep master
# ---------------------------------------------------------------------------
def step1_keep(src: Path) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / MASTER
    shutil.copy2(src, dst)
    info = {
        "step": 1,
        "live_pth": str(src.relative_to(ROOT)),
        "backup_pth": str(dst.relative_to(ROOT)),
        "backup_mb": round(_mb(dst), 3),
        "live_untouched": True,
    }
    _save_json(OUT / "01_keep_master.json", info)
    print(f"[1] Keep FP32 master → {dst.name} ({info['backup_mb']:.2f} MB)")
    return info


# ---------------------------------------------------------------------------
# 2) Profile
# ---------------------------------------------------------------------------
def step2_profile(src: Path, runs: int) -> dict:
    model = load_model(src)
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    try:
        from thop import profile as thop_profile

        flops, params = thop_profile(model, inputs=(x,), verbose=False)
        flops_g, params_m = float(flops) / 1e9, float(params) / 1e6
    except Exception:
        flops_g = None
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
    ms = _bench(model, x, runs)
    info = {
        "step": 2,
        "params_m": round(params_m, 3),
        "state_dict_mb": round(_state_mb(model.state_dict()), 3),
        "file_mb": round(_mb(src), 3),
        "flops_g": round(flops_g, 3) if flops_g is not None else None,
        "infer_ms": round(ms, 3),
        "dtype": "float32",
    }
    _save_json(OUT / "02_profile.json", info)
    print(f"[2] Profile FP32: {info['params_m']}M · {info['flops_g']} GFLOPs · {info['infer_ms']} ms")
    return info


# ---------------------------------------------------------------------------
# 3) Fuse Conv+BN → fused FP32 .pth
# ---------------------------------------------------------------------------
def step3_fuse(src: Path, runs: int) -> dict:
    arch = _arch(src)
    base = load_model(src)
    fused = copy.deepcopy(base)
    n = fuse_conv_bn(fused)
    x = _load_crop()
    with torch.no_grad():
        y0, y1 = base(x), fused(x)
    acc = _acc(y0, y1)
    ms0, ms1 = _bench(base, x, runs), _bench(fused, x, runs)
    path = OUT / FUSED
    torch.save(
        {
            "model_state_dict": fused.state_dict(),
            "arch": arch,
            "dtype": "float32",
            "fused_conv_bn": True,
            "n_fused_pairs": n,
            "source": str(src),
        },
        path,
    )
    info = {
        "step": 3,
        "n_fused_pairs": n,
        "pth": str(path.relative_to(ROOT)),
        "file_mb": round(_mb(path), 3),
        "state_dict_mb": round(_state_mb(fused.state_dict()), 3),
        "infer_ms_fp32": round(ms0, 3),
        "infer_ms_fused": round(ms1, 3),
        "accuracy_vs_master": acc,
    }
    _save_json(OUT / "03_fuse_fp32.json", info)
    print(f"[3] Fused FP32 .pth ({n} pairs) → {path.name} ({info['file_mb']:.2f} MB) Δ={acc['mean_px']} px")
    return info


# ---------------------------------------------------------------------------
# 4) FP16 .pth  ← FROM FP32 (not from INT8)
# ---------------------------------------------------------------------------
def step4_fp16_from_fp32(runs: int) -> dict:
    src = _fp32_source_path()
    model, blob = _load_fp32_model(src)
    arch = blob.get("arch") or {}

    sd16 = {
        k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    path = OUT / FP16
    torch.save(
        {
            "model_state_dict": sd16,
            "arch": arch,
            "dtype": "float16",
            "source_fp32": str(src),
            "derived_from": "FP32",  # explicit: not from INT8
            "fused_conv_bn": bool(blob.get("fused_conv_bn")),
        },
        path,
    )

    # Bench: upcast FP16 weights → float32 runtime (CPU-safe) vs FP32 source
    ref, _ = _load_fp32_model(src)
    half_model = _build(arch)
    half_model.load_state_dict(
        {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd16.items()},
        strict=False,
    )
    half_model.eval()
    x = _load_crop()
    with torch.no_grad():
        acc = _acc(ref(x), half_model(x))
    ms = _bench(half_model, x, runs)
    info = {
        "step": 4,
        "path_name": "FP16_from_FP32",
        "source_fp32": str(src.relative_to(ROOT)),
        "pth": str(path.relative_to(ROOT)),
        "file_mb": round(_mb(path), 3),
        "state_dict_mb": round(_state_mb(sd16), 3),
        "infer_ms": round(ms, 3),
        "accuracy_vs_fp32": acc,
    }
    _save_json(OUT / "04_fp16_from_fp32.json", info)
    print(f"[4] FP16 ← FP32 → {path.name} ({info['file_mb']:.2f} MB) Δ={acc['mean_px']} px")
    return info


# ---------------------------------------------------------------------------
# 5) INT8 .pth  ← FROM FP32 (not from FP16)
# ---------------------------------------------------------------------------
def step5_int8_from_fp32(runs: int) -> dict:
    src = _fp32_source_path()
    model, blob = _load_fp32_model(src)
    arch = blob.get("arch") or {}
    x = _load_crop()
    with torch.no_grad():
        y_ref = model(x)

    path = OUT / INT8
    method = "int8_packed_from_fp32"
    err = None
    acc = None
    ms = None

    # Try FX static PTQ from FP32; fallback to affine int8 pack from FP32 weights
    try:
        backend = "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines else "qnnpack"
        torch.backends.quantized.engine = backend
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        prepared = prepare_fx(copy.deepcopy(model), get_default_qconfig_mapping(backend), example)
        with torch.no_grad():
            for _ in range(8):
                prepared(x)
                prepared(torch.randn_like(x))
        qmodel = convert_fx(prepared)
        qmodel.eval()
        with torch.no_grad():
            y_q = qmodel(x)
            y_cmp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q.float()
        acc = _acc(y_ref, y_cmp)
        ms = _bench(qmodel, x, max(5, runs // 2))
        torch.save(
            {
                "quantized_state_dict": qmodel.state_dict(),
                "arch": arch,
                "dtype": "int8_fx",
                "backend": backend,
                "source_fp32": str(src),
                "derived_from": "FP32",  # explicit: not from FP16
            },
            path,
        )
        method = f"fx_static_{backend}_from_fp32"
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        print(f"[5] FX PTQ from FP32 failed ({exc}); packing INT8 weights from FP32")
        sd8, meta = {}, {}
        for k, v in model.state_dict().items():
            if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 0:
                f = v.float()
                scale = float(f.abs().max().clamp(min=1e-8)) / 127.0
                sd8[k] = torch.clamp(torch.round(f / scale), -128, 127).to(torch.int8)
                meta[k] = {"scale": scale, "zp": 0}
            else:
                sd8[k] = v
        torch.save(
            {
                "model_state_dict_int8": sd8,
                "dequant_meta": meta,
                "arch": arch,
                "dtype": "int8_packed",
                "source_fp32": str(src),
                "derived_from": "FP32",
                "ptq_error": err,
            },
            path,
        )
        # Round-trip dequant bench
        sd_f = {
            k: (v.float() * meta[k]["scale"] if k in meta and torch.is_tensor(v) and v.dtype == torch.int8 else v)
            for k, v in sd8.items()
        }
        m2 = _build(arch)
        m2.load_state_dict(sd_f, strict=False)
        m2.eval()
        with torch.no_grad():
            acc = _acc(y_ref, m2(x))
        ms = _bench(m2, x, runs)
        method = "int8_packed_from_fp32"

    info = {
        "step": 5,
        "path_name": "INT8_from_FP32",
        "method": method,
        "source_fp32": str(src.relative_to(ROOT)),
        "pth": str(path.relative_to(ROOT)),
        "file_mb": round(_mb(path), 3),
        "infer_ms": round(ms, 3) if ms is not None else None,
        "accuracy_vs_fp32": acc,
        "ptq_error": err,
    }
    _save_json(OUT / "05_int8_from_fp32.json", info)
    print(
        f"[5] INT8 ← FP32 ({method}) → {path.name} ({info['file_mb']:.2f} MB) "
        f"Δ={(acc or {}).get('mean_px')}"
    )
    return info


# ---------------------------------------------------------------------------
# 6) Benchmark + compare all three
# ---------------------------------------------------------------------------
def step6_compare_all_three(runs: int) -> dict:
    fp32_path = _fp32_source_path()
    fp16_path = OUT / FP16
    int8_path = OUT / INT8
    x = _load_crop()

    # FP32
    m32, blob32 = _load_fp32_model(fp32_path)
    with torch.no_grad():
        y32 = m32(x)
    ms32 = _bench(m32, x, runs)

    # FP16 (dequant weights to float for CPU)
    blob16 = torch.load(str(fp16_path), map_location="cpu", weights_only=False)
    arch = blob16.get("arch") or {}
    m16 = _build(arch)
    sd16 = blob16["model_state_dict"]
    m16.load_state_dict(
        {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd16.items()},
        strict=False,
    )
    m16.eval()
    with torch.no_grad():
        y16 = m16(x)
    ms16 = _bench(m16, x, runs)
    acc16 = _acc(y32, y16)

    # INT8
    blob8 = torch.load(str(int8_path), map_location="cpu", weights_only=False)
    if "model_state_dict_int8" in blob8:
        meta = blob8["dequant_meta"]
        sd8 = blob8["model_state_dict_int8"]
        sd_f = {
            k: (v.float() * meta[k]["scale"] if k in meta and torch.is_tensor(v) and v.dtype == torch.int8 else v)
            for k, v in sd8.items()
        }
        m8 = _build(arch)
        m8.load_state_dict(sd_f, strict=False)
        m8.eval()
        with torch.no_grad():
            y8 = m8(x)
        ms8 = _bench(m8, x, runs)
        acc8 = _acc(y32, y8)
        int8_method = blob8.get("dtype", "int8_packed")
    else:
        # FX quantized — reload via saved accuracy json if needed
        acc8 = json.loads((OUT / "05_int8_from_fp32.json").read_text()).get("accuracy_vs_fp32")
        ms8 = json.loads((OUT / "05_int8_from_fp32.json").read_text()).get("infer_ms")
        int8_method = blob8.get("dtype", "int8_fx")
        y8 = None  # noqa

    rows = [
        {
            "model": "FP32",
            "source": "master/fused",
            "pth_mb": round(_mb(fp32_path), 3),
            "infer_ms": round(ms32, 3),
            "mean_px_vs_fp32": 0.0,
            "pierce_px_vs_fp32": 0.0,
            "derived_from": "FP32",
            "path": str(fp32_path.relative_to(ROOT)),
        },
        {
            "model": "FP16",
            "source": "from_FP32",
            "pth_mb": round(_mb(fp16_path), 3),
            "infer_ms": round(ms16, 3),
            "mean_px_vs_fp32": acc16["mean_px"],
            "pierce_px_vs_fp32": acc16["pierce_px"],
            "derived_from": "FP32",
            "path": str(fp16_path.relative_to(ROOT)),
        },
        {
            "model": "INT8",
            "source": "from_FP32",
            "pth_mb": round(_mb(int8_path), 3),
            "infer_ms": round(ms8, 3) if ms8 is not None else None,
            "mean_px_vs_fp32": (acc8 or {}).get("mean_px"),
            "pierce_px_vs_fp32": (acc8 or {}).get("pierce_px"),
            "derived_from": "FP32",
            "method": int8_method,
            "path": str(int8_path.relative_to(ROOT)),
        },
    ]

    # Picks
    fp16_ok = (rows[1]["mean_px_vs_fp32"] or 99) <= 2.0
    int8_ok = (rows[2]["mean_px_vs_fp32"] or 99) <= 2.0
    browser = "FP16" if fp16_ok else "FP32"
    if int8_ok and rows[2]["pth_mb"] < rows[1]["pth_mb"]:
        browser = "INT8"

    info = {
        "step": 6,
        "note": "FP16 and INT8 each converted separately from FP32 (no FP32→FP16→INT8 chain)",
        "comparison": rows,
        "recommendations": {
            "Desktop": "FP32",
            "Browser": browser,
            "Mobile": "FP16" if fp16_ok else "FP32",
        },
    }
    _save_json(OUT / "06_compare_fp32_fp16_int8.json", info)
    print("[6] Benchmark all three (.pth)")
    for r in rows:
        print(
            f"    {r['model']:5s}  {r['pth_mb']:7.2f} MB  {r.get('infer_ms')} ms  "
            f"Δmean={r.get('mean_px_vs_fp32')} px  ← {r['derived_from']}"
        )
    print(f"    → Desktop=FP32  Browser={browser}")
    return info


def write_docs(compare: dict) -> None:
    lines = [
        "# SHGNet-56 size-reduce on **.pth**",
        "",
        "Order: keep FP32 → profile → fuse FP32 → **FP16←FP32** → **INT8←FP32** → compare all three.",
        "",
        "FP16 and INT8 are **separate** conversions from FP32 (not FP32→FP16→INT8).",
        "",
        "Live master `models/shgnet/SHGNet-56_final.pth` is never overwritten.",
        "",
        "| Model | .pth MB | Infer ms | Mean Δpx vs FP32 | From |",
        "|-------|---------|----------|------------------|------|",
    ]
    for r in compare.get("comparison") or []:
        lines.append(
            f"| {r['model']} | {r.get('pth_mb')} | {r.get('infer_ms')} | "
            f"{r.get('mean_px_vs_fp32')} | {r.get('derived_from')} |"
        )
    rec = compare.get("recommendations") or {}
    lines += [
        "",
        f"- **Desktop:** {rec.get('Desktop')}",
        f"- **Browser:** {rec.get('Browser')}",
        f"- **Mobile:** {rec.get('Mobile')}",
        "",
        "```bash",
        "python scripts/size_reduce_pth.py --all",
        "```",
        "",
    ]
    (ROOT / "docs" / "SIZE_REDUCE_PTH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "size_reduce_pth_compare.json").write_text(
        json.dumps(compare, indent=2) + "\n", encoding="utf-8"
    )
    tracked = ROOT / "docs" / "size_reduce" / "pth"
    tracked.mkdir(parents=True, exist_ok=True)
    for name in (
        "01_keep_master.json",
        "02_profile.json",
        "03_fuse_fp32.json",
        "04_fp16_from_fp32.json",
        "05_int8_from_fp32.json",
        "06_compare_fp32_fp16_int8.json",
    ):
        src = OUT / name
        if src.is_file():
            shutil.copy2(src, tracked / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PRETRAINED_56))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"Missing {ckpt}", file=sys.stderr)
        return 1

    do_all = args.all or args.step == 0
    print(
        "Order:\n"
        "  1 Keep FP32 .pth\n"
        "  2 Profile FP32\n"
        "  3 Fuse Conv+BN → fused FP32 .pth\n"
        "  4 FP16 .pth ← FP32 (separate)\n"
        "  5 INT8 .pth ← FP32 (separate)\n"
        "  6 Benchmark FP32 / FP16 / INT8\n"
    )

    if do_all or args.step == 1:
        step1_keep(ckpt)
    if do_all or args.step == 2:
        step2_profile(ckpt, args.runs)
    if do_all or args.step == 3:
        step3_fuse(ckpt, args.runs)
    if do_all or args.step == 4:
        step4_fp16_from_fp32(args.runs)
    if do_all or args.step == 5:
        step5_int8_from_fp32(args.runs)
    if do_all or args.step == 6:
        compare = step6_compare_all_three(args.runs)
        write_docs(compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
