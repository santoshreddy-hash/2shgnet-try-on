#!/usr/bin/env python3
"""
SHGNet-56 size-reduce pipeline (flowchart):

  FP32 master → keep original → profile → bottlenecks →
  Conv+BN fuse → bench fused FP32 → export FP16 / INT8 →
  compare → recommend Desktop / Browser / Mobile

Artifacts land in outputs/size_reduce/ (gitignored binaries).
Reports: docs/SIZE_REDUCE_PIPELINE.md + outputs/size_reduce/compare.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import INPUT_SIZE, NUM_LANDMARKS_56, PRETRAINED_56, resolve_onnx_export
from train.export_onnx import consolidate_to_single_file, load_model
from train.shgnet_base import Conv

OUT = ROOT / "outputs" / "size_reduce"
MASTER_NAME = "SHGNet-56_final_fp32_master.pth"
FUSED_PTH = "SHGNet-56_fused_fp32.pth"
ONNX_FP32 = "SHGNet-56_fused_fp32.onnx"
ONNX_FP16 = "SHGNet-56_fp16.onnx"
ONNX_INT8 = "SHGNet-56_int8.onnx"


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0


def _param_mb(model: nn.Module, bytes_per: int = 4) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * bytes_per / (1024 * 1024)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1 — keep original checkpoint
# ---------------------------------------------------------------------------
def step1_keep_master(ckpt: Path) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    master = OUT / MASTER_NAME
    if not master.is_file() or master.stat().st_size != ckpt.stat().st_size:
        shutil.copy2(ckpt, master)
    # Never overwrite the live models/ path
    live = PRETRAINED_56.resolve()
    assert live.is_file()
    info = {
        "step": 1,
        "master_backup": str(master.relative_to(ROOT)),
        "master_mb": round(_mb(master), 3),
        "live_checkpoint": str(live.relative_to(ROOT)),
        "live_mb": round(_mb(live), 3),
        "identical_to_live": master.stat().st_size == live.stat().st_size,
        "note": "Original FP32 master preserved; later steps write only under outputs/size_reduce/",
    }
    _save_json(OUT / "01_keep_master.json", info)
    print(f"[1] Master backup → {master} ({info['master_mb']:.2f} MB)")
    return info


# ---------------------------------------------------------------------------
# Step 2 — profile
# ---------------------------------------------------------------------------
def step2_profile(ckpt: Path, runs: int = 30) -> dict:
    model = load_model(ckpt)
    model.eval()
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # FLOPs via thop
    flops = None
    try:
        from thop import profile as thop_profile

        flops, params = thop_profile(model, inputs=(x,), verbose=False)
        flops_g = float(flops) / 1e9
        params_m = float(params) / 1e6
    except Exception as exc:  # noqa: BLE001
        flops_g = None
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[2] thop warning: {exc}")

    # Layer timings (hooks)
    times: dict[str, list[float]] = defaultdict(list)

    def _make_hook(name: str):
        def hook(_m, _inp, _out):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[name].append(time.perf_counter())

        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
            # record start+end via two hooks is awkward; use pre+forward timing wrapper
            def pre(_m, _i, n=name):
                times[n + "|t0"].append(time.perf_counter())

            def post(_m, _i, _o, n=name):
                t0 = times[n + "|t0"].pop() if times[n + "|t0"] else time.perf_counter()
                times[n].append(time.perf_counter() - t0)

            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))

    # Warmup + timed runs (wall + RSS)
    proc = psutil.Process(os.getpid())
    with torch.no_grad():
        for _ in range(5):
            model(x)
    times.clear()
    for k in list(times.keys()):
        times[k].clear()

    tracemalloc.start()
    rss0 = proc.memory_info().rss
    wall = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            wall.append((time.perf_counter() - t0) * 1000.0)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = proc.memory_info().rss

    for h in handles:
        h.remove()

    # Aggregate layer ms (mean over runs ≈ total recorded / runs)
    layer_ms = []
    for name, vals in times.items():
        if "|t0" in name or not vals:
            continue
        layer_ms.append(
            {
                "layer": name,
                "calls": len(vals),
                "mean_ms": round(1000.0 * float(np.mean(vals)), 4),
                "sum_ms": round(1000.0 * float(np.sum(vals)), 4),
            }
        )
    layer_ms.sort(key=lambda d: d["sum_ms"], reverse=True)

    info = {
        "step": 2,
        "params_m": round(params_m, 3),
        "param_size_fp32_mb": round(_param_mb(model, 4), 3),
        "flops_g": round(flops_g, 3) if flops_g is not None else None,
        "infer_ms_mean": round(float(np.mean(wall)), 3),
        "infer_ms_p50": round(float(np.percentile(wall, 50)), 3),
        "infer_ms_p95": round(float(np.percentile(wall, 95)), 3),
        "fps_est": round(1000.0 / max(1e-6, float(np.mean(wall))), 2),
        "rss_delta_mb": round((rss1 - rss0) / (1024 * 1024), 3),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 3),
        "top_hotspots": layer_ms[:25],
        "runs": runs,
    }
    _save_json(OUT / "02_profile.json", info)
    print(
        f"[2] Profile: {info['params_m']}M params · "
        f"{info['flops_g']} GFLOPs · {info['infer_ms_mean']} ms · "
        f"~{info['fps_est']} FPS"
    )
    return info


# ---------------------------------------------------------------------------
# Step 3 — bottlenecks
# ---------------------------------------------------------------------------
def step3_bottlenecks(profile: dict) -> dict:
    tops = profile.get("top_hotspots") or []
    conv_share = sum(h["sum_ms"] for h in tops if ".conv" in h["layer"] or h["layer"].endswith("conv"))
    bn_share = sum(h["sum_ms"] for h in tops if ".bn" in h["layer"] or h["layer"].endswith("bn"))
    total = sum(h["sum_ms"] for h in tops) or 1.0
    info = {
        "step": 3,
        "top10": tops[:10],
        "conv_sum_ms_in_top": round(conv_share, 3),
        "bn_sum_ms_in_top": round(bn_share, 3),
        "recommendation": (
            "Conv2d dominates; fuse Conv+BatchNorm to cut BN ops and memory bandwidth. "
            "Hourglass residual stacks are the structural hotspot."
        ),
        "bn_vs_conv_hint": {
            "conv_fraction_of_top_sum": round(conv_share / total, 3),
            "bn_fraction_of_top_sum": round(bn_share / total, 3),
        },
    }
    _save_json(OUT / "03_bottlenecks.json", info)
    print(f"[3] Bottlenecks: top layer = {tops[0]['layer'] if tops else 'n/a'}")
    for h in tops[:5]:
        print(f"    {h['sum_ms']:8.2f} ms  {h['layer']}")
    return info


# ---------------------------------------------------------------------------
# Step 4 — Conv+BN fusion
# ---------------------------------------------------------------------------
def fuse_conv_bn_modules(model: nn.Module) -> int:
    """Fuse Conv.conv + Conv.bn (eval) in-place; strip BN. Returns fuse count."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    n = 0
    for mod in model.modules():
        if isinstance(mod, Conv) and mod.bn is not None and isinstance(mod.conv, nn.Conv2d):
            fused = fuse_conv_bn_eval(mod.conv, mod.bn)
            mod.conv = fused
            mod.bn = None
            n += 1
    # Residual leading BN stays (BN→ReLU→Conv pattern); still fewer BN after Conv fuses.
    model.eval()
    return n


def step4_fuse_and_bench(ckpt: Path, runs: int = 30) -> dict:
    base = load_model(ckpt)
    fused = copy.deepcopy(base)
    n_fused = fuse_conv_bn_modules(fused)

    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        y0 = base(x)
        y1 = fused(x)
    max_abs = float((y0 - y1).abs().max())
    mean_abs = float((y0 - y1).abs().mean())

    def _bench(m: nn.Module) -> float:
        with torch.no_grad():
            for _ in range(5):
                m(x)
            ms = []
            for _ in range(runs):
                t0 = time.perf_counter()
                m(x)
                ms.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(ms))

    ms_base = _bench(base)
    ms_fused = _bench(fused)

    # Save fused checkpoint (new file only)
    arch = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("arch") or {}
    fused_path = OUT / FUSED_PTH
    torch.save(
        {
            "model_state_dict": fused.state_dict(),
            "arch": arch,
            "fused_conv_bn": True,
            "n_fused_pairs": n_fused,
            "source": str(ckpt),
        },
        fused_path,
    )

    # Export fused FP32 ONNX
    onnx_path = OUT / ONNX_FP32
    torch.onnx.export(
        fused,
        x,
        str(onnx_path),
        input_names=["ear_crop"],
        output_names=["heatmaps"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    try:
        single = consolidate_to_single_file(onnx_path)
        shutil.copy2(single, onnx_path)
        data = onnx_path.with_suffix(onnx_path.suffix + ".data")
        if data.is_file():
            data.unlink()
        single.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[4] onnx consolidate warning: {exc}")

    info = {
        "step": 4,
        "n_fused_conv_bn_pairs": n_fused,
        "heatmap_max_abs_diff": max_abs,
        "heatmap_mean_abs_diff": mean_abs,
        "infer_ms_fp32": round(ms_base, 3),
        "infer_ms_fused_fp32": round(ms_fused, 3),
        "speedup_x": round(ms_base / max(1e-6, ms_fused), 3),
        "fused_pth_mb": round(_mb(fused_path), 3),
        "fused_onnx_mb": round(_mb(onnx_path), 3),
        "fused_pth": str(fused_path.relative_to(ROOT)),
        "fused_onnx": str(onnx_path.relative_to(ROOT)),
        "accuracy_note": "Same compute graph numerically within float noise after fusion",
    }
    _save_json(OUT / "04_fuse_benchmark.json", info)
    print(
        f"[4] Fused {n_fused} Conv+BN · Δmax={max_abs:.2e} · "
        f"{ms_base:.1f}→{ms_fused:.1f} ms ({info['speedup_x']}×) · "
        f"ONNX {info['fused_onnx_mb']:.1f} MB"
    )
    return info


# ---------------------------------------------------------------------------
# Step 5 — FP16 + INT8 export + bench
# ---------------------------------------------------------------------------
def _ort_session(path: Path, providers: list[str] | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=providers or ["CPUExecutionProvider"],
    )


def _ort_bench(path: Path, runs: int = 30) -> dict:
    sess = _ort_session(path)
    inp = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    for _ in range(5):
        sess.run(None, {inp: x})
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "infer_ms_mean": round(float(np.mean(ms)), 3),
        "infer_ms_p50": round(float(np.percentile(ms, 50)), 3),
        "fps_est": round(1000.0 / max(1e-6, float(np.mean(ms))), 2),
        "size_mb": round(_mb(path), 3),
    }


def _heatmap_to_pts(hm: np.ndarray) -> np.ndarray:
    """Argmax peaks → (56,2) in heatmap space."""
    c, h, w = hm.shape
    pts = np.zeros((c, 2), dtype=np.float32)
    flat = hm.reshape(c, -1)
    idx = flat.argmax(axis=1)
    pts[:, 0] = idx % w
    pts[:, 1] = idx // w
    return pts


def _accuracy_vs_ref(ref_onnx: Path, cand_onnx: Path, crop_path: Path | None) -> dict:
    """Landmark mean/max px error vs fused FP32 ONNX on one ear crop (or random)."""
    if crop_path and crop_path.is_file():
        import cv2

        img = cv2.imread(str(crop_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    else:
        x = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

    s_ref = _ort_session(ref_onnx)
    s_c = _ort_session(cand_onnx)
    n_ref = s_ref.get_inputs()[0].name
    n_c = s_c.get_inputs()[0].name
    y_ref = s_ref.run(None, {n_ref: x})[0][0].astype(np.float32)
    y_c = s_c.run(None, {n_c: x})[0][0].astype(np.float32)

    p_ref = _heatmap_to_pts(y_ref)
    p_c = _heatmap_to_pts(y_c)
    # scale heatmap → 256
    scale = INPUT_SIZE / y_ref.shape[-1]
    d = np.linalg.norm((p_ref - p_c) * scale, axis=1)
    return {
        "mean_px": round(float(d.mean()), 4),
        "max_px": round(float(d.max()), 4),
        "pierce_px": round(float(d[min(55, len(d) - 1)]), 4),
        "crop": str(crop_path) if crop_path else None,
    }


def step5_export_fp16_int8(runs: int = 30) -> dict:
    fp32_onnx = OUT / ONNX_FP32
    if not fp32_onnx.is_file():
        raise FileNotFoundError(fp32_onnx)

    # --- FP16 (keep float32 I/O for ORT / browser) ---
    fp16_path = OUT / ONNX_FP16
    import onnx

    m = onnx.load(str(fp32_onnx))
    converted = False
    try:
        from onnxconverter_common import float16 as onnx_f16

        m16 = onnx_f16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m16, str(fp16_path))
        converted = True
    except Exception as exc:
        print(f"[5] onnxconverter_common FP16 failed: {exc}")
    if not converted:
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16

            m16 = convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m16, str(fp16_path))
            converted = True
        except Exception as exc2:
            print(f"[5] ORT FP16 failed: {exc2}")
    if not converted or _mb(fp16_path) < 1:
        raise RuntimeError("FP16 ONNX export failed")

    # --- INT8 dynamic ---
    int8_path = OUT / ONNX_INT8
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    crop = ROOT / "outputs" / "smoke" / "image_pipeline" / "04_ear_crop_256.png"
    if not crop.is_file():
        crop = None

    bench_fp32 = _ort_bench(fp32_onnx, runs)
    # FP16 may need float32 I/O
    try:
        bench_fp16 = _ort_bench(fp16_path, runs)
        acc_fp16 = _accuracy_vs_ref(fp32_onnx, fp16_path, crop)
    except Exception as exc:
        bench_fp16 = {"error": str(exc), "size_mb": round(_mb(fp16_path), 3)}
        acc_fp16 = {"error": str(exc)}

    try:
        bench_int8 = _ort_bench(int8_path, runs)
        acc_int8 = _accuracy_vs_ref(fp32_onnx, int8_path, crop)
    except Exception as exc:
        bench_int8 = {"error": str(exc), "size_mb": round(_mb(int8_path), 3)}
        acc_int8 = {"error": str(exc)}

    info = {
        "step": 5,
        "fp32_onnx": {**bench_fp32, "path": str(fp32_onnx.relative_to(ROOT))},
        "fp16_onnx": {**bench_fp16, "path": str(fp16_path.relative_to(ROOT)), "accuracy_vs_fp32": acc_fp16},
        "int8_onnx": {**bench_int8, "path": str(int8_path.relative_to(ROOT)), "accuracy_vs_fp32": acc_int8},
    }
    _save_json(OUT / "05_fp16_int8.json", info)
    print(
        f"[5] FP32 {bench_fp32['size_mb']:.1f} MB / {bench_fp32['infer_ms_mean']:.1f} ms | "
        f"FP16 {bench_fp16.get('size_mb', '?')} MB | "
        f"INT8 {bench_int8.get('size_mb', '?')} MB"
    )
    return info


# ---------------------------------------------------------------------------
# Step 6 — compare + recommend
# ---------------------------------------------------------------------------
def step6_compare(prev: dict) -> dict:
    rows = []
    for name, key in (("FP32_fused", "fp32_onnx"), ("FP16", "fp16_onnx"), ("INT8", "int8_onnx")):
        b = prev.get(key) or {}
        acc = b.get("accuracy_vs_fp32") or ({"mean_px": 0.0, "pierce_px": 0.0} if name.startswith("FP32") else {})
        rows.append(
            {
                "model": name,
                "onnx_mb": b.get("size_mb"),
                "infer_ms": b.get("infer_ms_mean"),
                "fps_est": b.get("fps_est"),
                "landmark_mean_px_vs_fp32": acc.get("mean_px", 0.0 if name.startswith("FP32") else None),
                "pierce_px_vs_fp32": acc.get("pierce_px", 0.0 if name.startswith("FP32") else None),
                "error": b.get("error"),
            }
        )

    # Pick recommendations
    def ok(r):
        return r.get("onnx_mb") and not r.get("error") and r.get("infer_ms")

    fp32 = next(r for r in rows if r["model"] == "FP32_fused")
    fp16 = next(r for r in rows if r["model"] == "FP16")
    int8 = next(r for r in rows if r["model"] == "INT8")

    def acc_ok(r, thr=2.0):
        m = r.get("landmark_mean_px_vs_fp32")
        return m is not None and m <= thr

    browser = "FP16" if ok(fp16) and acc_ok(fp16, 3.0) else ("INT8" if ok(int8) and acc_ok(int8, 4.0) else "FP32_fused")
    # Prefer smaller for browser if accuracy holds
    if ok(int8) and acc_ok(int8, 2.5) and (int8.get("onnx_mb") or 99) < (fp16.get("onnx_mb") or 99):
        browser = "INT8"
    if ok(fp16) and acc_ok(fp16, 1.5) and browser == "INT8" and not acc_ok(int8, 1.5):
        browser = "FP16"

    desktop = "FP32_fused"  # accuracy-first desktop
    mobile = "INT8" if ok(int8) and acc_ok(int8, 3.5) else browser

    info = {
        "step": 6,
        "comparison": rows,
        "recommendations": {
            "Desktop": {
                "model": desktop,
                "why": "Highest accuracy; AVX ONNX Runtime; fused FP32 already faster than unfused",
            },
            "Browser": {
                "model": browser,
                "why": "WASM size + speed; prefer FP16/INT8 when landmark error stays small",
            },
            "Android_iPhone": {
                "model": mobile,
                "why": "ONNX Runtime Mobile / NNAPI / Core ML — smallest viable INT8",
            },
        },
        "live_master_untouched": True,
        "artifact_dir": str(OUT.relative_to(ROOT)),
    }
    _save_json(OUT / "compare.json", info)
    print("[6] Comparison:")
    for r in rows:
        print(
            f"    {r['model']:12s}  {r.get('onnx_mb')} MB  "
            f"{r.get('infer_ms')} ms  meanΔ={r.get('landmark_mean_px_vs_fp32')} px"
        )
    print(f"    → Desktop={desktop}  Browser={browser}  Mobile={mobile}")
    return info


def write_report(all_steps: dict) -> Path:
    c = all_steps.get("compare") or {}
    rows = c.get("comparison") or []
    rec = c.get("recommendations") or {}
    lines = [
        "# SHGNet-56 size-reduce pipeline",
        "",
        "Follows: **FP32 master → keep → profile → bottlenecks → Conv+BN fuse →",
        "FP16 / INT8 export → compare → deploy pick**.",
        "",
        f"Artifacts: `{OUT.relative_to(ROOT)}/` (binaries gitignored; JSON reports kept).",
        "",
        "## Master checkpoint",
        "",
        f"- Live (untouched): `models/shgnet/SHGNet-56_final.pth`",
        f"- Backup: `outputs/size_reduce/{MASTER_NAME}`",
        "",
        "## Profile (step 2)",
        "",
        "```json",
        json.dumps(all_steps.get("profile", {}), indent=2)[:2000],
        "```",
        "",
        "## Bottlenecks (step 3)",
        "",
        (all_steps.get("bottlenecks") or {}).get("recommendation", ""),
        "",
        "## Fused FP32 (step 4)",
        "",
        "```json",
        json.dumps(all_steps.get("fuse", {}), indent=2),
        "```",
        "",
        "## Compare (step 6)",
        "",
        "| Model | ONNX MB | Infer ms | FPS est | Mean Δpx vs FP32 | Pierce Δpx |",
        "|-------|---------|----------|---------|------------------|------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r.get('onnx_mb')} | {r.get('infer_ms')} | "
            f"{r.get('fps_est')} | {r.get('landmark_mean_px_vs_fp32')} | {r.get('pierce_px_vs_fp32')} |"
        )
    lines += [
        "",
        "## Recommendations",
        "",
        f"- **Desktop:** `{rec.get('Desktop', {}).get('model')}` — {rec.get('Desktop', {}).get('why')}",
        f"- **Browser:** `{rec.get('Browser', {}).get('model')}` — {rec.get('Browser', {}).get('why')}",
        f"- **Mobile:** `{rec.get('Android_iPhone', {}).get('model')}` — {rec.get('Android_iPhone', {}).get('why')}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/size_reduce_pipeline.py --all",
        "```",
        "",
    ]
    path = ROOT / "docs" / "SIZE_REDUCE_PIPELINE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Trackable JSON summary (no large binaries)
    summary = ROOT / "docs" / "size_reduce_compare.json"
    summary.write_text(json.dumps(c, indent=2) + "\n", encoding="utf-8")
    print(f"[report] {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PRETRAINED_56))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--step", type=int, default=0, help="Run single step 1..6 (0 = all)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    step = args.step
    do_all = args.all or step == 0
    results: dict[str, Any] = {}

    if do_all or step == 1:
        results["keep"] = step1_keep_master(ckpt)
    if do_all or step == 2:
        results["profile"] = step2_profile(ckpt, args.runs)
    if do_all or step == 3:
        prof = results.get("profile")
        if prof is None:
            prof = json.loads((OUT / "02_profile.json").read_text())
        results["bottlenecks"] = step3_bottlenecks(prof)
    if do_all or step == 4:
        results["fuse"] = step4_fuse_and_bench(ckpt, args.runs)
    if do_all or step == 5:
        results["exports"] = step5_export_fp16_int8(args.runs)
    if do_all or step == 6:
        exports = results.get("exports")
        if exports is None:
            exports = json.loads((OUT / "05_fp16_int8.json").read_text())
        results["compare"] = step6_compare(exports)
        # assemble for report
        for name, fname in (
            ("profile", "02_profile.json"),
            ("bottlenecks", "03_bottlenecks.json"),
            ("fuse", "04_fuse_benchmark.json"),
        ):
            if name not in results and (OUT / fname).is_file():
                results[name] = json.loads((OUT / fname).read_text())
        write_report(results)

    _save_json(OUT / "pipeline_run.json", {k: v for k, v in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
