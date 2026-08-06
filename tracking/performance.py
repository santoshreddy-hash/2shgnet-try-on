"""Adaptive performance profiles: detect device capability, load settings, scale dynamically."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _ROOT / "performance_profiles.json"

ProfileName = Literal["high", "medium", "low"]
PerfMode = Literal["auto", "high", "medium", "low"]


@dataclass
class OnnxRuntimeOpts:
    graph_optimization: str = "all"
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    enable_mem_pattern: bool = True
    enable_cpu_mem_arena: bool = True


@dataclass
class PerformanceProfile:
    name: ProfileName
    label: str
    target_fps: int
    camera_width: int
    camera_height: int
    process_width: int
    process_height: int
    yolo_every: int
    shg_every: int
    flip_inference: str  # "adaptive" | "off" | "always"
    flip_score_threshold: float
    min_shg_score: float
    roi_tracking: bool
    landmark_reuse: bool
    yolo_on_track_lost_only: bool
    yolo_conf_drop: float
    skip_shg_on_still: bool
    still_motion_px: float
    stick_n_track: int
    adapt_cap: int
    allow_resolution_scale: bool
    overlay_only_render: bool
    reuse_preprocess_buffers: bool
    fps_max: int = 25
    infer_width: int = 640
    infer_height: int = 360
    yolo_imgsz: int = 640
    freeze_camera_ladder: bool = True
    onnx: OnnxRuntimeOpts = field(default_factory=OnnxRuntimeOpts)


@dataclass
class DynamicScalingConfig:
    enabled: bool = True
    drop_below_ratio: float = 0.72
    recover_above_ratio: float = 0.92
    cooldown_frames: int = 20
    warmup_frames: int = 45
    max_extra_skip: int = 2
    resolution_scale_step: float = 0.90
    min_resolution_scale: float = 0.75
    work_budget_ratio: float = 0.92
    recover_work_ratio: float = 0.70


@dataclass
class DeviceCapability:
    cpu_cores: int
    ram_gb: float
    has_gpu_ep: bool
    gpu_providers: list[str]
    platform: str
    score: float
    recommended: ProfileName
    detail: str


def _load_raw(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or _DEFAULT_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing performance config: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _onnx_from_dict(d: dict[str, Any] | None) -> OnnxRuntimeOpts:
    d = d or {}
    return OnnxRuntimeOpts(
        graph_optimization=str(d.get("graph_optimization", "all")),
        intra_op_num_threads=int(d.get("intra_op_num_threads", 0)),
        inter_op_num_threads=int(d.get("inter_op_num_threads", 0)),
        enable_mem_pattern=bool(d.get("enable_mem_pattern", True)),
        enable_cpu_mem_arena=bool(d.get("enable_cpu_mem_arena", True)),
    )


_QUALITY_DEFAULTS: dict[str, Any] = {
    "infer_width": 640,
    "infer_height": 360,
    "yolo_imgsz": 640,
    "flip_inference": "adaptive",
    "flip_score_threshold": 0.12,
    "min_shg_score": 0.08,
    "yolo_conf_drop": 0.08,
    "skip_shg_on_still": False,
    "still_motion_px": 1.5,
    "stick_n_track": 16,
    "roi_tracking": True,
    "landmark_reuse": True,
    "yolo_on_track_lost_only": True,
    "allow_resolution_scale": False,
    "freeze_camera_ladder": True,
    "overlay_only_render": True,
    "reuse_preprocess_buffers": True,
}


def merge_quality(profile_d: dict[str, Any], quality: dict[str, Any] | None) -> dict[str, Any]:
    """Quality knobs win over per-profile overrides so high/low stay result-identical."""
    q = {**_QUALITY_DEFAULTS, **(quality or {})}
    q.pop("comment", None)
    merged = {**profile_d, **q}
    # Canonical infer size drives process size (camera may match).
    iw = int(merged.get("infer_width", 640))
    ih = int(merged.get("infer_height", 360))
    merged["process_width"] = iw
    merged["process_height"] = ih
    merged["camera_width"] = int(merged.get("camera_width", iw))
    merged["camera_height"] = int(merged.get("camera_height", ih))
    # Never let weak-device profiles disable flip or still-skip SHG.
    merged["flip_inference"] = str(q.get("flip_inference", "adaptive"))
    merged["skip_shg_on_still"] = bool(q.get("skip_shg_on_still", False))
    merged["allow_resolution_scale"] = bool(q.get("allow_resolution_scale", False))
    return merged


def profile_from_dict(name: ProfileName, d: dict[str, Any]) -> PerformanceProfile:
    default_fps_max = 25
    iw = int(d.get("infer_width", d.get("process_width", 640)))
    ih = int(d.get("infer_height", d.get("process_height", 360)))
    return PerformanceProfile(
        name=name,
        label=str(d.get("label", name)),
        target_fps=int(d["target_fps"]),
        camera_width=int(d.get("camera_width", iw)),
        camera_height=int(d.get("camera_height", ih)),
        process_width=iw,
        process_height=ih,
        yolo_every=max(1, int(d["yolo_every"])),
        shg_every=max(1, int(d["shg_every"])),
        flip_inference=str(d.get("flip_inference", "adaptive")),
        flip_score_threshold=float(d.get("flip_score_threshold", 0.12)),
        min_shg_score=float(d.get("min_shg_score", 0.08)),
        roi_tracking=bool(d.get("roi_tracking", True)),
        landmark_reuse=bool(d.get("landmark_reuse", True)),
        yolo_on_track_lost_only=bool(d.get("yolo_on_track_lost_only", True)),
        yolo_conf_drop=float(d.get("yolo_conf_drop", 0.08)),
        skip_shg_on_still=bool(d.get("skip_shg_on_still", False)),
        still_motion_px=float(d.get("still_motion_px", 1.5)),
        stick_n_track=int(d.get("stick_n_track", 16)),
        adapt_cap=int(d.get("adapt_cap", 4)),
        allow_resolution_scale=bool(d.get("allow_resolution_scale", False)),
        overlay_only_render=bool(d.get("overlay_only_render", True)),
        reuse_preprocess_buffers=bool(d.get("reuse_preprocess_buffers", True)),
        fps_max=int(d.get("fps_max", default_fps_max)),
        infer_width=iw,
        infer_height=ih,
        yolo_imgsz=int(d.get("yolo_imgsz", 640)),
        freeze_camera_ladder=bool(d.get("freeze_camera_ladder", True)),
        onnx=_onnx_from_dict(d.get("onnx")),
    )


def load_profiles(path: Path | None = None) -> tuple[dict[str, PerformanceProfile], dict[str, Any], DynamicScalingConfig]:
    raw = _load_raw(path)
    quality = dict(raw.get("quality") or {})
    profiles: dict[str, PerformanceProfile] = {}
    for name in ("high", "medium", "low"):
        if name not in raw.get("profiles", {}):
            # Backward compat: synthesize medium between high/low if missing
            if name == "medium" and "high" in raw.get("profiles", {}) and "low" in raw.get("profiles", {}):
                mid = dict(raw["profiles"]["high"])
                mid["label"] = mid.get("label", "Medium-Performance")
                mid["target_fps"] = int(raw["profiles"]["high"].get("target_fps", 25))
                mid["shg_every"] = int(raw["profiles"]["high"].get("shg_every", 1))
                merged = merge_quality(mid, quality)
                profiles[name] = profile_from_dict(name, merged)  # type: ignore[arg-type]
                continue
            raise KeyError(f"performance_profiles.json missing profiles.{name}")
        merged = merge_quality(dict(raw["profiles"][name]), quality)
        profiles[name] = profile_from_dict(name, merged)  # type: ignore[arg-type]
    auto = dict(raw.get("auto_detect", {}))
    ds_raw = raw.get("dynamic_scaling", {})
    # Quality freeze: never shrink infer resolution under load.
    allow_res = bool(quality.get("allow_resolution_scale", False))
    dynamic = DynamicScalingConfig(
        enabled=bool(ds_raw.get("enabled", True)),
        drop_below_ratio=float(ds_raw.get("drop_below_ratio", 0.72)),
        recover_above_ratio=float(ds_raw.get("recover_above_ratio", 0.92)),
        cooldown_frames=int(ds_raw.get("cooldown_frames", 20)),
        warmup_frames=int(ds_raw.get("warmup_frames", 45)),
        max_extra_skip=int(ds_raw.get("max_extra_skip", 1)),
        resolution_scale_step=float(ds_raw.get("resolution_scale_step", 1.0 if not allow_res else 0.90)),
        min_resolution_scale=float(ds_raw.get("min_resolution_scale", 1.0 if not allow_res else 0.75)),
        work_budget_ratio=float(ds_raw.get("work_budget_ratio", 0.92)),
        recover_work_ratio=float(ds_raw.get("recover_work_ratio", 0.70)),
    )
    if not allow_res:
        dynamic.min_resolution_scale = 1.0
        dynamic.resolution_scale_step = 1.0
    return profiles, auto, dynamic


def _ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return float(out) / (1024.0**3)
        if platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return kb / (1024.0**2)
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / (1024.0**3)
    except Exception:
        pass
    return 0.0


def _ort_gpu_providers() -> list[str]:
    try:
        import onnxruntime as ort

        avail = set(ort.get_available_providers())
        gpuish = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "TensorrtExecutionProvider",
        ]
        return [p for p in gpuish if p in avail]
    except Exception:
        return []


def detect_device_capability(auto_cfg: dict[str, Any] | None = None) -> DeviceCapability:
    auto_cfg = auto_cfg or {}
    cores = max(1, int(os.cpu_count() or 1))
    ram = _ram_gb()
    gpu_providers = _ort_gpu_providers()
    has_gpu = len(gpu_providers) > 0

    min_cores = int(auto_cfg.get("high_min_cpu_cores", 6))
    min_ram = float(auto_cfg.get("high_min_ram_gb", 8))
    high_if_gpu = bool(auto_cfg.get("high_if_gpu_ep", True))
    high_min_score = float(auto_cfg.get("high_min_score", 75))
    medium_min_score = float(auto_cfg.get("medium_min_score", 45))

    score = 0.0
    score += min(cores / float(min_cores), 1.5) * 40.0
    if ram > 0:
        score += min(ram / float(min_ram), 1.5) * 30.0
    else:
        score += 15.0
    if has_gpu:
        score += 30.0

    recommend: ProfileName = "low"
    reasons: list[str] = []
    if (has_gpu and high_if_gpu):
        recommend = "high"
        reasons.append(f"GPU EP={gpu_providers[0]}")
    elif cores >= min_cores and ram >= min_ram:
        recommend = "high"
        reasons.append(f"CPU={cores} cores RAM={ram:.1f}GB")
    elif cores >= min_cores and ram <= 0 and score >= medium_min_score:
        # RAM probe failed — allow high only with solid score prior
        recommend = "high"
        reasons.append(f"CPU={cores} cores RAM=? score={score:.0f}")
    elif score >= high_min_score:
        recommend = "high"
        reasons.append(f"score={score:.0f}≥{high_min_score:.0f}")
    elif score >= medium_min_score or cores >= max(4, min_cores - 2):
        recommend = "medium"
        reasons.append(f"score={score:.0f} → medium (cores={cores} RAM={ram:.1f}GB)")
    else:
        reasons.append(f"CPU={cores} cores RAM={ram:.1f}GB score={score:.0f} → low")

    return DeviceCapability(
        cpu_cores=cores,
        ram_gb=ram,
        has_gpu_ep=has_gpu,
        gpu_providers=gpu_providers,
        platform=platform.platform(),
        score=float(score),
        recommended=recommend,
        detail="; ".join(reasons) or "default",
    )


def resolve_profile(
    mode: PerfMode = "auto",
    path: Path | None = None,
    *,
    force_benchmark: bool = False,
) -> tuple[PerformanceProfile, DeviceCapability, DynamicScalingConfig]:
    """Pick high/medium/low from --performance mode (auto detects at startup)."""
    profiles, auto_cfg, dynamic = load_profiles(path)
    capability = detect_device_capability(auto_cfg)

    if mode in ("high", "medium", "low"):
        chosen = profiles[mode]  # type: ignore[index]
        capability.recommended = mode  # type: ignore[assignment]
    else:
        chosen = profiles[capability.recommended]
        # Optional micro-benchmark can nudge recommendation on borderline machines
        if force_benchmark or (capability.score < 85 and capability.score > 40):
            bench = _quick_cpu_bench(int(auto_cfg.get("benchmark_frames", 3)))
            min_fps = float(auto_cfg.get("high_min_bench_fps", 22.0))
            if bench is not None:
                capability.detail += f"; bench≈{bench:.1f}fps"
                if bench < min_fps * 0.7 and chosen.name == "high":
                    chosen = profiles["medium"]
                    capability.recommended = "medium"
                    capability.detail += " → demote to medium"
                elif bench < min_fps * 0.45 and chosen.name in ("high", "medium"):
                    chosen = profiles["low"]
                    capability.recommended = "low"
                    capability.detail += " → demote to low"
                elif bench >= min_fps and chosen.name == "low" and capability.has_gpu_ep:
                    chosen = profiles["high"]
                    capability.recommended = "high"
                    capability.detail += " → promote to high"

    return chosen, capability, dynamic


def _quick_cpu_bench(frames: int = 3) -> Optional[float]:
    """Tiny numpy/OpenCV workload as a coarse capability probe."""
    try:
        import cv2

        frames = max(1, frames)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(frames):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _ = cv2.GaussianBlur(g, (5, 5), 0)
            _ = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        dt = time.perf_counter() - t0
        if dt <= 0:
            return None
        # Normalize: ~3 frames of light work; map to rough "pipeline FPS" estimate
        return float(frames / dt) * 0.15
    except Exception:
        return None


@dataclass
class DynamicScaler:
    """Temporarily sparsify inference / shrink resolution when FPS falls below target."""

    base: PerformanceProfile
    cfg: DynamicScalingConfig
    adapt_yolo: int = 0
    adapt_shg: int = 0
    resolution_scale: float = 1.0
    _cooldown: int = 0
    _stressed: bool = False
    _frame: int = 0

    def __post_init__(self) -> None:
        self.adapt_yolo = int(self.base.yolo_every)
        self.adapt_shg = int(self.base.shg_every)
        self.resolution_scale = 1.0
        self._cooldown = 0
        self._stressed = False
        self._frame = 0

    def reset(self) -> None:
        self.__post_init__()

    @property
    def process_size(self) -> tuple[int, int]:
        """Canonical infer size. Resolution scale is ignored when quality-frozen."""
        if not self.base.allow_resolution_scale:
            w = max(160, int(self.base.infer_width or self.base.process_width))
            h = max(120, int(self.base.infer_height or self.base.process_height))
        else:
            w = max(320, int(round(self.base.process_width * self.resolution_scale)))
            h = max(240, int(round(self.base.process_height * self.resolution_scale)))
        # keep even dims for codecs
        return w - (w % 2), h - (h % 2)

    def update(self, inst_fps: float, work_ms: float = 0.0) -> None:
        """Scale using work-time budget first; FPS is a secondary signal after warmup."""
        if not self.cfg.enabled:
            return
        target = float(self.base.target_fps)
        if target <= 0:
            return

        self._frame += 1
        # Ignore model/camera warmup — first frames are always slow and used to
        # permanently demote high mode to Y/4 @ reduced res before recovery could fire.
        if self._frame <= max(1, int(self.cfg.warmup_frames)):
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            return

        budget_ms = (1000.0 / target) * float(self.cfg.work_budget_ratio)
        recover_ms = (1000.0 / target) * float(self.cfg.recover_work_ratio)
        drop_fps = target * self.cfg.drop_below_ratio
        recover_fps = target * self.cfg.recover_above_ratio

        max_yolo = min(self.base.adapt_cap, self.base.yolo_every + self.cfg.max_extra_skip)
        max_shg = min(self.base.adapt_cap, self.base.shg_every + self.cfg.max_extra_skip)

        overloaded = (work_ms > budget_ms) or (inst_fps > 0 and inst_fps < drop_fps)
        recovered = (0 < work_ms <= recover_ms) or (inst_fps >= recover_fps)

        if overloaded:
            self._stressed = True
            # Keep YOLO fixed at profile cadence (Y/2); sparsify SHG / resolution only.
            self.adapt_shg = min(max_shg, self.adapt_shg + 1)
            self.adapt_yolo = int(self.base.yolo_every)
            if self.base.allow_resolution_scale:
                new_scale = max(
                    self.cfg.min_resolution_scale,
                    self.resolution_scale * self.cfg.resolution_scale_step,
                )
                if new_scale < self.resolution_scale - 1e-6:
                    self.resolution_scale = new_scale
            self._cooldown = self.cfg.cooldown_frames
        elif recovered and self._stressed:
            improved = False
            if self.adapt_shg > self.base.shg_every:
                self.adapt_shg = max(self.base.shg_every, self.adapt_shg - 1)
                improved = True
            if self.adapt_yolo > self.base.yolo_every:
                self.adapt_yolo = max(self.base.yolo_every, self.adapt_yolo - 1)
                improved = True
            if self.base.allow_resolution_scale and self.resolution_scale < 0.999:
                self.resolution_scale = min(
                    1.0, self.resolution_scale / max(1e-6, self.cfg.resolution_scale_step)
                )
                improved = True
            elif not self.base.allow_resolution_scale:
                self.resolution_scale = 1.0
            if (
                self.adapt_yolo <= self.base.yolo_every
                and self.adapt_shg <= self.base.shg_every
                and self.resolution_scale >= 0.999
            ):
                self._stressed = False
                self.resolution_scale = 1.0
            if improved:
                self._cooldown = self.cfg.cooldown_frames

    def snapshot_overrides(self) -> PerformanceProfile:
        """Return a profile copy with current adaptive cadence (for logging)."""
        return replace(
            self.base,
            yolo_every=self.adapt_yolo,
            shg_every=self.adapt_shg,
            process_width=self.process_size[0],
            process_height=self.process_size[1],
        )
