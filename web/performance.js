/**
 * Browser adaptive performance — same profiles as desktop
 * (`performance_profiles.json`). Detects high / medium / low.
 *
 * Quality is frozen (640×360, LEFT flip rules, crop). All tiers target ~25 FPS
 * with SHG every 1 (max 2 under load). Tip-hold keeps UI smooth with no lag.
 */

const FPS_MIN = 20;
const FPS_MAX_HIGH = 25;
const FPS_MAX_LOW = 25;

const DEFAULT_PROFILES = {
  auto_detect: {
    high_min_cpu_cores: 6,
    high_min_ram_gb: 8,
    high_if_gpu_ep: true,
    medium_min_score: 45,
    high_min_score: 75,
  },
  browser_auto_detect: {
    high_min_cpu_cores: 6,
    high_min_device_memory_gb: 4,
    mobile_default_low: false,
    mobile_cap: "medium",
    high_if_webgpu: false,
    webgpu_score_bonus: 0,
    medium_min_score: 45,
    high_min_score: 70,
    low_max_cores: 3,
  },
  quality: {
    infer_width: 640,
    infer_height: 360,
    yolo_imgsz: 640,
    flip_inference: "adaptive",
    left_always_flip_compare: true,
    left_flip_recheck_every: 8,
    freeze_camera_ladder: true,
    allow_resolution_scale: false,
  },
  dynamic_scaling: {
    enabled: true,
    drop_below_ratio: 0.65,
    recover_above_ratio: 0.9,
    cooldown_frames: 30,
    warmup_frames: 45,
    max_extra_skip: 1,
    work_budget_ratio: 0.98,
    recover_work_ratio: 0.75,
  },
  profiles: {
    high: {
      label: "High-Performance",
      target_fps: 25,
      camera_width: 640,
      camera_height: 360,
      process_width: 640,
      process_height: 360,
      yolo_every: 1,
      shg_every: 1,
      fps_max: 25,
      yolo_imgsz: 640,
    },
    medium: {
      label: "Medium-Performance",
      target_fps: 25,
      camera_width: 640,
      camera_height: 360,
      process_width: 640,
      process_height: 360,
      yolo_every: 1,
      shg_every: 1,
      fps_max: 25,
      yolo_imgsz: 640,
    },
    low: {
      label: "Low-Compute",
      target_fps: 25,
      camera_width: 640,
      camera_height: 360,
      process_width: 640,
      process_height: 360,
      yolo_every: 1,
      shg_every: 1,
      fps_max: 25,
      yolo_imgsz: 640,
    },
  },
  browser: {
    high: { shg_every: 1, shg_every_max: 2, target_fps: 25, yolo_imgsz: 640 },
    medium: { shg_every: 1, shg_every_max: 2, target_fps: 25, yolo_imgsz: 640 },
    low: { shg_every: 1, shg_every_max: 2, target_fps: 25, yolo_imgsz: 640 },
  },
  camera_ladder: [{ width: 640, height: 360 }],
};

/** @type {any} */
let cachedRaw = null;

export async function loadPerformanceConfig() {
  if (cachedRaw) return cachedRaw;
  try {
    const res = await fetch("/performance_profiles.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    cachedRaw = await res.json();
  } catch (e) {
    console.warn("[perf] using built-in profiles", e);
    cachedRaw = structuredClone
      ? structuredClone(DEFAULT_PROFILES)
      : JSON.parse(JSON.stringify(DEFAULT_PROFILES));
  }
  return cachedRaw;
}

function isMobileLike() {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  if (/Android|iPhone|iPad|iPod|Mobile|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua))
    return true;
  if (navigator.userAgentData?.mobile) return true;
  const coarse =
    typeof matchMedia === "function" &&
    matchMedia("(pointer: coarse)").matches &&
    typeof matchMedia === "function" &&
    matchMedia("(max-width: 900px)").matches;
  return !!coarse;
}

async function hasWebGpu() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

/**
 * Pure classifier — injectable for tests.
 * @param {{
 *   cores: number,
 *   deviceMemoryGb: number,
 *   mobile: boolean,
 *   webgpu: boolean,
 * }} signals
 * @param {any} [cfg]
 */
export function classifyBrowserCapability(signals, cfg = {}) {
  const conf = { ...DEFAULT_PROFILES.browser_auto_detect, ...cfg };
  const cores = Math.max(1, Number(signals.cores) || 2);
  const deviceMemoryGb = Number(signals.deviceMemoryGb) || 0;
  const mobile = !!signals.mobile;
  const webgpu = !!signals.webgpu;

  const minCores = Number(conf.high_min_cpu_cores) || 6;
  const minMem = Number(conf.high_min_device_memory_gb) || 4;
  const highIfWebgpu = conf.high_if_webgpu === true;
  const webgpuBonus = Number(conf.webgpu_score_bonus) || 0;
  const highMinScore = Number(conf.high_min_score) || 70;
  const mediumMinScore = Number(conf.medium_min_score) || 45;
  const lowMaxCores = Number(conf.low_max_cores) || 3;
  const mobileDefaultLow = conf.mobile_default_low === true;
  const mobileCap = String(conf.mobile_cap || "").toLowerCase();

  let score = 0;
  score += Math.min(cores / minCores, 1.5) * 40;
  if (deviceMemoryGb > 0) {
    score += Math.min(deviceMemoryGb / minMem, 1.5) * 30;
  } else {
    // Safari/Firefox often omit deviceMemory — mild prior, not a free high boost
    score += mobile ? 10 : 16;
  }
  if (webgpu) score += webgpuBonus;
  if (mobile) score -= 12;

  /** @type {'high'|'medium'|'low'} */
  let recommended = "medium";
  const reasons = [];

  if (mobile && mobileDefaultLow) {
    recommended = "low";
    reasons.push("mobile_default_low");
  } else if (cores <= lowMaxCores && score < mediumMinScore) {
    recommended = "low";
    reasons.push(`weak CPU cores=${cores} score=${score.toFixed(0)}`);
  } else if (
    highIfWebgpu &&
    webgpu &&
    cores >= minCores - 2 &&
    score >= highMinScore
  ) {
    recommended = "high";
    reasons.push(`WebGPU + score=${score.toFixed(0)}`);
  } else if (
    cores >= minCores &&
    (deviceMemoryGb <= 0 || deviceMemoryGb >= minMem) &&
    score >= mediumMinScore
  ) {
    recommended = "high";
    reasons.push(`CPU=${cores} mem=${deviceMemoryGb || "?"}GB score=${score.toFixed(0)}`);
  } else if (score >= highMinScore && cores >= Math.max(4, minCores - 2)) {
    recommended = "high";
    reasons.push(`score=${score.toFixed(0)}`);
  } else if (score >= mediumMinScore || cores >= 4) {
    recommended = "medium";
    reasons.push(`score=${score.toFixed(0)} → medium`);
  } else {
    recommended = "low";
    reasons.push(`score=${score.toFixed(0)} → low (cores=${cores})`);
  }

  // Phones/tablets: WASM ONNX is heavy — cap unless explicitly uncapped
  if (mobile && mobileCap === "medium" && recommended === "high") {
    recommended = "medium";
    reasons.push("mobile_cap=medium");
  } else if (mobile && mobileCap === "low" && recommended !== "low") {
    recommended = "low";
    reasons.push("mobile_cap=low");
  }

  return {
    recommended,
    cores,
    deviceMemoryGb,
    mobile,
    webgpu,
    score,
    detail: reasons.join("; ") || "default",
  };
}

/**
 * @param {any} raw
 * @returns {Promise<{
 *   recommended: 'high'|'medium'|'low',
 *   cores: number,
 *   deviceMemoryGb: number,
 *   mobile: boolean,
 *   webgpu: boolean,
 *   score: number,
 *   detail: string,
 * }>}
 */
export async function detectBrowserCapability(raw) {
  const cfg = {
    ...DEFAULT_PROFILES.browser_auto_detect,
    ...(raw?.browser_auto_detect || {}),
  };
  const cores = Math.max(1, Number(navigator.hardwareConcurrency) || 2);
  const deviceMemoryGb = Number(navigator.deviceMemory) || 0;
  const mobile = isMobileLike();
  const webgpu = await hasWebGpu();
  return classifyBrowserCapability(
    { cores, deviceMemoryGb, mobile, webgpu },
    cfg
  );
}

function qualityFromRaw(raw) {
  return {
    ...DEFAULT_PROFILES.quality,
    ...(raw?.quality || {}),
  };
}

function pickProfileDict(raw, name) {
  const quality = qualityFromRaw(raw);
  const base = {
    ...(DEFAULT_PROFILES.profiles[name] || DEFAULT_PROFILES.profiles.low),
    ...(raw?.profiles?.[name] || {}),
  };
  const browserOv = {
    ...(DEFAULT_PROFILES.browser?.[name] || {}),
    ...(raw?.browser?.[name] || {}),
  };
  const merged = { ...base, ...browserOv };
  const iw = Math.max(160, Number(quality.infer_width) || 640);
  const ih = Math.max(120, Number(quality.infer_height) || 360);
  // Quality wins — same infer geometry on every device.
  merged.process_width = iw;
  merged.process_height = ih;
  merged.camera_width = iw;
  merged.camera_height = ih;
  merged.yolo_imgsz = Number(quality.yolo_imgsz) || 640;
  merged.freeze_camera_ladder = quality.freeze_camera_ladder !== false;
  merged.allow_resolution_scale = false;
  return merged;
}

function frozenLadder(raw, quality) {
  const iw = Math.max(160, Number(quality.infer_width) || 640);
  const ih = Math.max(120, Number(quality.infer_height) || 360);
  if (quality.freeze_camera_ladder !== false) {
    return [{ width: iw, height: ih }];
  }
  return raw?.camera_ladder || DEFAULT_PROFILES.camera_ladder;
}

/**
 * Resolve active runtime settings for the browser pipeline.
 * @param {'auto'|'high'|'medium'|'low'} [mode]
 * @param {{ deferSlowdown?: boolean }} [opts] skip sync CPU probe (better INP)
 */
export async function resolveBrowserPerformance(mode = "auto", opts = {}) {
  const raw = await loadPerformanceConfig();
  const quality = qualityFromRaw(raw);
  const capability = await detectBrowserCapability(raw);
  const autoRecommended = capability.recommended;
  const forced =
    mode === "high" || mode === "medium" || mode === "low" ? mode : null;
  let name = forced || capability.recommended;
  const deferSlowdown = opts.deferSlowdown === true;
  let slowdown = 1;
  if (!deferSlowdown) {
    slowdown = estimateCpuSlowdownFactor();
    // Extreme throttle → demote only in auto mode (manual override always wins)
    if (!forced && slowdown >= 6) {
      name = "low";
      capability.recommended = "low";
      capability.detail = `${capability.detail}; cpu~${slowdown.toFixed(1)}x → low`;
    } else if (!forced && slowdown >= 3.5 && name === "high") {
      name = "medium";
      capability.recommended = "medium";
      capability.detail = `${capability.detail}; cpu~${slowdown.toFixed(1)}x → medium`;
    } else if (!forced && slowdown >= 3.5 && name === "medium" && slowdown >= 5) {
      name = "low";
      capability.recommended = "low";
      capability.detail = `${capability.detail}; cpu~${slowdown.toFixed(1)}x → low`;
    }
  }

  capability.autoRecommended = autoRecommended;
  capability.applied = name;
  capability.forced = !!forced;
  capability.cpuSlowdown = slowdown;
  capability.slowdownPending = deferSlowdown && !forced;

  const d = pickProfileDict(raw, name);
  const fpsMax = 25;
  let targetFps = clamp(Number(d.target_fps) || 25, FPS_MIN, fpsMax);

  const ladder = frozenLadder(raw, quality);
  const camW = ladder[0].width;
  const camH = ladder[0].height;
  const shgEvery = Math.max(1, Number(d.shg_every) || 1);
  // Cap at 2 so low devices stay result-matched (tip-hold fills the gap, no laggy stalls)
  const shgEveryMax = Math.min(
    2,
    Math.max(shgEvery, Number(d.shg_every_max) || 2)
  );

  const profile = {
    name,
    label: String(d.label || name),
    targetFps,
    fpsMin: FPS_MIN,
    fpsMax,
    cameraWidth: camW,
    cameraHeight: camH,
    processWidth: camW,
    processHeight: camH,
    inferWidth: camW,
    inferHeight: camH,
    yoloEvery: 1,
    shgEvery,
    yoloEveryMin: 1,
    shgEveryMin: 1,
    yoloEveryMax: 2,
    shgEveryMax,
    yoloImgsz: 640,
    lockYolo: true,
    lockFps: true,
    lockCamera: true,
    freezeCameraLadder: true,
    cameraLadder: ladder,
    cpuSlowdown: slowdown,
    flipInference: String(quality.flip_inference || "adaptive"),
    leftFlipRecheckEvery: Number(quality.left_flip_recheck_every) || 8,
  };

  const ds = {
    ...DEFAULT_PROFILES.dynamic_scaling,
    ...(raw?.dynamic_scaling || {}),
  };

  console.log(
    `[perf] ${profile.name} · target ${profile.targetFps} fps · ` +
      `cam ${profile.cameraWidth}x${profile.cameraHeight} (quality-frozen) · ` +
      `Y/${profile.yoloEvery} S/${profile.shgEvery}··${profile.shgEveryMax} · ` +
      `cpu~${slowdown.toFixed(1)}x · ${capability.detail}`
  );

  return { profile, capability, dynamic: ds, raw, slowdown, quality };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Runtime sparsify / densify of SHG cadence from measured pipe ms.
 * Camera size and YOLO cadence stay locked (same landmark quality).
 * Tip-hold runs every display frame so landmarks stay stuck.
 */
export class BrowserDynamicScaler {
  /**
   * @param {{
   *   yoloEvery: number,
   *   shgEvery: number,
   *   yoloEveryMin: number,
   *   shgEveryMin: number,
   *   yoloEveryMax: number,
   *   shgEveryMax: number,
   *   targetFps: number,
   *   fpsMin: number,
   *   fpsMax: number,
   *   lockCamera?: boolean,
   *   lockFps?: boolean,
   * }} profile
   * @param {any} dynamicCfg
   */
  constructor(profile, dynamicCfg) {
    this.base = { ...profile };
    this.cfg = dynamicCfg || DEFAULT_PROFILES.dynamic_scaling;
    this.yoloEvery = profile.yoloEvery;
    this.shgEvery = profile.shgEvery;
    this.targetFps = profile.targetFps;
    this.frames = 0;
    this.cooldown = 0;
  }

  reset(profile) {
    if (profile) this.base = { ...profile };
    this.yoloEvery = this.base.yoloEvery;
    this.shgEvery = this.base.shgEvery;
    this.targetFps = this.base.targetFps;
    this.frames = 0;
    this.cooldown = 0;
  }

  /**
   * @param {number} pipeMsEma
   * @param {number} [displayFpsEma]
   */
  observe(pipeMsEma, displayFpsEma) {
    if (!this.cfg?.enabled) return;
    this.frames++;
    if (this.cooldown > 0) this.cooldown--;
    if (this.frames < (Number(this.cfg.warmup_frames) || 45)) return;
    if (this.cooldown > 0) return;
    if (!Number.isFinite(pipeMsEma) || pipeMsEma <= 0) return;

    const budget = 1000 / Math.max(1, this.targetFps);
    const dropRatio = Number(this.cfg.drop_below_ratio) || 0.65;
    const recoverRatio = Number(this.cfg.recover_above_ratio) || 0.9;
    const workBudget = Number(this.cfg.work_budget_ratio) || 0.98;
    const recoverWork = Number(this.cfg.recover_work_ratio) || 0.75;

    const overloaded =
      pipeMsEma > budget * workBudget ||
      (Number.isFinite(displayFpsEma) &&
        displayFpsEma > 0 &&
        displayFpsEma < this.targetFps * dropRatio);

    const headroom =
      pipeMsEma < budget * recoverWork &&
      (!Number.isFinite(displayFpsEma) ||
        displayFpsEma <= 0 ||
        displayFpsEma >= this.targetFps * recoverRatio);

    // Always keep YOLO + target FPS + camera fixed; only sparsify SHG under load.
    this.yoloEvery = this.base.yoloEvery;
    this.targetFps = this.base.targetFps;

    if (overloaded) {
      const prevS = this.shgEvery;
      this.shgEvery = Math.min(this.base.shgEveryMax, this.shgEvery + 1);
      if (prevS !== this.shgEvery) {
        this.cooldown = Number(this.cfg.cooldown_frames) || 30;
      }
    } else if (headroom) {
      const prevS = this.shgEvery;
      this.shgEvery = Math.max(this.base.shgEveryMin, this.shgEvery - 1);
      if (prevS !== this.shgEvery) {
        this.cooldown = Number(this.cfg.cooldown_frames) || 30;
      }
    }
  }

  snapshot() {
    return {
      yoloEvery: this.yoloEvery,
      shgEvery: this.shgEvery,
      targetFps: this.targetFps,
      name: this.base.name,
      label: this.base.label,
    };
  }
}

export function parsePerfModeFromUrl() {
  try {
    const q = new URLSearchParams(location.search);
    const m = (q.get("performance") || q.get("perf") || "auto").toLowerCase();
    if (m === "high" || m === "medium" || m === "low" || m === "auto") return m;
  } catch {
    /* ignore */
  }
  return "auto";
}

/** Human-readable working plan for the UI. */
export function describeWorkingPlan(profile, capability, oneEuro) {
  const tier = profile?.name || "medium";
  const lines = [
    `Infer size ${profile.inferWidth || profile.cameraWidth}×${profile.inferHeight || profile.cameraHeight} (fixed)`,
    `Target ${profile.targetFps} FPS · YOLO every ${profile.yoloEvery} · SHG every ${profile.shgEvery} (max ${profile.shgEveryMax})`,
    `LEFT ear: adaptive flip verify on lock + every ${profile.leftFlipRecheckEvery || 8} SHG`,
    `Tip-hold between SHG frames (smooth UI, no wait on WASM)`,
    `Workers: YOLO + SHGNet off main thread`,
  ];
  if (oneEuro) {
    lines.push(
      `One Euro: min=${oneEuro.min_cutoff} β=${oneEuro.beta} d=${oneEuro.d_cutoff} ` +
        `rest=${oneEuro.rest_speed_px} hold=${oneEuro.rest_hold_frames} step≤${oneEuro.max_step_px}`
    );
  }
  if (tier === "high") {
    lines.push("Plan: responsive One Euro + densest SHG refresh");
  } else if (tier === "medium") {
    lines.push("Plan: jewellery-baseline One Euro + tip-hold under load");
  } else {
    lines.push("Plan: smoother One Euro for tip-hold / SHG every-2 under load");
  }
  return {
    tier,
    label: profile.label || tier,
    detail: capability?.detail || "",
    score: capability?.score,
    cores: capability?.cores,
    mobile: capability?.mobile,
    webgpu: capability?.webgpu,
    lines,
  };
}

/**
 * Full compatibility check → tier → working plan.
 * @param {'auto'|'high'|'medium'|'low'} [mode]
 * @param {{ deferSlowdown?: boolean }} [opts]
 */
export async function runCompatibilityCheck(mode, opts = {}) {
  const forced = mode || parsePerfModeFromUrl();
  const { profile, capability, dynamic, quality, slowdown } =
    await resolveBrowserPerformance(forced, opts);
  const plan = describeWorkingPlan(profile, capability);
  return {
    mode: forced,
    recommended: capability.recommended,
    applied: profile.name,
    profile,
    capability,
    dynamic,
    quality,
    slowdown,
    plan,
  };
}

/**
 * Apply deferred CPU probe result (idle). Returns new tier name if demoted, else null.
 * Landmark quality stays frozen; only throughput / One Euro tier may change.
 * @param {any} capability
 * @returns {{ demoted: boolean, name: string, slowdown: number, capability: any }}
 */
export function applyDeferredCpuSlowdown(capability) {
  const slowdown = estimateCpuSlowdownFactor();
  let name = capability?.applied || capability?.recommended || "medium";
  const forced = !!capability?.forced;
  let demoted = false;
  if (!forced) {
    if (slowdown >= 6 && name !== "low") {
      name = "low";
      demoted = true;
    } else if (slowdown >= 3.5 && name === "high") {
      name = "medium";
      demoted = true;
    } else if (slowdown >= 5 && name === "medium") {
      name = "low";
      demoted = true;
    }
  }
  const next = { ...(capability || {}) };
  next.cpuSlowdown = slowdown;
  next.slowdownPending = false;
  next.applied = name;
  next.recommended = name;
  if (demoted) {
    next.detail = `${next.detail || ""}; idle cpu~${slowdown.toFixed(1)}x → ${name}`;
  } else {
    next.detail = `${next.detail || ""}; idle cpu~${slowdown.toFixed(1)}x`;
  }
  return { demoted, name, slowdown, capability: next };
}

/**
 * Cheap CPU throttle probe (DevTools 6x / weak phones).
 * Warm once, measure twice, take the better (lower) factor to reduce noise.
 * Returns ~1 on normal desktop, ~4–8 when heavily throttled.
 */
export function estimateCpuSlowdownFactor() {
  const iters = 1_200_000;
  const runOnce = () => {
    const t0 = performance.now();
    let x = 0;
    for (let i = 0; i < iters; i++) x = (x + i) | 0;
    const ms = Math.max(0.1, performance.now() - t0);
    if (x === -1) console.debug(x);
    return ms;
  };
  runOnce(); // warmup
  const ms = Math.min(runOnce(), runOnce());
  const baselineMs = 2.0;
  const factor = ms / baselineMs;
  return Math.max(1, Math.min(12, factor));
}

/** Prefer smaller capture when CPU is throttled — disabled when quality-frozen. */
export function pickCameraForSlowdown(baseW, baseH, slowdown, ladder) {
  const steps =
    ladder && ladder.length ? ladder : DEFAULT_PROFILES.camera_ladder;
  // Single-rung / frozen ladder → always canonical size
  if (steps.length <= 1) {
    const s = steps[0];
    return { width: s.width, height: s.height, index: 0, slowdown };
  }
  let idx = 0;
  for (let i = 0; i < steps.length; i++) {
    if (steps[i].width <= baseW && steps[i].height <= baseH) {
      idx = i;
      break;
    }
  }
  if (slowdown >= 5) idx = Math.min(steps.length - 1, idx + 2);
  else if (slowdown >= 3) idx = Math.min(steps.length - 1, idx + 1);
  const s = steps[idx];
  return { width: s.width, height: s.height, index: idx, slowdown };
}

/**
 * Runtime camera downscale when WASM pipe is heavy.
 * No-op when quality freeze locks the camera ladder to one size.
 */
export class CameraSizeAdapter {
  constructor(ladder, startW, startH) {
    this.ladder =
      ladder && ladder.length ? ladder : DEFAULT_PROFILES.camera_ladder;
    this.frozen = this.ladder.length <= 1;
    this.index = this._indexNear(startW, startH);
    this.width = this.ladder[this.index].width;
    this.height = this.ladder[this.index].height;
    this.cooldown = 0;
    this.frames = 0;
  }

  _indexNear(w, h) {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < this.ladder.length; i++) {
      const d =
        Math.abs(this.ladder[i].width - w) +
        Math.abs(this.ladder[i].height - h);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  /**
   * @param {number} heavyMs last YOLO/SHG ms
   * @param {number} targetFps
   * @returns {{changed:boolean,width:number,height:number}}
   */
  observe(heavyMs, targetFps) {
    if (this.frozen) {
      return { changed: false, width: this.width, height: this.height };
    }
    this.frames++;
    if (this.cooldown > 0) this.cooldown--;
    if (this.frames < 4) {
      return { changed: false, width: this.width, height: this.height };
    }
    if (this.cooldown > 0) {
      return { changed: false, width: this.width, height: this.height };
    }
    if (!Number.isFinite(heavyMs) || heavyMs <= 0) {
      return { changed: false, width: this.width, height: this.height };
    }
    const budget = 1000 / Math.max(1, targetFps);
    if (heavyMs > budget * 3.0 && this.index < this.ladder.length - 1) {
      this.index += 1;
      this.width = this.ladder[this.index].width;
      this.height = this.ladder[this.index].height;
      this.cooldown = 40;
      return { changed: true, width: this.width, height: this.height };
    }
    if (heavyMs < budget * 1.1 && this.index > 0) {
      this.index -= 1;
      this.width = this.ladder[this.index].width;
      this.height = this.ladder[this.index].height;
      this.cooldown = 80;
      return { changed: true, width: this.width, height: this.height };
    }
    return { changed: false, width: this.width, height: this.height };
  }
}
