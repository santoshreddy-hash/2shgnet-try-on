/**
 * Browser live ear landmarks — aligned with desktop app.py:
 *
 *   webcam → mirror → YOLO ONNX tip → tip-centered full-ear crop
 *         → 2-SHGNet ONNX (flip LEFT; refine) → One Euro → overlay
 *
 * Quality (infer size / flip / gates) is frozen across devices; only FPS +
 * SHG cadence adapt. Tip-hold fills skipped SHG frames.
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";
import { OneEuroLandmarks } from "./one_euro.js";
import { YoloPoseBrowser } from "./yolo_pose.js";
import { canvasRgbaToBgrChw, heatmapsToPointsSoft } from "./preprocess.js";
import {
  BrowserDynamicScaler,
  parsePerfModeFromUrl,
  CameraSizeAdapter,
  runCompatibilityCheck,
  describeWorkingPlan,
  applyDeferredCpuSlowdown,
  resolveBrowserPerformance,
} from "./performance.js";
import { yieldToPaint, yieldForInput, runWhenIdle } from "./yield.js";
import { OrtInferenceWorker } from "./ort_worker_client.js";
import {
  CROP_PAD,
  REFINE_PAD,
  YOLO_BOX_CONF,
  MIN_SHG_SCORE,
  PIERCING_INDEX,
  pinnaHeight,
  isSideProfile,
  landmarksOk,
  tipCropCenter,
  rescueCropCenter,
} from "./ear_geometry.js";

const SHGNET_URL = "/models/shgnet/SHGNet-56.onnx";
const YOLO_URL = "/models/yolo/yolo26n-pose.onnx";
const WASM_PATH = "/vendor/onnxruntime-web/dist/";

// Sparse inference — set from performance profile (device adaptive)
let YOLO_EVERY = 1;
let SHG_EVERY = 1;
let YOLO_EVERY_MIN = 1;
let SHG_EVERY_MIN = 1;
const YOLO_EVERY_MAX = 2;
const SHG_EVERY_MAX = 2;
const YOLO_IMGSZ = 640;
const YOLO_CONF = YOLO_BOX_CONF;
let CAM_FPS_MIN = 20;
let CAM_FPS_MAX = 25;
let CAM_FPS_DEFAULT = 25;
let DT_FALLBACK = 1 / CAM_FPS_DEFAULT;
/** Mirror selfie preview — only when using front/user camera */
let mirrorFeed = true;
const NUM_LANDMARKS = 56;
/** Capture size from active performance profile. */
let CAM_WIDTH = 640;
let CAM_HEIGHT = 360;
let activeProfileName = "auto";
let activeProfileLabel = "";
/** @type {BrowserDynamicScaler | null} */
let perfScaler = null;
/** @type {CameraSizeAdapter | null} */
let camAdapter = null;
/** Wall-clock gate so heavy WASM work cannot back-to-back fill the pipe. */
let lastInferDoneTs = 0;
let lastHeavyMs = 0;
let camResizeInFlight = false;
/** Smooth-camera mode for 6x / weak CPUs — paint never blocked by SAD/WASM prep. */
let smoothMode = false;
let cpuSlowdown = 1;
let tipVelX = 0;
let tipVelY = 0;
let lastTipTs = 0;
let lastInferStartTs = 0;
let hudEvery = 1;
/** YOLO ONNX is fixed 1×3×640×640 — never change at runtime. */
const YOLO_IMGSZ_RUNTIME = 640;

const fpsSlider = document.getElementById("fpsSlider");
const fpsTargetVal = document.getElementById("fpsTargetVal");

function targetFps() {
  if (perfScaler) return perfScaler.targetFps;
  const v = Number(fpsSlider?.value);
  const raw = Number.isFinite(v) ? v : CAM_FPS_DEFAULT;
  return Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, Math.round(raw)));
}

/** Clamp any FPS reading into the allowed live band. */
function clampFps(v) {
  if (!Number.isFinite(v) || v <= 0) return 0;
  return Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, v));
}

function applyResolvedProfile(profile, dynamic, capability) {
  activeProfileName = profile.name;
  activeProfileLabel = profile.label;
  CAM_FPS_MIN = profile.fpsMin;
  CAM_FPS_MAX = profile.fpsMax;
  CAM_FPS_DEFAULT = profile.targetFps;
  DT_FALLBACK = 1 / CAM_FPS_DEFAULT;
  CAM_WIDTH = profile.cameraWidth;
  CAM_HEIGHT = profile.cameraHeight;
  YOLO_EVERY = 1;
  SHG_EVERY = profile.shgEvery;
  YOLO_EVERY_MIN = 1;
  SHG_EVERY_MIN = profile.shgEveryMin;
  LEFT_FLIP_RECHECK_EVERY = Math.max(
    4,
    Number(profile.leftFlipRecheckEvery) || 8
  );
  hudEvery = 1;
  perfScaler = new BrowserDynamicScaler(profile, dynamic);
  camAdapter = new CameraSizeAdapter(
    profile.cameraLadder,
    profile.cameraWidth,
    profile.cameraHeight
  );
  if (fpsSlider) {
    fpsSlider.min = String(CAM_FPS_MIN);
    fpsSlider.max = String(CAM_FPS_MAX);
    fpsSlider.value = String(profile.targetFps);
  }
  if (fpsTargetVal) fpsTargetVal.textContent = String(profile.targetFps);
  console.log(
    `[perf] applied ${profile.name} · ${profile.targetFps} fps · ` +
      `${CAM_WIDTH}x${CAM_HEIGHT} · Y/${YOLO_EVERY} S/${SHG_EVERY} · ` +
      `cpu~${cpuSlowdown.toFixed(1)}x · smooth=${smoothMode}`
  );
}

/** User override from UI select, else URL `?performance=`. */
let selectedPerfMode = parsePerfModeFromUrl();
/** @type {Awaited<ReturnType<typeof runCompatibilityCheck>> | null} */
let lastCompat = null;

const tierPill = document.getElementById("tierPill");
const planTitle = document.getElementById("planTitle");
const planList = document.getElementById("planList");
const deviceDetail = document.getElementById("deviceDetail");
const tierSelect = document.getElementById("tierSelect");

function updateDeviceCard(compat) {
  if (!compat?.plan) return;
  const { plan, recommended, applied, mode } = compat;
  const tier = applied || plan.tier;
  if (tierPill) {
    tierPill.textContent = tier;
    tierPill.className = `tier-pill ${tier}`;
  }
  if (planTitle) {
    const autoRec = compat.capability?.autoRecommended || recommended;
    if (mode !== "auto") {
      planTitle.textContent = `${plan.label} (manual ${mode}) · auto would be ${autoRec}`;
    } else if (tier !== autoRec) {
      planTitle.textContent = `${plan.label} · auto ${autoRec} → applied ${tier}`;
    } else {
      planTitle.textContent = `${plan.label} · auto: ${autoRec}`;
    }
  }
  if (planList) {
    planList.innerHTML = "";
    for (const line of plan.lines) {
      const li = document.createElement("li");
      li.textContent = line;
      planList.appendChild(li);
    }
  }
  if (deviceDetail) {
    deviceDetail.textContent = plan.detail || "";
  }
}

/**
 * Device compatibility test → high/medium/low → apply working plan.
 * Called automatically when the user clicks Load models.
 */
async function runDeviceCompatibilityTest(modeOverride) {
  const raw =
    modeOverride ||
    tierSelect?.value ||
    selectedPerfMode ||
    parsePerfModeFromUrl();
  selectedPerfMode =
    raw === "auto" || raw === "high" || raw === "medium" || raw === "low"
      ? raw
      : "auto";

  if (tierPill) {
    tierPill.textContent = "Checking…";
    tierPill.className = "tier-pill checking";
  }
  if (planTitle) planTitle.textContent = "Running device compatibility test…";
  if (planList) {
    planList.innerHTML =
      "<li>Scoring CPU cores, memory, mobile, CPU probe…</li>";
  }
  setStatus("1/2 Device compatibility test…\nCategorizing high / medium / low…");
  await yieldForInput();

  // Defer CPU probe — keeps Load models click responsive (INP)
  const compat = await runCompatibilityCheck(selectedPerfMode, {
    deferSlowdown: true,
  });
  lastCompat = compat;
  applyResolvedProfile(compat.profile, compat.dynamic, compat.capability);
  updateDeviceCard(compat);
  if (tierSelect) tierSelect.disabled = false;

  const { profile, capability } = compat;
  const applied = profile.name;
  const autoRec = capability.autoRecommended || capability.recommended;
  setStatus(
    `1/2 Compatibility OK → ${applied.toUpperCase()} ` +
      `(score ${Number(capability.score || 0).toFixed(0)}, auto=${autoRec})\n` +
      `${capability.detail}\n` +
      `2/2 Loading models + One Euro [${applied}]…`
  );
  await yieldForInput();
  return { profile, capability, compat };
}

/** @deprecated alias — Load models path uses runDeviceCompatibilityTest */
async function initPerformanceProfile(modeOverride) {
  return runDeviceCompatibilityTest(modeOverride);
}

/** Active ORT EP — WASM only (WebGPU on hold) */
const ortEp = "wasm";
let pipeMsEma = 0;
let lastDisplayTs = 0; // last painted display tick (for true FPS)
let lastDrawTs = 0; // throttle clock for target FPS
let rawRel = null; // last SHG shape relative to tip (for skip frames)
const statusEl = document.getElementById("status");
const sizesEl = document.getElementById("sizes");
const loadBtn = document.getElementById("loadModel");
const startCamBtn = document.getElementById("startCam");
const stopCamBtn = document.getElementById("stopCam");
const video = document.getElementById("video");
const canvas = document.getElementById("out");
const ctx = canvas.getContext("2d", { alpha: false });

let shgSession = null;
let yoloPose = null;
/** @type {OrtInferenceWorker | null} */
let ortWorker = null;
let useDedicatedWorker = false;
let stream = null;
let live = false;
let yoloBusy = false;
let shgBusy = false;
let framePipelineBusy = false;
let rafId = 0;
let lastTs = 0;
let fpsEma = 0;
let inferMsEma = 0;
let frameIdx = 0;
let inferTick = 0;
let lastYolo = null;
let side = null;
let tip = null;
let holdTip = null; // YOLO tip used for stick-hold (rawRel is vs this)
let lastYoloTip = null;
let tipSnap = false; // snap One Euro when YOLO tip jumps
let tipPatch = null; // {g, x0, y0} grayscale template around tip
const TIP_PATCH = 15; // odd — larger template sticks better on ear rim
const TIP_SEARCH = 32;
const TIP_COARSE = 2; // px stride for coarse SAD, then refine
let geo = null; // {cx, cy, side}
let rawPts = null;
let lastBox = null;
let firstLock = true;
let overlay = null; // committed {tip, box, landmarks, side} — same frame
/** Cached LEFT flip choice after verified lock (avoids 2× SHG every frame → lag). */
let leftFlipPrefer = true;
let leftFlipShgCount = 0;
let LEFT_FLIP_RECHECK_EVERY = 8;

const snapCanvas = document.createElement("canvas");
const snapCtx = snapCanvas.getContext("2d", { willReadFrequently: true });

// Defaults match one_euro_settings.json / desktop tracking.settings
const ONE_EURO_DEFAULTS = {
  min_cutoff: 3.2,
  beta: 1.1,
  d_cutoff: 1.45,
  rest_speed_px: 6.0,
  rest_hold_frames: 1,
  rest_release_mult: 1.15,
  max_step_px: 110.0,
};
let oneEuroCfg = { ...ONE_EURO_DEFAULTS };
const smoother = new OneEuroLandmarks(
  NUM_LANDMARKS,
  oneEuroCfg.min_cutoff,
  oneEuroCfg.beta,
  oneEuroCfg.d_cutoff,
  oneEuroCfg.rest_speed_px,
  oneEuroCfg.rest_hold_frames,
  oneEuroCfg.rest_release_mult
);

function oneEuroMaxStep() {
  return Math.max(1, Number(oneEuroCfg.max_step_px) || 42);
}

/** Load One Euro for the active device tier (high|medium|low). */
async function loadOneEuroSettings(tier = activeProfileName) {
  const name =
    tier === "high" || tier === "medium" || tier === "low" ? tier : "medium";
  try {
    const res = await fetch("/one_euro_settings.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const base = { ...ONE_EURO_DEFAULTS };
    for (const k of Object.keys(ONE_EURO_DEFAULTS)) {
      if (data[k] != null) base[k] = data[k];
    }
    const tierCfg =
      (data.profiles && data.profiles[name]) ||
      (data.profiles && data.profiles.medium) ||
      {};
    const picked = {};
    for (const k of Object.keys(ONE_EURO_DEFAULTS)) {
      if (tierCfg[k] != null) picked[k] = tierCfg[k];
    }
    oneEuroCfg = { ...base, ...picked, profile: name };
    smoother.applySettings(oneEuroCfg);
    smoother.reset();
    console.log(
      `[OneEuro] ${name}: min=${oneEuroCfg.min_cutoff} β=${oneEuroCfg.beta} ` +
        `d=${oneEuroCfg.d_cutoff} rest=${oneEuroCfg.rest_speed_px} ` +
        `hold=${oneEuroCfg.rest_hold_frames} step≤${oneEuroCfg.max_step_px}`
    );
  } catch (e) {
    console.warn("[OneEuro] using built-in defaults (JSON load failed)", e);
    oneEuroCfg = { ...ONE_EURO_DEFAULTS, profile: name };
    smoother.applySettings(oneEuroCfg);
  }
}

const cropCanvas = document.createElement("canvas");
cropCanvas.width = 256;
cropCanvas.height = 256;
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

const padCanvas = document.createElement("canvas");
const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });

let lastStatusWriteTs = 0;
let pendingStatusMsg = null;
const STATUS_THROTTLE_MS = 450;

function setStatus(msg, force = false) {
  pendingStatusMsg = msg;
  const now = performance.now();
  if (!force && live && now - lastStatusWriteTs < STATUS_THROTTLE_MS) return;
  statusEl.textContent = msg;
  lastStatusWriteTs = now;
  pendingStatusMsg = null;
}

function flushPendingStatus() {
  if (pendingStatusMsg != null) {
    statusEl.textContent = pendingStatusMsg;
    pendingStatusMsg = null;
    lastStatusWriteTs = performance.now();
  }
}

function updateButtons() {
  startCamBtn.disabled = !(shgSession && yoloPose && !live);
  stopCamBtn.disabled = !live;
  loadBtn.disabled = live;
}

function reportSizes() {
  sizesEl.innerHTML = `
    <strong>Browser assets (no .pth)</strong>
    <table>
      <thead><tr><th>Asset</th><th>Size</th><th>Role</th></tr></thead>
      <tbody>
        <tr><td>SHGNet-56.onnx</td><td>~26 MB</td><td>56 landmarks (55 + piercing)</td></tr>
        <tr><td>yolo26n-pose.onnx</td><td>~12 MB</td><td>ear tip + side</td></tr>
        <tr><td>onnxruntime-web WASM</td><td>~5–15 MB</td><td>inference</td></tr>
        <tr><td><strong>Active EP</strong></td><td><strong>Web Worker</strong></td><td>WASM off UI thread</td></tr>
      </tbody>
    </table>
  `;
}

let ortProxy = true;

/**
 * Always prefer ORT wasm.proxy (dedicated proxy Worker).
 * Keeps ort-wasm-simd-threaded.wasm + session.run off the UI thread.
 */
function configureOrt(useProxy = true) {
  ort.env.wasm.wasmPaths = WASM_PATH;
  const canSAB =
    typeof SharedArrayBuffer !== "undefined" &&
    (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
  // With proxy worker, keep thread count modest to avoid oversubscription
  const cores = navigator.hardwareConcurrency || 2;
  ort.env.wasm.numThreads = canSAB
    ? Math.min(useProxy ? 2 : 4, cores)
    : 1;
  ort.env.wasm.proxy = !!useProxy;
  ortProxy = !!useProxy;
}

async function createSession(url, label) {
  setStatus(`Loading ${label} (wasm${ortProxy ? " · Worker" : ""})…`);
  await yieldForInput();
  return ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

/** After Ready: idle CPU probe may demote tier (One Euro / cadence only). */
function scheduleIdleSlowdownRefine() {
  if (!lastCompat?.capability?.slowdownPending) return;
  runWhenIdle(async () => {
    try {
      const { demoted, name, slowdown, capability } = applyDeferredCpuSlowdown(
        lastCompat.capability
      );
      lastCompat.capability = capability;
      lastCompat.slowdown = slowdown;
      if (!demoted) {
        console.log(`[perf] idle cpu~${slowdown.toFixed(1)}x — keep ${name}`);
        return;
      }
      console.log(`[perf] idle demote → ${name} (cpu~${slowdown.toFixed(1)}x)`);
      const { profile, dynamic } = await resolveBrowserPerformance(name, {
        deferSlowdown: true,
      });
      // Force applied name (resolve with forced mode)
      applyResolvedProfile(profile, dynamic, capability);
      await loadOneEuroSettings(profile.name);
      if (lastCompat) {
        lastCompat.applied = profile.name;
        lastCompat.profile = profile;
        lastCompat.plan = describeWorkingPlan(profile, capability, oneEuroCfg);
        updateDeviceCard(lastCompat);
      }
      if (!live) {
        setStatus(
          `Ready · refined to ${profile.label} after idle CPU check · ` +
            `One Euro [${profile.name}] min=${oneEuroCfg.min_cutoff} β=${oneEuroCfg.beta}`
        );
      }
    } catch (e) {
      console.warn("[perf] idle refine failed", e);
    }
  }, 1500);
}

async function loadModels() {
  // Paint response to the click immediately (INP)
  loadBtn.disabled = true;
  startCamBtn.disabled = true;
  setStatus("Responding… preparing device check");
  await yieldForInput();
  try {
    // Always run compatibility test first, then load with that plan.
    const { profile, capability } = await runDeviceCompatibilityTest(
      tierSelect?.value || selectedPerfMode
    );
    await yieldForInput();
    await loadOneEuroSettings(profile.name);
    await yieldForInput();
    if (lastCompat) {
      lastCompat.plan = describeWorkingPlan(profile, capability, oneEuroCfg);
      updateDeviceCard(lastCompat);
    }
    setStatus(
      `Loading YOLO + SHGNet (${profile.label})…\n` +
        `Device tier: ${capability.recommended}\n` +
        `One Euro [${profile.name}]: min=${oneEuroCfg.min_cutoff} β=${oneEuroCfg.beta} ` +
        `d=${oneEuroCfg.d_cutoff} step≤${oneEuroCfg.max_step_px}`
    );
    await yieldForInput();

    // Primary path: one Worker per model (YOLO | SHGNet)
    useDedicatedWorker = false;
    ortWorker = null;
    try {
      ortWorker = new OrtInferenceWorker({
        yoloWorkerUrl: "/yolo-worker.js",
        shgWorkerUrl: "/shg-worker.js",
      });
      await yieldForInput();
      await ortWorker.init({
        yoloUrl: YOLO_URL,
        shgUrl: SHGNET_URL,
        wasmPaths: WASM_PATH,
        yoloImgsz: YOLO_IMGSZ_RUNTIME,
      });
      useDedicatedWorker = true;
      yoloPose = {
        lastMs: 0,
        async detect(source) {
          const det = await ortWorker.detectYolo(source);
          this.lastMs = ortWorker.lastYoloMs;
          return det;
        },
      };
      shgSession = { worker: true };
      ortProxy = true;
    } catch (workerErr) {
      console.warn("[ort] dual Workers failed; ORT proxy fallback", workerErr);
      if (ortWorker) {
        ortWorker.terminate();
        ortWorker = null;
      }
      await yieldForInput();
      configureOrt(true);
      let shg;
      let yolo;
      try {
        yolo = await createSession(YOLO_URL, "YOLO pose ONNX (~12 MB)");
        await yieldForInput();
        shg = await createSession(SHGNET_URL, "SHGNet ONNX (~26 MB)");
      } catch (firstErr) {
        console.warn("ORT proxy failed; main-thread wasm", firstErr);
        await yieldForInput();
        configureOrt(false);
        yolo = await createSession(YOLO_URL, "YOLO pose ONNX (~12 MB)");
        await yieldForInput();
        shg = await createSession(SHGNET_URL, "SHGNet ONNX (~26 MB)");
      }
      shgSession = shg;
      yoloPose = new YoloPoseBrowser(
        yolo,
        (data, dims) => new ort.Tensor("float32", data, dims),
        YOLO_IMGSZ_RUNTIME,
        YOLO_CONF
      );
      yoloPose.yieldBeforeRun = true;
    }

    await yieldForInput();
    setStatus(
      `Ready · ${profile.label} · direct frame pipeline · ` +
        `EP: ${useDedicatedWorker ? "WebWorker×2 (YOLO|SHG)" : ortProxy ? "wasm+proxy" : "wasm"}\n` +
        `Device: ${capability.detail}\n` +
        `Target ${profile.targetFps} fps · YOLO→crop→SHGNet→One Euro · cam ${profile.cameraWidth}x${profile.cameraHeight}\n` +
        `One Euro [${oneEuroCfg.profile || profile.name}]: ` +
        `min=${oneEuroCfg.min_cutoff} β=${oneEuroCfg.beta} ` +
        `d=${oneEuroCfg.d_cutoff} rest=${oneEuroCfg.rest_speed_px} ` +
        `hold=${oneEuroCfg.rest_hold_frames} step≤${oneEuroCfg.max_step_px}\n` +
        "Click Start live cam — allow camera when prompted."
    );
    updateButtons();
    reportSizes();
    scheduleIdleSlowdownRefine();
  } catch (e) {
    console.error(e);
    setStatus(`Load failed: ${e?.message || e}\nRetry Load models in Chrome/Edge.`);
    loadBtn.disabled = false;
    updateButtons();
  }
}

function sideLabel(side) {
  if (side === "LEFT") return "Left";
  if (side === "RIGHT") return "Right";
  return side || "—";
}

/** Anatomical side → on-screen label when preview is mirrored. */
function displayEarLabel(side) {
  if (!side) return "—";
  if (mirrorFeed) {
    if (side === "LEFT") return "Right";
    if (side === "RIGHT") return "Left";
  }
  return sideLabel(side);
}

function clearEarLock(reason) {
  smoother.reset();
  rawPts = null;
  rawRel = null;
  firstLock = true;
  geo = null;
  overlay = null;
  holdTip = null;
  tip = null;
  lastYoloTip = null;
  tipVelX = 0;
  tipVelY = 0;
  tipSnap = true;
  leftFlipPrefer = true;
  leftFlipShgCount = 0;
  side = null;
  if (reason) console.log(`[ear] clear lock: ${reason}`);
}

function updateGeoFromYolo(yolo, vw, vh) {
  if (!isSideProfile(yolo)) {
    if (rawRel || overlay || holdTip) clearEarLock("not_side_profile");
    else overlay = null;
    return false;
  }
  const tipPt = yolo.tip;
  // Side flip L↔R: never tip-hold landmarks across the face
  if (side && yolo.side && yolo.side !== side) {
    clearEarLock(`side_${side}_to_${yolo.side}`);
  }
  // Huge tip jump (ear→face→other ear) → drop lock instead of sliding
  if (holdTip || lastYoloTip) {
    const prev = holdTip || lastYoloTip;
    const jump = Math.hypot(tipPt.x - prev.x, tipPt.y - prev.y);
    const lim = Math.max(36, (geo?.side || 80) * 0.45);
    if (rawRel && jump > lim) {
      clearEarLock(`tip_jump_${jump.toFixed(0)}px`);
    }
  }

  const pinna = pinnaHeight(yolo, vw, vh);
  const sideLen = pinna * CROP_PAD;
  const { ncx, ncy, mx } = tipCropCenter(tipPt, pinna, yolo, yolo.side, vw);
  if (!geo) {
    geo = { cx: ncx, cy: ncy, side: sideLen };
  } else {
    const a = 0.45;
    geo = {
      cx: (1 - a) * geo.cx + a * ncx,
      cy: (1 - a) * geo.cy + a * ncy,
      side: (1 - a) * geo.side + a * sideLen,
    };
  }
  // Tip must stay well inside the square (desktop rescue)
  const half = geo.side * 0.5;
  if (
    Math.abs(tipPt.x - geo.cx) > half * 0.55 ||
    Math.abs(tipPt.y - geo.cy) > half * 0.55
  ) {
    const r = rescueCropCenter(tipPt, pinna, mx);
    geo = { cx: r.cx, cy: r.cy, side: geo.side };
  }
  side = yolo.side;
  tip = tipPt;
  holdTip = { x: tipPt.x, y: tipPt.y };
  lastYoloTip = { x: tipPt.x, y: tipPt.y };
  tipPatch = null;
  return true;
}

/** Square crop with gray pad (matches Python extract_square_crop). */
function drawSquareCrop(source, cx, cy, sidePx, needFlip) {
  const s = Math.max(32, Math.round(sidePx));
  const ox = Math.round(cx - s * 0.5);
  const oy = Math.round(cy - s * 0.5);
  padCanvas.width = s;
  padCanvas.height = s;
  padCtx.fillStyle = "rgb(114,114,114)";
  padCtx.fillRect(0, 0, s, s);
  // Paste clipped region (do NOT stretch) — matches Python extract_square_crop
  const sw = source.width || source.videoWidth || 0;
  const sh = source.height || source.videoHeight || 0;
  const sx1 = Math.max(0, ox);
  const sy1 = Math.max(0, oy);
  const sx2 = Math.min(sw, ox + s);
  const sy2 = Math.min(sh, oy + s);
  const dx = sx1 - ox;
  const dy = sy1 - oy;
  if (sx2 > sx1 && sy2 > sy1) {
    padCtx.drawImage(
      source,
      sx1,
      sy1,
      sx2 - sx1,
      sy2 - sy1,
      dx,
      dy,
      sx2 - sx1,
      sy2 - sy1
    );
  }

  cropCtx.save();
  if (needFlip) {
    cropCtx.translate(256, 0);
    cropCtx.scale(-1, 1);
  }
  cropCtx.drawImage(padCanvas, 0, 0, s, s, 0, 0, 256, 256);
  cropCtx.restore();
  return { ox, oy, sidePx: s };
}

function cropToChw() {
  const img = cropCtx.getImageData(0, 0, 256, 256);
  return canvasRgbaToBgrChw(img);
}

function cropToTensor() {
  return new ort.Tensor("float32", cropToChw(), [1, 3, 256, 256]);
}

async function runShg(needFlip, ox, oy, sidePx) {
  const t0 = performance.now();
  let pts256;
  if (useDedicatedWorker && ortWorker?.ready) {
    const out = await ortWorker.runShg(cropToChw(), [1, 3, 256, 256]);
    pts256 = heatmapsToPointsSoft({ data: out.data, dims: out.dims }, 256);
    inferMsEma = inferMsEma
      ? inferMsEma * 0.85 + (ortWorker.lastShgMs || 0) * 0.15
      : ortWorker.lastShgMs || performance.now() - t0;
  } else {
    const out = await shgSession.run({
      [shgSession.inputNames[0]]: cropToTensor(),
    });
    const ms = performance.now() - t0;
    inferMsEma = inferMsEma ? inferMsEma * 0.85 + ms * 0.15 : ms;
    pts256 = heatmapsToPointsSoft(out[shgSession.outputNames[0]], 256);
  }
  const score = pts256.score ?? 0;
  if (needFlip) {
    pts256 = pts256.map(([x, y]) => [255 - x, y]);
  }
  const scale = sidePx / 256;
  const pts = pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
  return { pts, score };
}

function hullSquare(pts, pad) {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  const span = Math.max(x1 - x0, y1 - y0, 20);
  return { cx: 0.5 * (x0 + x1), cy: 0.5 * (y0 + y1), side: span * pad };
}

function mirrorX(x, w) {
  return w - 1 - x;
}

/** Letterbox video into quality-frozen CAM_WIDTH×CAM_HEIGHT. */
function letterboxVideoTo(destCtx, destW, destH) {
  const sw = video.videoWidth;
  const sh = video.videoHeight;
  if (!sw || !sh) return false;
  const scale = Math.min(destW / sw, destH / sh);
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));
  const ox = Math.floor((destW - dw) / 2);
  const oy = Math.floor((destH - dh) / 2);
  destCtx.fillStyle = "#000";
  destCtx.fillRect(0, 0, destW, destH);
  if (mirrorFeed) {
    destCtx.save();
    destCtx.translate(destW, 0);
    destCtx.scale(-1, 1);
    destCtx.drawImage(video, ox, oy, dw, dh);
    destCtx.restore();
  } else {
    destCtx.drawImage(video, ox, oy, dw, dh);
  }
  return true;
}

function paintVideo() {
  if (!video.videoWidth || !video.videoHeight) return null;
  const vw = Math.max(160, CAM_WIDTH | 0) || 640;
  const vh = Math.max(120, CAM_HEIGHT | 0) || 360;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  if (!letterboxVideoTo(ctx, vw, vh)) return null;
  return { vw, vh };
}

function drawHud(smoothedDisp, boxDisp, tipDisp, info) {
  if (boxDisp) {
    const [x1, y1, x2, y2] = boxDisp;
    ctx.strokeStyle = "rgba(80, 200, 120, 0.95)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (tipDisp) {
    ctx.fillStyle = "rgb(0,140,255)";
    ctx.beginPath();
    ctx.arc(tipDisp[0], tipDisp[1], 4, 0, Math.PI * 2);
    ctx.fill();
  }
  if (smoothedDisp) {
    ctx.fillStyle = "rgb(0,220,255)";
    for (let i = 0; i < smoothedDisp.length; i++) {
      if (i === PIERCING_INDEX) continue;
      const [x, y] = smoothedDisp[i];
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    if (smoothedDisp[PIERCING_INDEX]) {
      const [px, py] = smoothedDisp[PIERCING_INDEX];
      ctx.fillStyle = "rgb(255,70,90)";
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  if (info) {
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(8, 8, 280, 40);
    ctx.fillStyle = "#f0f0f0";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(info, 16, 26);
    ctx.fillText("YOLO|SHG · separate Workers", 16, 42);
  }
}

async function inferFromSnapshot(source, vw, vh, tipPt, sideNow, geoNow, yolo) {
  if (!geoNow || !tipPt || !sideNow) return null;
  if (!shgSession && !(useDedicatedWorker && ortWorker?.ready)) return null;

  let { cx, cy, side: sideLen } = geoNow;
  const half = sideLen * 0.5;
  // Tip must stay inside crop — rescue like jewellery extract
  if (
    Math.abs(tipPt.x - cx) > half * 0.55 ||
    Math.abs(tipPt.y - cy) > half * 0.55
  ) {
    if (yolo) {
      const pinna = sideLen / Math.max(CROP_PAD, 1e-3);
      const { mx } = tipCropCenter(tipPt, pinna, yolo, sideNow, vw);
      const r = rescueCropCenter(tipPt, pinna, mx);
      cx = r.cx;
      cy = r.cy;
    } else {
      cx = tipPt.x;
      cy = tipPt.y;
    }
    geo = { cx, cy, side: sideLen };
  }

  const preferFlip = sideNow === "LEFT" ? leftFlipPrefer : false;
  let { ox, oy, sidePx } = drawSquareCrop(source, cx, cy, sideLen, preferFlip);
  let box = [
    Math.max(0, Math.round(cx - sidePx * 0.5)),
    Math.max(0, Math.round(cy - sidePx * 0.5)),
    Math.min(vw, Math.round(cx + sidePx * 0.5)),
    Math.min(vh, Math.round(cy + sidePx * 0.5)),
  ];

  let { pts, score } = await runShg(preferFlip, ox, oy, sidePx);
  const lock = firstLock;
  let ok1 = landmarksOk(pts, tipPt, sidePx);
  leftFlipShgCount += 1;
  // LEFT: dual-flip on first lock + periodic recheck / weak score.
  // After lock, skip extra SHG most frames so browser stays lag-free.
  const recheckLeft =
    sideNow === "LEFT" &&
    (lock ||
      !ok1 ||
      score < 0.12 ||
      leftFlipShgCount % LEFT_FLIP_RECHECK_EVERY === 0);
  const shouldFlipCompare =
    recheckLeft || (lock && (!ok1 || score < 0.35)) || score < 0.10;
  if (shouldFlipCompare) {
    drawSquareCrop(source, cx, cy, sideLen, !preferFlip);
    const alt = await runShg(!preferFlip, ox, oy, sidePx);
    const okAlt = landmarksOk(alt.pts, tipPt, sidePx);
    if (
      alt.score > score + 0.02 ||
      (okAlt && !ok1) ||
      (okAlt && alt.score >= score)
    ) {
      pts = alt.pts;
      score = alt.score;
      ok1 = landmarksOk(pts, tipPt, sidePx);
      if (sideNow === "LEFT") leftFlipPrefer = !preferFlip;
    } else if (sideNow === "LEFT") {
      leftFlipPrefer = preferFlip;
    }
  }

  // LEFT: never accept without landmarksOk (no soft escape on low).
  if (sideNow === "LEFT" && !(ok1 && score > MIN_SHG_SCORE)) return null;

  // Optional hull refine — first lock only, skip if Workers (keeps pipe short)
  if (!useDedicatedWorker && lock && ok1 && score > MIN_SHG_SCORE) {
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    for (const [x, y] of pts) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
    const fill = Math.max(x1 - x0, y1 - y0) / Math.max(1, sidePx);
    if (fill < 0.52 || fill > 0.82) {
      const hull = hullSquare(pts, REFINE_PAD);
      if (
        Math.abs(tipPt.x - hull.cx) < hull.side * 0.45 &&
        Math.abs(tipPt.y - hull.cy) < hull.side * 0.45
      ) {
        const c2 = drawSquareCrop(
          source,
          hull.cx,
          hull.cy,
          hull.side,
          preferFlip
        );
        const refined = await runShg(
          preferFlip,
          c2.ox,
          c2.oy,
          c2.sidePx
        );
        if (
          landmarksOk(refined.pts, tipPt, c2.sidePx) &&
          refined.score >= score * 0.9
        ) {
          pts = refined.pts;
          score = refined.score;
          ok1 = true;
        }
      }
    }
  }

  if (!(ok1 && score > MIN_SHG_SCORE)) return null;

  // Expand crop from raw pts (match app.py)
  {
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    for (const [x, y] of pts) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
    const h = sideLen * 0.5;
    const need = Math.max(
      cx - h - x0,
      x1 - (cx + h),
      cy - h - y0,
      y1 - (cy + h),
      0
    );
    if (need > 2) {
      const pinnaEst = sideLen / Math.max(CROP_PAD, 1e-3);
      const maxSide = pinnaEst * CROP_PAD * 1.35;
      sideLen = Math.min(sideLen + 2 * need + sideLen * 0.03, maxSide);
      geo = { cx, cy, side: sideLen };
      sidePx = Math.max(32, Math.round(sideLen));
      box = [
        Math.max(0, Math.round(cx - sidePx * 0.5)),
        Math.max(0, Math.round(cy - sidePx * 0.5)),
        Math.min(vw, Math.round(cx + sidePx * 0.5)),
        Math.min(vh, Math.round(cy + sidePx * 0.5)),
      ];
    }
  }

  return { pts, box, tip: { x: tipPt.x, y: tipPt.y }, side: sideNow };
}

function grabSnap(vw, vh) {
  const tw = vw || CAM_WIDTH || 640;
  const th = vh || CAM_HEIGHT || 360;
  snapCanvas.width = tw;
  snapCanvas.height = th;
  letterboxVideoTo(snapCtx, tw, th);
}

/** YOLO job — own Worker; never waits on SHG. */
async function runYoloJob(vw, vh) {
  if (!yoloPose || yoloBusy) return;
  yoloBusy = true;
  const t0 = performance.now();
  try {
    // Reuse painted canvas (already mirrored) — skip second video draw
    const y = await yoloPose.detect(canvas.width ? canvas : (grabSnap(vw, vh), snapCanvas));
    if (!y) return;
    if (side && y.side !== side) {
      clearEarLock(`yolo_side_${side}_to_${y.side}`);
    }
    lastYolo = y;
    if (!updateGeoFromYolo(y, vw, vh)) {
      setStatus("Turn head — clear SIDE PROFILE of one ear");
      return;
    }
    const prev = holdTip;
    const tipPt = tip;
    const now = performance.now();
    if (lastYoloTip && lastTipTs) {
      const dtt = Math.max(0.016, (now - lastTipTs) / 1000);
      tipVelX = (tipPt.x - lastYoloTip.x) / dtt;
      tipVelY = (tipPt.y - lastYoloTip.y) / dtt;
      const sp = Math.hypot(tipVelX, tipVelY);
      if (sp > 600) {
        tipVelX *= 600 / sp;
        tipVelY *= 600 / sp;
      }
    }
    lastYoloTip = { x: tipPt.x, y: tipPt.y };
    lastTipTs = now;
    // Hard snap to YOLO tip — soft blend made landmarks trail the ear
    holdTip = { x: tipPt.x, y: tipPt.y };
    tip = holdTip;
    tipPatch = null;
    tipSnap = true;
    if (prev && geo && rawRel) {
      geo = {
        cx: geo.cx + (holdTip.x - prev.x),
        cy: geo.cy + (holdTip.y - prev.y),
        side: geo.side,
      };
    }
    if (rawRel && side && geo) {
      applyTipHold(holdTip, side, geo, vw, vh, DT_FALLBACK, true);
      tipSnap = false;
    }
    const ms = performance.now() - t0;
    lastHeavyMs = yoloPose?.lastMs || ms;
    pipeMsEma = pipeMsEma ? pipeMsEma * 0.9 + ms * 0.1 : ms;
    adaptInferenceLoad();
  } finally {
    yoloBusy = false;
    lastInferDoneTs = performance.now();
  }
}

/** SHG job — own Worker; never waits on YOLO. Tip-hold keeps UI smooth meanwhile. */
async function runShgJob(vw, vh, dt) {
  if (shgBusy) return;
  if (!geo || !tip || !side) return;
  if (!shgSession && !(useDedicatedWorker && ortWorker?.ready)) return;
  shgBusy = true;
  const t0 = performance.now();
  try {
    // Reuse painted canvas — avoid grabSnap cost on every SHG
    const src = canvas.width ? canvas : (grabSnap(vw, vh), snapCanvas);
    const tipPt = { x: (holdTip || tip).x, y: (holdTip || tip).y };
    const result = await inferFromSnapshot(
      src,
      vw,
      vh,
      tipPt,
      side,
      geo,
      lastYolo
    );
    if (!result) return;

    // Keep live tip if YOLO moved during SHG; only refresh shape offsets
    const liveTip = holdTip || tipPt;
    const newRel = result.pts.map(([x, y]) => [x - tipPt.x, y - tipPt.y]);
    rawPts = result.pts;
    lastBox = result.box;

    const snap = firstLock;
    const stepPx = oneEuroMaxStep();
    rawRel = smoother.filterOffsets
      ? smoother.filterOffsets(newRel, dt, result.side, { maxStepPx: stepPx, snap })
      : smoother
          .updateRelative(
            newRel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]),
            tipPt,
            dt,
            result.side,
            { maxStepPx: stepPx, snap }
          )
          .map(([x, y]) => [x - tipPt.x, y - tipPt.y]);
    if (snap) firstLock = false;
    smoother.syncRelative(rawRel);

    const landmarks = smoother.compose(liveTip, rawRel);
    overlay = {
      tip: { x: liveTip.x, y: liveTip.y },
      box: result.box,
      landmarks,
      side: result.side,
      pierce: landmarks[PIERCING_INDEX]
        ? { x: landmarks[PIERCING_INDEX][0], y: landmarks[PIERCING_INDEX][1] }
        : liveTip,
    };
    const ms = performance.now() - t0;
    lastHeavyMs = ms;
    pipeMsEma = pipeMsEma ? pipeMsEma * 0.85 + ms * 0.15 : ms;
    adaptInferenceLoad();
  } finally {
    shgBusy = false;
    lastInferDoneTs = performance.now();
  }
}

/** Grab grayscale patch around tip from the painted canvas. */
function captureTipPatch(tx, ty) {
  const x0 = Math.round(tx - (TIP_PATCH - 1) / 2);
  const y0 = Math.round(ty - (TIP_PATCH - 1) / 2);
  if (x0 < 0 || y0 < 0 || x0 + TIP_PATCH > canvas.width || y0 + TIP_PATCH > canvas.height)
    return null;
  const img = ctx.getImageData(x0, y0, TIP_PATCH, TIP_PATCH);
  const g = new Float32Array(TIP_PATCH * TIP_PATCH);
  for (let i = 0, p = 0; i < img.data.length; i += 4, p++) {
    g[p] = 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
  }
  return { g, x0, y0 };
}

/**
 * Track tip every display frame (between YOLO) so landmarks stick to the ear.
 * SAD match — clamped to last YOLO tip so background texture cannot steal the lock.
 */
function tipNearFrameEdge(tx, ty, vw, vh) {
  const e = Math.min(vw, vh) * 0.06;
  return tx < e || ty < e || tx > vw - e || ty > vh - e;
}

function trackTipOnCanvas(vw, vh) {
  if (!holdTip || !tipPatch || !canvas.width) return holdTip;
  const anchor = lastYoloTip || holdTip;
  const maxDrift = Math.min(vw, vh) * 0.08;
  const cx = Math.round(holdTip.x);
  const cy = Math.round(holdTip.y);
  const half = (TIP_PATCH - 1) / 2;
  const x1 = Math.max(0, cx - TIP_SEARCH);
  const y1 = Math.max(0, cy - TIP_SEARCH);
  const x2 = Math.min(canvas.width - TIP_PATCH, cx + TIP_SEARCH);
  const y2 = Math.min(canvas.height - TIP_PATCH, cy + TIP_SEARCH);
  if (x2 <= x1 || y2 <= y1) return holdTip;

  const region = ctx.getImageData(x1, y1, x2 - x1 + TIP_PATCH, y2 - y1 + TIP_PATCH);
  const rw = region.width;
  const tpl = tipPatch.g;
  let best = Infinity;
  let bx = cx;
  let by = cy;
  const limX = x2 - x1;
  const limY = y2 - y1;
  for (let oy = 0; oy <= limY; oy += TIP_COARSE) {
    for (let ox = 0; ox <= limX; ox += TIP_COARSE) {
      let sad = 0;
      for (let py = 0; py < TIP_PATCH; py++) {
        for (let px = 0; px < TIP_PATCH; px++) {
          const i = ((oy + py) * rw + (ox + px)) * 4;
          const gv =
            0.299 * region.data[i] +
            0.587 * region.data[i + 1] +
            0.114 * region.data[i + 2];
          sad += Math.abs(gv - tpl[py * TIP_PATCH + px]);
        }
      }
      if (sad < best) {
        best = sad;
        bx = x1 + ox + half;
        by = y1 + oy + half;
      }
    }
  }
  // Fine refine
  const rx0 = Math.max(0, bx - half - x1 - TIP_COARSE);
  const ry0 = Math.max(0, by - half - y1 - TIP_COARSE);
  const rx1 = Math.min(limX, bx - half - x1 + TIP_COARSE);
  const ry1 = Math.min(limY, by - half - y1 + TIP_COARSE);
  for (let oy = ry0; oy <= ry1; oy++) {
    for (let ox = rx0; ox <= rx1; ox++) {
      let sad = 0;
      for (let py = 0; py < TIP_PATCH; py++) {
        for (let px = 0; px < TIP_PATCH; px++) {
          const i = ((oy + py) * rw + (ox + px)) * 4;
          const gv =
            0.299 * region.data[i] +
            0.587 * region.data[i + 1] +
            0.114 * region.data[i + 2];
          sad += Math.abs(gv - tpl[py * TIP_PATCH + px]);
        }
      }
      if (sad < best) {
        best = sad;
        bx = x1 + ox + half;
        by = y1 + oy + half;
      }
    }
  }

  if (Math.hypot(bx - holdTip.x, by - holdTip.y) > TIP_SEARCH) return holdTip;
  if (Math.hypot(bx - anchor.x, by - anchor.y) > maxDrift) return holdTip;
  if (tipNearFrameEdge(bx, by, vw, vh)) return holdTip;
  const maxSad = TIP_PATCH * TIP_PATCH * 45;
  if (best > maxSad) return holdTip;
  return { x: bx, y: by };
}

/** Rigid tip-lock: landmarks = latestTip + smoothed offsets (One Euro never delays tip). */
function jumpIsLarge(prev, next) {
  if (!prev || !next) return true;
  return Math.hypot(next.x - prev.x, next.y - prev.y) > 40;
}

function applyTipHold(tipPt, sideNow, geoNow, vw, vh, _dt, snap) {
  if (!rawRel || !tipPt || !sideNow || !geoNow) return false;
  if (snap) smoother.syncRelative(rawRel);
  const rigid = smoother.compose
    ? smoother.compose(tipPt, rawRel)
    : rawRel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]);
  const sidePx = geoNow.side;
  overlay = {
    tip: { x: tipPt.x, y: tipPt.y },
    box: [
      Math.max(0, Math.round(geoNow.cx - sidePx * 0.5)),
      Math.max(0, Math.round(geoNow.cy - sidePx * 0.5)),
      Math.min(vw, Math.round(geoNow.cx + sidePx * 0.5)),
      Math.min(vh, Math.round(geoNow.cy + sidePx * 0.5)),
    ],
    landmarks: rigid,
    side: sideNow,
    pierce: rigid[PIERCING_INDEX]
      ? { x: rigid[PIERCING_INDEX][0], y: rigid[PIERCING_INDEX][1] }
      : { x: tipPt.x, y: tipPt.y },
  };
  return true;
}

function noteDisplayFps(ts) {
  if (lastDisplayTs) {
    const inst = 1000 / Math.max(1e-3, ts - lastDisplayTs);
    // EMA then hard-clamp to 20–30 so HUD never reports out of band
    const ema = fpsEma ? fpsEma * 0.85 + inst * 0.15 : inst;
    fpsEma = clampFps(ema);
  } else {
    fpsEma = 0;
  }
  lastDisplayTs = ts;
}

function paintOverlayHud() {
  const tipDisp = overlay?.tip ? [overlay.tip.x, overlay.tip.y] : null;
  const boxDisp = overlay?.box || null;
  const lmDisp = overlay?.landmarks || null;
  const earLabel = displayEarLabel(overlay?.side || side);
  const want = targetFps();
  const shownFps = fpsEma ? clampFps(fpsEma) : 0;
  const epLabel = useDedicatedWorker
    ? "WW×2"
    : ortProxy
      ? `${ortEp}+proxy`
      : ortEp;
  // Canvas HUD every frame (cheap); DOM status throttled for INP
  drawHud(
    lmDisp,
    boxDisp,
    tipDisp,
    `LIVE ${shownFps.toFixed(0)}/${want} fps · ${activeProfileName} · pipe ${pipeMsEma.toFixed(0)} ms · ${epLabel}`
  );
  setStatus(
    `LIVE ${shownFps.toFixed(0)} display FPS · ${activeProfileLabel || activeProfileName} · target ${want} ` +
      `(band ${CAM_FPS_MIN}–${CAM_FPS_MAX}) · pipe ${pipeMsEma.toFixed(0)} ms (SHG ${inferMsEma.toFixed(0)} ms) (${epLabel})\n` +
      `Ear: ${earLabel} · 56 pts ${lmDisp ? "on" : "…"} · YOLO→crop→SHG→One Euro\n` +
      `Note: latest completed frame result is rendered; busy frames are skipped.`
  );
}

/**
 * One serialized frame pipeline: YOLO tip → crop box/crop → SHGNet-56 →
 * One Euro → final overlay. Display painting never waits for this promise.
 */
async function runFramePipeline(vw, vh, dt) {
  if (framePipelineBusy || !yoloPose || (!shgSession && !(useDedicatedWorker && ortWorker?.ready))) return;
  framePipelineBusy = true;
  const t0 = performance.now();
  try {
    // Freeze one camera image for the whole sequence so YOLO, crop, and SHG
    // operate on the same frame rather than mixing frames while workers run.
    grabSnap(vw, vh);
    const source = snapCanvas;
    const yolo = await yoloPose.detect(source);
    if (!yolo || !updateGeoFromYolo(yolo, vw, vh)) {
      overlay = null;
      setStatus("No clear side-profile ear detected");
      return;
    }

    lastYolo = yolo;
    const tipPt = { x: yolo.tip.x, y: yolo.tip.y };
    holdTip = tipPt;
    tip = tipPt;
    lastYoloTip = tipPt;
    side = yolo.side;
    // Tip-hold only after verified lock — never during L↔R transition
    if (rawRel && !firstLock && geo) {
      applyTipHold(tipPt, yolo.side, geo, vw, vh, dt, false);
    } else {
      overlay = null;
    }

    const result = await inferFromSnapshot(
      source,
      vw,
      vh,
      tipPt,
      yolo.side,
      geo,
      yolo
    );
    if (!result) {
      if (!rawRel) {
        overlay = null;
        setStatus("Ear detected · waiting for valid SHGNet landmarks");
      }
      return;
    }

    const snap = firstLock;
    const liveTip = holdTip || tipPt;
    const newRel = result.pts.map(([x, y]) => [x - tipPt.x, y - tipPt.y]);
    // One Euro on offsets only — never filter tip
    const stepPx = oneEuroMaxStep();
    const smoothRel = smoother.filterOffsets
      ? smoother.filterOffsets(newRel, dt, yolo.side, { maxStepPx: stepPx, snap })
      : smoother
          .updateRelative(
            newRel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]),
            tipPt,
            dt,
            yolo.side,
            { maxStepPx: stepPx, snap }
          )
          .map(([x, y]) => [x - tipPt.x, y - tipPt.y]);
    rawRel = smoothRel;
    smoother.syncRelative(rawRel);
    firstLock = false;
    rawPts = result.pts;

    const landmarks = smoother.compose(liveTip, rawRel);
    overlay = {
      tip: { x: liveTip.x, y: liveTip.y },
      box: result.box,
      landmarks,
      side: yolo.side,
      pierce: landmarks[PIERCING_INDEX]
        ? { x: landmarks[PIERCING_INDEX][0], y: landmarks[PIERCING_INDEX][1] }
        : liveTip,
    };
    lastHeavyMs = yoloPose.lastMs || performance.now() - t0;
    pipeMsEma = pipeMsEma ? pipeMsEma * 0.85 + (performance.now() - t0) * 0.15 : performance.now() - t0;
    adaptInferenceLoad();
  } finally {
    framePipelineBusy = false;
    lastInferDoneTs = performance.now();
  }
}

function onDisplayTick(ts) {
  if (!live) return;

  const frame = paintVideo();
  if (!frame) {
    noteDisplayFps(ts);
    paintOverlayHud();
    return;
  }
  const { vw, vh } = frame;
  const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : DT_FALLBACK;
  lastTs = ts;
  frameIdx++;

  noteDisplayFps(ts);

  // Tip-hold only while a verified ear lock exists — never slide L↔R across face.
  const earLocked = !!(rawRel && !firstLock && side && geo && (holdTip || tip));
  if (earLocked) {
    const prev = holdTip || tip;
    let tracked = prev;
    // Light velocity coast between YOLO updates so tip stays live at 20–30 FPS
    if (lastYoloTip && prev) {
      tipVelX *= 0.88;
      tipVelY *= 0.88;
      const pull = 0.45;
      let stepX =
        tipVelX * dt + (lastYoloTip.x - prev.x) * pull * Math.min(1, dt * 30);
      let stepY =
        tipVelY * dt + (lastYoloTip.y - prev.y) * pull * Math.min(1, dt * 30);
      const step = Math.hypot(stepX, stepY);
      const maxStep = Math.min(22, (geo?.side || 80) * 0.12);
      if (step > maxStep && step > 1e-6) {
        stepX *= maxStep / step;
        stepY *= maxStep / step;
      }
      tracked = { x: prev.x + stepX, y: prev.y + stepY };
      // Abort tip-hold if coasting would leap (transition / bad track)
      if (Math.hypot(tracked.x - prev.x, tracked.y - prev.y) > maxStep * 1.5) {
        clearEarLock("tip_coast_abort");
      } else {
        geo = {
          cx: geo.cx + (tracked.x - prev.x),
          cy: geo.cy + (tracked.y - prev.y),
          side: geo.side,
        };
        holdTip = tracked;
        tip = tracked;
        applyTipHold(holdTip || tip, side, geo, vw, vh, dt, tipSnap);
        tipSnap = false;
      }
    } else {
      applyTipHold(holdTip || tip, side, geo, vw, vh, dt, tipSnap);
      tipSnap = false;
    }
  } else if (overlay && (!rawRel || firstLock)) {
    overlay = null;
  }

  // Serialized YOLO → crop → SHGNet on cadence. Display never waits.
  const modelsReady =
    shgSession || (useDedicatedWorker && ortWorker?.ready);
  if (modelsReady) {
    inferTick++;
    const shgEvery = Math.max(1, SHG_EVERY || 1);
    const yoloEvery = Math.max(1, YOLO_EVERY || 1);
    const needLock = !rawRel || firstLock;
    // Offset SHG phase so YOLO+SHG don't starve on the same tick under load
    const runYolo = needLock || inferTick % yoloEvery === 0;
    const runShg =
      needLock || (inferTick + Math.floor(shgEvery / 2)) % shgEvery === 0;
    if (runYolo || runShg) {
      runFramePipeline(vw, vh, dt).catch((e) =>
        setStatus(`Frame pipeline error: ${e?.message || e}`)
      );
    }
  }

  paintOverlayHud();
}

function loopLive(ts) {
  if (!live) return;
  rafId = requestAnimationFrame(loopLive);

  // Hard display throttle: never paint above CAM_FPS_MAX, never target below CAM_FPS_MIN
  const want = targetFps();
  const interval = 1000 / want;
  if (lastDrawTs && ts - lastDrawTs < interval - 0.5) return;
  // Keep phase stable; also guard against catch-up bursts > max FPS
  if (lastDrawTs) {
    const elapsed = ts - lastDrawTs;
    const steps = Math.max(1, Math.floor(elapsed / interval));
    lastDrawTs += steps * interval;
    // If we fell far behind, resync so we don't burst
    if (ts - lastDrawTs > interval) lastDrawTs = ts;
  } else {
    lastDrawTs = ts;
  }

  onDisplayTick(ts);
}

async function applyCamSizeConstraint(width, height) {
  if (!stream || camResizeInFlight) return;
  const track = stream.getVideoTracks()[0];
  if (!track) return;
  camResizeInFlight = true;
  try {
    await track.applyConstraints({
      width: { ideal: width },
      height: { ideal: height },
    });
    CAM_WIDTH = width;
    CAM_HEIGHT = height;
    tipPatch = null;
    console.log(`[cam] resized → ${width}x${height} (FPS/Y/2 unchanged)`);
  } catch (e) {
    console.warn("[cam] resize failed", e);
  } finally {
    camResizeInFlight = false;
  }
}

function adaptInferenceLoad() {
  // Quality freeze: camera + YOLO locked. Under load, sparsify SHG only.
  if (perfScaler) {
    perfScaler.observe(pipeMsEma, fpsEma);
    YOLO_EVERY = Math.max(1, perfScaler.yoloEvery || perfScaler.base?.yoloEvery || 1);
    SHG_EVERY = Math.max(1, perfScaler.shgEvery || 1);
    if (fpsSlider) fpsSlider.value = String(perfScaler.targetFps);
    if (fpsTargetVal) fpsTargetVal.textContent = String(perfScaler.targetFps);
  }
  // CameraSizeAdapter is a no-op when ladder is frozen to one size.
  if (camAdapter && !camAdapter.frozen && lastHeavyMs > 0) {
    const adj = camAdapter.observe(lastHeavyMs, targetFps());
    if (adj.changed) {
      applyCamSizeConstraint(adj.width, adj.height);
    }
  }
}

function scheduleVideoFrames() {
  // Full rAF display clock — independent of slow WASM inference
  if (!rafId) rafId = requestAnimationFrame(loopLive);
}

async function applyCamFpsConstraint(idealFps) {
  if (!stream) return;
  const track = stream.getVideoTracks()[0];
  if (!track) return;
  const fps = clampFps(idealFps) || CAM_FPS_DEFAULT;
  try {
    await track.applyConstraints({
      frameRate: { ideal: fps, min: CAM_FPS_MIN, max: CAM_FPS_MAX },
    });
  } catch (_) {
    try {
      await track.applyConstraints({
        frameRate: { ideal: fps, max: CAM_FPS_MAX },
      });
    } catch (_) {
      /* display throttle still enforces band */
    }
  }
}

async function resolveFrontCameraConstraints(wantFps) {
  const sizeFps = {
    width: { ideal: CAM_WIDTH },
    height: { ideal: CAM_HEIGHT },
    frameRate: { ideal: wantFps, max: CAM_FPS_MAX },
  };
  // Prefer front/user camera (never environment/back)
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { exact: "user" }, ...sizeFps },
    });
    return stream;
  } catch (_) {
    /* try softer constraints */
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "user" }, ...sizeFps },
    });
    return stream;
  } catch (_) {
    /* enumerate */
  }
  try {
    // Labels available after a prior permission grant
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    const front =
      cams.find((d) => /front|user|face|selfie/i.test(d.label || "")) ||
      cams.find((d) => !/back|rear|environment|world|ultra/i.test(d.label || ""));
    if (front?.deviceId) {
      return navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { deviceId: { exact: front.deviceId }, ...sizeFps },
      });
    }
  } catch (_) {
    /* fall through */
  }
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", ...sizeFps },
  });
}

async function startCamera() {
  if (live) return;
  if (!shgSession || !yoloPose) {
    setStatus("Models not ready — wait for Ready or click Load models.", true);
    return;
  }
  startCamBtn.disabled = true;
  setStatus("Requesting front camera…", true);
  await yieldForInput();
  const wantFps = targetFps();
  try {
    stream = await resolveFrontCameraConstraints(wantFps);
  } catch (e) {
    setStatus(`Camera error: ${e?.name || e}`, true);
    updateButtons();
    return;
  }
  video.srcObject = stream;
  video.style.display = "none";
  await video.play();
  await applyCamFpsConstraint(wantFps);
  try {
    const track = stream.getVideoTracks()[0];
    const s = track.getSettings?.() || {};
    // Mirror only front/user selfie cam — never back/environment
    const facing = String(s.facingMode || "user").toLowerCase();
    mirrorFeed = facing !== "environment";
    console.log(
      `[cam] feed ${s.width}x${s.height} @ ${s.frameRate} fps facing=${facing || "?"} mirror=${mirrorFeed}`
    );
    if (facing === "environment") {
      setStatus("Back camera selected — retrying front…");
      for (const t of stream.getTracks()) t.stop();
      stream = await resolveFrontCameraConstraints(wantFps);
      video.srcObject = stream;
      await video.play();
      const s2 = stream.getVideoTracks()[0]?.getSettings?.() || {};
      mirrorFeed = String(s2.facingMode || "user").toLowerCase() !== "environment";
    }
  } catch (_) {
    mirrorFeed = true;
  }
  live = true;
  overlay = null;
  rawPts = null;
  lastYolo = null;
  lastBox = null;
  geo = null;
  tip = null;
  holdTip = null;
  lastYoloTip = null;
  tipSnap = false;
  tipPatch = null;
  side = null;
  firstLock = true;
  leftFlipPrefer = true;
  leftFlipShgCount = 0;
  frameIdx = 0;
  inferTick = 0;
  smoother.reset();
  lastTs = 0;
  lastDisplayTs = 0;
  lastDrawTs = 0;
  pipeMsEma = 0;
  rawRel = null;
  fpsEma = 0;
  lastInferDoneTs = 0;
  lastHeavyMs = 0;
  lastInferStartTs = 0;
  tipVelX = 0;
  tipVelY = 0;
  lastTipTs = 0;
  yoloBusy = false;
  shgBusy = false;
  if (perfScaler) perfScaler.reset();
  updateButtons();
  rafId = 0;
  scheduleVideoFrames();
}

function stopCamera() {
  live = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;
  flushPendingStatus();
  updateButtons();
  setStatus("Camera stopped.", true);
}

loadBtn.addEventListener("click", () => {
  loadModels().catch((e) => setStatus(String(e), true));
});
startCamBtn.addEventListener("click", () =>
  startCamera().catch((e) => {
    setStatus(String(e), true);
    updateButtons();
  })
);
stopCamBtn.addEventListener("click", () => stopCamera());
window.addEventListener("beforeunload", () => stopCamera());

if (fpsSlider && fpsTargetVal) {
  fpsTargetVal.textContent = String(targetFps());
  // Label only on input (cheap); apply cam constraint on change (INP)
  fpsSlider.addEventListener("input", () => {
    const fps = Math.max(
      CAM_FPS_MIN,
      Math.min(CAM_FPS_MAX, Math.round(Number(fpsSlider.value) || CAM_FPS_DEFAULT))
    );
    fpsTargetVal.textContent = String(fps);
  });
  fpsSlider.addEventListener("change", () => {
    const fps = Math.max(
      CAM_FPS_MIN,
      Math.min(CAM_FPS_MAX, Math.round(Number(fpsSlider.value) || CAM_FPS_DEFAULT))
    );
    fpsTargetVal.textContent = String(fps);
    if (perfScaler) perfScaler.targetFps = fps;
    lastDrawTs = 0;
    if (live) {
      yieldForInput().then(() => applyCamFpsConstraint(fps));
    }
  });
}

reportSizes();
updateButtons();
if (tierSelect) {
  tierSelect.dataset.bound = "1";
  const urlMode = parsePerfModeFromUrl();
  tierSelect.value = urlMode;
  selectedPerfMode = urlMode;
  tierSelect.disabled = false;
  tierSelect.addEventListener("change", () => {
    selectedPerfMode = tierSelect.value || "auto";
    if (planTitle) {
      planTitle.textContent =
        selectedPerfMode === "auto"
          ? "Will auto-detect on Load models"
          : `Will use ${selectedPerfMode} plan on Load models`;
    }
  });
}
if (tierPill) {
  tierPill.textContent = "pending";
  tierPill.className = "tier-pill checking";
}
if (planTitle) planTitle.textContent = "Click Load models to test this device";
if (planList) {
  planList.innerHTML =
    "<li>Compatibility test runs when you click Load models</li>" +
    "<li>Then categorizes high / medium / low and applies that plan</li>";
}
setStatus(
  "Click Load models — runs device compatibility test, picks high/medium/low, then loads YOLO + SHGNet."
);
// Do NOT auto-load or auto-check: check is tied to Load models click.
