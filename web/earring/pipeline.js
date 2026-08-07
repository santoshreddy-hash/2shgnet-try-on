/**
 * Jewellery-style live pipeline (mirrors desktop / web/infer.js):
 *   webcam → mirror → YOLO tip → tip-centered ear crop
 *         → SHGNet-56 (LEFT flip) → tip-relative One Euro → overlay
 *
 * Paint every rAF; inference is async (never blocks display).
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";
import { OneEuroLandmarks } from "../one_euro.js";
import { YoloPoseBrowser } from "../yolo_pose.js";
import { canvasRgbaToBgrChw, heatmapsToPointsSoft } from "../preprocess.js";
import {
  resolveBrowserPerformance,
  BrowserDynamicScaler,
  parsePerfModeFromUrl,
  CameraSizeAdapter,
} from "../performance.js";
import { yieldToPaint } from "../yield.js";
import { OrtInferenceWorker } from "../ort_worker_client.js";
import { TipLkTracker } from "../tip_lk.js";
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
} from "../ear_geometry.js";

const SHGNET_URL = "/models/shgnet/SHGNet-56.onnx";
const YOLO_URL = "/models/yolo/yolo26n-pose.onnx";
const WASM_PATH = "/vendor/onnxruntime-web/dist/";

const YOLO_IMGSZ = 640;
const YOLO_CONF = YOLO_BOX_CONF;
const CAM_FPS_MIN_DEFAULT = 20;
const CAM_FPS_MAX_DEFAULT = 25;
const CAM_FPS_DEFAULT = 25;
const DT_FALLBACK = 1 / CAM_FPS_DEFAULT;
/** Mirror selfie preview — only for front/user camera */
let MIRROR_DEFAULT = true;
const NUM_LANDMARKS = 56;
export { PIERCING_INDEX };

const ONE_EURO_DEFAULTS = {
  min_cutoff: 2.2,
  beta: 1.25,
  d_cutoff: 1.4,
  rest_speed_px: 4.0,
  rest_hold_frames: 2,
  rest_release_mult: 1.3,
  max_step_px: 40.0,
};

export class EarTryOnPipeline {
  /**
   * @param {{ video: HTMLVideoElement, canvas: HTMLCanvasElement, onStatus?: (s:string)=>void }} opts
   */
  constructor(opts) {
    this.video = opts.video;
    this.canvas = opts.canvas;
    this.ctx = this.canvas.getContext("2d", { alpha: false });
    this.onStatus = opts.onStatus || (() => {});

    this.shgSession = null;
    this.yoloPose = null;
    /** @type {OrtInferenceWorker | null} */
    this.ortWorker = null;
    this.useDedicatedWorker = false;
    this.stream = null;
    this.live = false;
    this.processing = false;
    this.rafId = 0;
    this.ortProxy = false;
    this.targetFps = CAM_FPS_DEFAULT;
    this.fpsMin = CAM_FPS_MIN_DEFAULT;
    this.fpsMax = CAM_FPS_MAX_DEFAULT;
    this.camWidth = 640;
    this.camHeight = 360;
    this.yoloEvery = 2;
    this.shgEvery = 1;
    this.profileName = "auto";
    this.profileLabel = "";
    /** @type {BrowserDynamicScaler | null} */
    this.perfScaler = null;
    this.capabilityDetail = "";
    this.lastInferDoneTs = 0;
    this.lastHeavyMs = 0;
    /** @type {CameraSizeAdapter | null} */
    this.camAdapter = null;
    this.camResizeInFlight = false;
    this.smoothMode = false;
    this.cpuSlowdown = 1;
    this.mirrorFeed = MIRROR_DEFAULT;
    this.tipVelX = 0;
    this.tipVelY = 0;
    this.tipLk = new TipLkTracker();
    this.lastTipTs = 0;
    this.lastInferStartTs = 0;
    this.yoloImgsz = 640;

    this.oneEuroCfg = { ...ONE_EURO_DEFAULTS };
    this.smoother = new OneEuroLandmarks(
      NUM_LANDMARKS,
      this.oneEuroCfg.min_cutoff,
      this.oneEuroCfg.beta,
      this.oneEuroCfg.d_cutoff,
      this.oneEuroCfg.rest_speed_px,
      this.oneEuroCfg.rest_hold_frames,
      this.oneEuroCfg.rest_release_mult
    );

    this.snapCanvas = document.createElement("canvas");
    this.snapCtx = this.snapCanvas.getContext("2d", { willReadFrequently: true });
    this.cropCanvas = document.createElement("canvas");
    this.cropCanvas.width = 256;
    this.cropCanvas.height = 256;
    this.cropCtx = this.cropCanvas.getContext("2d", { willReadFrequently: true });
    this.padCanvas = document.createElement("canvas");
    this.padCtx = this.padCanvas.getContext("2d", { willReadFrequently: true });

    this.resetTracking();
    /** @type {null | { tip, box, landmarks, side, pierce }} */
    this.overlay = null;
    this.onOverlay = null; // (overlay, meta) => void
  }

  resetTracking() {
    this.lastTs = 0;
    this.lastDisplayTs = 0;
    this.lastDrawTs = 0;
    this.fpsEma = 0;
    this.pipeMsEma = 0;
    this.inferMsEma = 0;
    this.frameIdx = 0;
    this.inferTick = 0;
    this.lastYolo = null;
    this.side = null;
    this.tip = null;
    this.holdTip = null;
    this.tipSnap = false;
    this.tipPatch = null;
    this.lastYoloTip = null;
    this.TIP_PATCH = 15;
    this.TIP_SEARCH = 32;
    this.TIP_COARSE = 2;
    this.geo = null;
    this.rawRel = null;
    this.firstLock = true;
    this.overlay = null;
    this.tipLk?.reset();
    this.smoother.reset();
  }

  oneEuroMaxStep() {
    const v = Number(this.oneEuroCfg.max_step_px);
    return Number.isFinite(v) ? Math.max(0, v) : 0;
  }

  async loadOneEuroSettings(tier = this.profileName) {
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
      this.oneEuroCfg = { ...base, ...picked, profile: name };
      this.smoother.applySettings(this.oneEuroCfg);
      this.smoother.reset();
    } catch {
      this.oneEuroCfg = { ...ONE_EURO_DEFAULTS, profile: name };
      this.smoother.applySettings(this.oneEuroCfg);
    }
  }

  async initPerformanceProfile() {
    const mode = parsePerfModeFromUrl();
    const { profile, capability, dynamic } = await resolveBrowserPerformance(mode);
    this.profileName = profile.name;
    this.profileLabel = profile.label;
    this.capabilityDetail = capability.detail;
    this.targetFps = profile.targetFps;
    this.fpsMin = profile.fpsMin;
    this.fpsMax = profile.fpsMax;
    this.camWidth = profile.cameraWidth;
    this.camHeight = profile.cameraHeight;
    this.yoloEvery = 2;
    this.shgEvery = profile.shgEvery;
    this.cpuSlowdown = Number(profile.cpuSlowdown) || 1;
    this.smoothMode = true;
    this.yoloImgsz = 640;
    this.perfScaler = new BrowserDynamicScaler(profile, dynamic);
    this.camAdapter = new CameraSizeAdapter(
      profile.cameraLadder,
      profile.cameraWidth,
      profile.cameraHeight
    );
    return { profile, capability };
  }

  async applyCamSize(width, height) {
    if (!this.stream || this.camResizeInFlight) return;
    const track = this.stream.getVideoTracks()[0];
    if (!track) return;
    this.camResizeInFlight = true;
    try {
      await track.applyConstraints({
        width: { ideal: width },
        height: { ideal: height },
      });
      this.camWidth = width;
      this.camHeight = height;
      this.tipPatch = null;
    } catch (_) {
      /* keep prior size */
    } finally {
      this.camResizeInFlight = false;
    }
  }

  adaptInferenceLoad() {
    if (this.perfScaler) {
      this.perfScaler.observe(this.pipeMsEma, this.fpsEma);
      this.yoloEvery = 2;
      this.shgEvery = this.perfScaler.base?.lockFps
        ? this.perfScaler.base.shgEvery
        : this.perfScaler.shgEvery;
      this.targetFps = this.perfScaler.targetFps;
    }
    if (this.camAdapter && this.lastHeavyMs > 0) {
      const adj = this.camAdapter.observe(this.lastHeavyMs, this.targetFps);
      if (adj.changed) this.applyCamSize(adj.width, adj.height);
    }
  }

  configureOrt(useProxy = true) {
    ort.env.wasm.wasmPaths = WASM_PATH;
    const canSAB =
      typeof SharedArrayBuffer !== "undefined" &&
      (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
    const cores = navigator.hardwareConcurrency || 2;
    ort.env.wasm.numThreads = canSAB
      ? Math.min(useProxy ? 2 : 4, cores)
      : 1;
    ort.env.wasm.proxy = !!useProxy;
    this.ortProxy = !!useProxy;
  }

  async createSession(url) {
    await yieldToPaint();
    return ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  async loadModels() {
    this.onStatus("Loading YOLO + SHGNet in separate Web Workers…");
    await yieldToPaint();
    const { profile, capability } = await this.initPerformanceProfile();
    await this.loadOneEuroSettings(profile.name);
    await yieldToPaint();

    this.useDedicatedWorker = false;
    if (this.ortWorker) {
      this.ortWorker.terminate();
      this.ortWorker = null;
    }
    try {
      this.ortWorker = new OrtInferenceWorker({
        yoloWorkerUrl: "/yolo-worker.js",
        shgWorkerUrl: "/shg-worker.js",
      });
      await this.ortWorker.init({
        yoloUrl: YOLO_URL,
        shgUrl: SHGNET_URL,
        wasmPaths: WASM_PATH,
        yoloImgsz: this.yoloImgsz,
      });
      this.useDedicatedWorker = true;
      this.shgSession = { worker: true };
      this.yoloPose = {
        lastMs: 0,
        detect: async (source) => {
          const det = await this.ortWorker.detectYolo(source);
          this.yoloPose.lastMs = this.ortWorker.lastYoloMs;
          return det;
        },
      };
      this.ortProxy = true;
    } catch (workerErr) {
      console.warn("[ort] dual Workers failed; ORT proxy fallback", workerErr);
      if (this.ortWorker) {
        this.ortWorker.terminate();
        this.ortWorker = null;
      }
      this.configureOrt(true);
      let shg;
      let yolo;
      try {
        yolo = await this.createSession(YOLO_URL);
        await yieldToPaint();
        shg = await this.createSession(SHGNET_URL);
      } catch (firstErr) {
        console.warn("ORT proxy Worker failed; falling back to main-thread wasm", firstErr);
        this.configureOrt(false);
        yolo = await this.createSession(YOLO_URL);
        await yieldToPaint();
        shg = await this.createSession(SHGNET_URL);
      }
      this.shgSession = shg;
      this.yoloPose = new YoloPoseBrowser(
        yolo,
        (data, dims) => new ort.Tensor("float32", data, dims),
        this.yoloImgsz,
        YOLO_CONF
      );
      this.yoloPose.yieldBeforeRun = true;
    }

    this.onStatus(
      `Ready · ${profile.label}${this.smoothMode ? " · smooth-cam" : ""} · ` +
        `${this.useDedicatedWorker ? "WW×2 (YOLO|SHG)" : this.ortProxy ? "wasm+Worker" : "wasm"}\n` +
        `${capability.detail} · ${profile.targetFps} fps · Y/2 · ${profile.cameraWidth}x${profile.cameraHeight}`
    );
    return true;
  }

  get ready() {
    return !!(this.shgSession && this.yoloPose);
  }

  clearEarLock(reason) {
    this.smoother.reset();
    this.rawRel = null;
    this.firstLock = true;
    this.geo = null;
    this.overlay = null;
    this.holdTip = null;
    this.tip = null;
    this.lastYoloTip = null;
    this.tipVelX = 0;
    this.tipVelY = 0;
    this.tipLk?.reset();
    this.tipSnap = true;
    this.side = null;
    if (reason) console.log(`[earring] clear lock: ${reason}`);
  }

  /** Soft YOLO tip merge — avoid fighting LK / shaking landmarks. */
  adoptYoloTip(tipPt, { reseatLk = true } = {}) {
    const prev = this.holdTip;
    this.lastYoloTip = { x: tipPt.x, y: tipPt.y };
    if (!prev) {
      this.holdTip = { x: tipPt.x, y: tipPt.y };
      this.tip = this.holdTip;
      if (reseatLk) this.tipLk.reset();
      this.tipSnap = true;
      return this.holdTip;
    }
    const jump = Math.hypot(tipPt.x - prev.x, tipPt.y - prev.y);
    if (jump > 10) {
      this.holdTip = { x: tipPt.x, y: tipPt.y };
      this.tip = this.holdTip;
      if (reseatLk) this.tipLk.reset();
      this.tipSnap = true;
    } else if (jump > 1.5) {
      this.holdTip = {
        x: prev.x * 0.75 + tipPt.x * 0.25,
        y: prev.y * 0.75 + tipPt.y * 0.25,
      };
      this.tip = this.holdTip;
    }
    return this.holdTip;
  }

  updateGeoFromYolo(yolo, vw, vh) {
    if (!isSideProfile(yolo)) {
      if (this.rawRel || this.overlay || this.holdTip) this.clearEarLock("not_side_profile");
      else this.overlay = null;
      return false;
    }
    const tipPt = yolo.tip;
    if (this.side && yolo.side && yolo.side !== this.side) {
      this.clearEarLock(`side_${this.side}_to_${yolo.side}`);
    }
    if (this.holdTip || this.lastYoloTip) {
      const prev = this.holdTip || this.lastYoloTip;
      const jump = Math.hypot(tipPt.x - prev.x, tipPt.y - prev.y);
      const lim = Math.max(36, (this.geo?.side || 80) * 0.45);
      if (this.rawRel && jump > lim) this.clearEarLock(`tip_jump_${jump.toFixed(0)}px`);
    }
    const pinna = pinnaHeight(yolo, vw, vh);
    const sideLen = pinna * CROP_PAD;
    const { ncx, ncy, mx } = tipCropCenter(tipPt, pinna, yolo, yolo.side, vw);
    if (!this.geo) {
      this.geo = { cx: ncx, cy: ncy, side: sideLen };
    } else {
      const a = 0.45;
      this.geo = {
        cx: (1 - a) * this.geo.cx + a * ncx,
        cy: (1 - a) * this.geo.cy + a * ncy,
        side: (1 - a) * this.geo.side + a * sideLen,
      };
    }
    const half = this.geo.side * 0.5;
    if (
      Math.abs(tipPt.x - this.geo.cx) > half * 0.55 ||
      Math.abs(tipPt.y - this.geo.cy) > half * 0.55
    ) {
      const r = rescueCropCenter(tipPt, pinna, mx);
      this.geo = { cx: r.cx, cy: r.cy, side: this.geo.side };
    }
    this.side = yolo.side;
    this.adoptYoloTip(tipPt, { reseatLk: true });
    return true;
  }

  drawSquareCrop(source, cx, cy, sidePx, needFlip) {
    const s = Math.max(32, Math.round(sidePx));
    const ox = Math.round(cx - s * 0.5);
    const oy = Math.round(cy - s * 0.5);
    this.padCanvas.width = s;
    this.padCanvas.height = s;
    this.padCtx.fillStyle = "rgb(114,114,114)";
    this.padCtx.fillRect(0, 0, s, s);
    const sw = source.width || source.videoWidth || 0;
    const sh = source.height || source.videoHeight || 0;
    const sx1 = Math.max(0, ox);
    const sy1 = Math.max(0, oy);
    const sx2 = Math.min(sw, ox + s);
    const sy2 = Math.min(sh, oy + s);
    const dx = sx1 - ox;
    const dy = sy1 - oy;
    if (sx2 > sx1 && sy2 > sy1) {
      this.padCtx.drawImage(
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
    this.cropCtx.save();
    if (needFlip) {
      this.cropCtx.translate(256, 0);
      this.cropCtx.scale(-1, 1);
    }
    this.cropCtx.drawImage(this.padCanvas, 0, 0, s, s, 0, 0, 256, 256);
    this.cropCtx.restore();
    return { ox, oy, sidePx: s };
  }

  cropToTensor() {
    const img = this.cropCtx.getImageData(0, 0, 256, 256);
    return new ort.Tensor("float32", canvasRgbaToBgrChw(img), [1, 3, 256, 256]);
  }

  hullSquare(pts, pad) {
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

  async runShg(needFlip, ox, oy, sidePx) {
    const t0 = performance.now();
    let pts256;
    if (this.useDedicatedWorker && this.ortWorker?.ready) {
      const chw = canvasRgbaToBgrChw(
        this.cropCtx.getImageData(0, 0, 256, 256)
      );
      const out = await this.ortWorker.runShg(chw, [1, 3, 256, 256]);
      pts256 = heatmapsToPointsSoft({ data: out.data, dims: out.dims }, 256);
      this.inferMsEma = this.inferMsEma
        ? this.inferMsEma * 0.85 + (this.ortWorker.lastShgMs || 0) * 0.15
        : this.ortWorker.lastShgMs || performance.now() - t0;
    } else {
      const out = await this.shgSession.run({
        [this.shgSession.inputNames[0]]: this.cropToTensor(),
      });
      const ms = performance.now() - t0;
      this.inferMsEma = this.inferMsEma ? this.inferMsEma * 0.85 + ms * 0.15 : ms;
      pts256 = heatmapsToPointsSoft(out[this.shgSession.outputNames[0]], 256);
    }
    const score = pts256.score ?? 0;
    if (needFlip) {
      pts256 = pts256.map(([x, y]) => [255 - x, y]);
    }
    const scale = sidePx / 256;
    const pts = pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
    return { pts, score };
  }

  async inferFromSnapshot(source, vw, vh, tipPt, sideNow, geoNow, yolo) {
    if (!geoNow || !tipPt || !sideNow) return null;
    if (
      !this.shgSession &&
      !(this.useDedicatedWorker && this.ortWorker?.ready)
    )
      return null;

    let { cx, cy, side: sideLen } = geoNow;
    const half = sideLen * 0.5;
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
      this.geo = { cx, cy, side: sideLen };
    }

    const preferFlip = sideNow === "LEFT";
    let { ox, oy, sidePx } = this.drawSquareCrop(
      source,
      cx,
      cy,
      sideLen,
      preferFlip
    );
    let box = [
      Math.max(0, Math.round(cx - sidePx * 0.5)),
      Math.max(0, Math.round(cy - sidePx * 0.5)),
      Math.min(vw, Math.round(cx + sidePx * 0.5)),
      Math.min(vh, Math.round(cy + sidePx * 0.5)),
    ];

    let { pts, score } = await this.runShg(preferFlip, ox, oy, sidePx);
    const lock = this.firstLock;
    let ok1 = landmarksOk(pts, tipPt, sidePx);
    // Dual-flip only when needed — always-on LEFT compare doubled WASM latency.
    const shouldFlipCompare =
      (preferFlip && (lock || !ok1 || score < 0.10)) ||
      (lock && !ok1) ||
      score < 0.08;
    if (shouldFlipCompare) {
      this.drawSquareCrop(source, cx, cy, sideLen, !preferFlip);
      const alt = await this.runShg(!preferFlip, ox, oy, sidePx);
      const okAlt = landmarksOk(alt.pts, tipPt, sidePx);
      if (
        alt.score > score + 0.02 ||
        (okAlt && !ok1) ||
        (okAlt && alt.score >= score)
      ) {
        pts = alt.pts;
        score = alt.score;
        ok1 = landmarksOk(pts, tipPt, sidePx);
      }
    }
    if (preferFlip && !(ok1 && score > MIN_SHG_SCORE)) return null;

    if (lock && ok1 && score > MIN_SHG_SCORE) {
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
        const hull = this.hullSquare(pts, REFINE_PAD);
        if (
          Math.abs(tipPt.x - hull.cx) < hull.side * 0.45 &&
          Math.abs(tipPt.y - hull.cy) < hull.side * 0.45
        ) {
          const c2 = this.drawSquareCrop(
            source,
            hull.cx,
            hull.cy,
            hull.side,
            preferFlip
          );
          const refined = await this.runShg(
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
        this.geo = { cx, cy, side: sideLen };
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

  /** Piercing #56 locked to tip (no One Euro lag) so the earring stud sticks. */
  stickyPierce(tipPt) {
    if (!this.rawRel || !tipPt || !this.rawRel[PIERCING_INDEX]) return null;
    const [rx, ry] = this.rawRel[PIERCING_INDEX];
    return { x: tipPt.x + rx, y: tipPt.y + ry };
  }

  applyTipHold(tipPt, sideNow, geoNow, vw, vh, _dt, snap) {
    if (!this.rawRel || !tipPt || !sideNow || !geoNow) return false;
    // Rigid tip-lock — One Euro never delays tip; offsets already smoothed at SHG
    if (snap) this.smoother.syncRelative(this.rawRel);
    const rigid = this.smoother.compose
      ? this.smoother.compose(tipPt, this.rawRel)
      : this.rawRel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]);
    const pierce = this.stickyPierce(tipPt);
    if (pierce && rigid[PIERCING_INDEX]) {
      rigid[PIERCING_INDEX] = [pierce.x, pierce.y];
    }
    const sidePx = geoNow.side;
    this.overlay = {
      tip: { x: tipPt.x, y: tipPt.y },
      box: [
        Math.max(0, Math.round(geoNow.cx - sidePx * 0.5)),
        Math.max(0, Math.round(geoNow.cy - sidePx * 0.5)),
        Math.min(vw, Math.round(geoNow.cx + sidePx * 0.5)),
        Math.min(vh, Math.round(geoNow.cy + sidePx * 0.5)),
      ],
      landmarks: rigid,
      side: sideNow,
      pierce,
    };
    return true;
  }

  async processFrame(vw, vh, dt) {
    const tick = this.inferTick++;
    let runYolo =
      (!this.holdTip && !this.tip) || this.firstLock
        ? true
        : tick % Math.max(1, this.yoloEvery) === 0;
    // Dense SHG: every frame when free (S/1). Old ===1 broke when shgEvery===1.
    let runShgNet =
      !this.rawRel || this.firstLock
        ? !!(this.holdTip || this.tip)
        : tick % Math.max(1, this.shgEvery) === 0;
    if (
      runYolo &&
      this.holdTip &&
      this.rawRel &&
      !this.firstLock &&
      this.lastHeavyMs > 180 &&
      tick % (this.yoloEvery * 2) !== 0
    ) {
      runYolo = false;
    }
    if (!runYolo && !runShgNet) {
      this.lastInferDoneTs = performance.now();
      return;
    }

    await yieldToPaint();
    const tPipe = performance.now();
    this.snapCanvas.width = vw;
    this.snapCanvas.height = vh;
    if (this.mirrorFeed) {
      this.snapCtx.save();
      this.snapCtx.translate(vw, 0);
      this.snapCtx.scale(-1, 1);
      this.snapCtx.drawImage(this.video, 0, 0, vw, vh);
      this.snapCtx.restore();
    } else {
      this.snapCtx.drawImage(this.video, 0, 0, vw, vh);
    }

    let tipPt = this.tip;
    let sideNow = this.side;
    let geoNow = this.geo;

    if (runYolo && this.yoloPose) {
      await yieldToPaint();
      const y = await this.yoloPose.detect(this.snapCanvas);
      if (y) {
        if (this.side && y.side !== this.side) {
          this.clearEarLock(`yolo_side_${this.side}_to_${y.side}`);
        }
        this.lastYolo = y;
        if (!this.updateGeoFromYolo(y, vw, vh)) {
          this.onStatus("Turn head — clear SIDE PROFILE of one ear");
          this.overlay = null;
          return;
        }
        this.tipSnap = true;
        tipPt = this.tip;
        sideNow = this.side;
        geoNow = this.geo;
        const now = performance.now();
        if (this.lastYoloTip && this.lastTipTs) {
          const dtt = Math.max(0.016, (now - this.lastTipTs) / 1000);
          this.tipVelX = (tipPt.x - this.lastYoloTip.x) / dtt;
          this.tipVelY = (tipPt.y - this.lastYoloTip.y) / dtt;
        }
        this.lastYoloTip = { x: tipPt.x, y: tipPt.y };
        this.lastTipTs = now;
        this.tipLk.reset();
      }
    }

    if (!geoNow || !tipPt || !sideNow) return;

    if (!runShgNet) {
      if (runYolo && tipPt) {
        const prev = this.holdTip;
        this.holdTip = { x: tipPt.x, y: tipPt.y };
        this.tip = this.holdTip;
        this.tipPatch = null;
        if (prev && this.geo && this.rawRel) {
          this.geo = {
            cx: this.geo.cx + (this.holdTip.x - prev.x),
            cy: this.geo.cy + (this.holdTip.y - prev.y),
            side: this.geo.side,
          };
        }
        this.tipSnap = true;
      }
      const ms = performance.now() - tPipe;
      this.pipeMsEma = this.pipeMsEma ? this.pipeMsEma * 0.9 + ms * 0.1 : ms;
      if (runYolo) this.lastHeavyMs = this.yoloPose?.lastMs || ms;
      this.lastInferDoneTs = performance.now();
      this.adaptInferenceLoad();
      return;
    }

    await yieldToPaint();
    const result = await this.inferFromSnapshot(
      this.snapCanvas,
      vw,
      vh,
      tipPt,
      sideNow,
      geoNow,
      this.lastYolo
    );
    if (!result) return;

    const shgTip = { x: result.tip.x, y: result.tip.y };
    this.holdTip = { x: shgTip.x, y: shgTip.y };
    this.tip = this.holdTip;
    this.tipPatch = null;
    const newRel = result.pts.map(([x, y]) => [x - shgTip.x, y - shgTip.y]);

    const anchor = this.holdTip;
    const snap = this.firstLock;
    const stepPx = this.oneEuroMaxStep();
    this.rawRel = this.smoother.filterOffsets
      ? this.smoother.filterOffsets(newRel, dt, result.side, {
          maxStepPx: stepPx,
          snap,
        })
      : this.smoother
          .updateRelative(result.pts, anchor, dt, result.side, {
            maxStepPx: stepPx,
            snap,
          })
          .map(([x, y]) => [x - anchor.x, y - anchor.y]);
    if (snap) this.firstLock = false;
    this.smoother.syncRelative(this.rawRel);

    const landmarks = this.smoother.compose(anchor, this.rawRel);
    const pierce = this.stickyPierce(anchor);
    if (pierce && landmarks[PIERCING_INDEX]) {
      landmarks[PIERCING_INDEX] = [pierce.x, pierce.y];
    }

    this.overlay = {
      tip: { x: anchor.x, y: anchor.y },
      box: result.box,
      landmarks,
      side: result.side,
      pierce,
    };
    const ms = performance.now() - tPipe;
    this.lastHeavyMs = ms;
    this.pipeMsEma = this.pipeMsEma ? this.pipeMsEma * 0.85 + ms * 0.15 : ms;
    this.lastInferDoneTs = performance.now();
    this.adaptInferenceLoad();
  }

  captureTipPatch(tx, ty) {
    const x0 = Math.round(tx - (this.TIP_PATCH - 1) / 2);
    const y0 = Math.round(ty - (this.TIP_PATCH - 1) / 2);
    if (
      x0 < 0 ||
      y0 < 0 ||
      x0 + this.TIP_PATCH > this.canvas.width ||
      y0 + this.TIP_PATCH > this.canvas.height
    )
      return null;
    const img = this.ctx.getImageData(x0, y0, this.TIP_PATCH, this.TIP_PATCH);
    const g = new Float32Array(this.TIP_PATCH * this.TIP_PATCH);
    for (let i = 0, p = 0; i < img.data.length; i += 4, p++) {
      g[p] =
        0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
    }
    return { g, x0, y0 };
  }

  trackTipOnCanvas() {
    if (!this.holdTip || !this.tipPatch || !this.canvas.width) return this.holdTip;
    const cx = Math.round(this.holdTip.x);
    const cy = Math.round(this.holdTip.y);
    const half = (this.TIP_PATCH - 1) / 2;
    const x1 = Math.max(0, cx - this.TIP_SEARCH);
    const y1 = Math.max(0, cy - this.TIP_SEARCH);
    const x2 = Math.min(this.canvas.width - this.TIP_PATCH, cx + this.TIP_SEARCH);
    const y2 = Math.min(
      this.canvas.height - this.TIP_PATCH,
      cy + this.TIP_SEARCH
    );
    if (x2 <= x1 || y2 <= y1) return this.holdTip;

    const region = this.ctx.getImageData(
      x1,
      y1,
      x2 - x1 + this.TIP_PATCH,
      y2 - y1 + this.TIP_PATCH
    );
    const rw = region.width;
    const tpl = this.tipPatch.g;
    const grayAt = (ox, oy, px, py) => {
      const i = ((oy + py) * rw + (ox + px)) * 4;
      return (
        0.299 * region.data[i] +
        0.587 * region.data[i + 1] +
        0.114 * region.data[i + 2]
      );
    };
    const sadAt = (ox, oy) => {
      let sad = 0;
      for (let py = 0; py < this.TIP_PATCH; py++) {
        for (let px = 0; px < this.TIP_PATCH; px++) {
          sad += Math.abs(grayAt(ox, oy, px, py) - tpl[py * this.TIP_PATCH + px]);
        }
      }
      return sad;
    };

    let best = Infinity;
    let bx = cx;
    let by = cy;
    const limX = x2 - x1;
    const limY = y2 - y1;
    const step = this.TIP_COARSE || 2;
    for (let oy = 0; oy <= limY; oy += step) {
      for (let ox = 0; ox <= limX; ox += step) {
        const sad = sadAt(ox, oy);
        if (sad < best) {
          best = sad;
          bx = x1 + ox + half;
          by = y1 + oy + half;
        }
      }
    }
    const rx0 = Math.max(0, bx - half - x1 - step);
    const ry0 = Math.max(0, by - half - y1 - step);
    const rx1 = Math.min(limX, bx - half - x1 + step);
    const ry1 = Math.min(limY, by - half - y1 + step);
    for (let oy = ry0; oy <= ry1; oy++) {
      for (let ox = rx0; ox <= rx1; ox++) {
        const sad = sadAt(ox, oy);
        if (sad < best) {
          best = sad;
          bx = x1 + ox + half;
          by = y1 + oy + half;
        }
      }
    }
    const dist = Math.hypot(bx - this.holdTip.x, by - this.holdTip.y);
    if (dist > this.TIP_SEARCH) return this.holdTip;
    if (best > this.TIP_PATCH * this.TIP_PATCH * 45) return this.holdTip;
    return { x: bx, y: by };
  }

  paintVideo() {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh) return null;
    if (this.canvas.width !== vw || this.canvas.height !== vh) {
      this.canvas.width = vw;
      this.canvas.height = vh;
    }
    if (this.mirrorFeed) {
      this.ctx.save();
      this.ctx.translate(vw, 0);
      this.ctx.scale(-1, 1);
      this.ctx.drawImage(this.video, 0, 0, vw, vh);
      this.ctx.restore();
    } else {
      this.ctx.drawImage(this.video, 0, 0, vw, vh);
    }
    return { vw, vh };
  }

  clampFps(v) {
    if (!Number.isFinite(v) || v <= 0) return 0;
    return Math.max(this.fpsMin, Math.min(this.fpsMax, v));
  }

  onDisplayTick(ts) {
    if (!this.live) return;
    const frame = this.paintVideo();
    if (!frame) {
      this.emitFrame(ts);
      return;
    }
    const { vw, vh } = frame;
    const dt = this.lastTs
      ? Math.min(0.05, (ts - this.lastTs) / 1000)
      : DT_FALLBACK;
    this.lastTs = ts;
    this.frameIdx++;

    const earLocked = !!(
      this.rawRel &&
      !this.firstLock &&
      this.side &&
      this.geo &&
      (this.holdTip || this.tip)
    );
    if (earLocked) {
      const prev = this.holdTip || this.tip;
      const lk = this.tipLk.update(this.ctx, prev);
      let tracked = prev;
      if (lk.ok) {
        const step = Math.hypot(lk.x - prev.x, lk.y - prev.y);
        if (step >= 0.45) tracked = { x: lk.x, y: lk.y };
      }
      if (
        this.lastYoloTip &&
        Math.hypot(tracked.x - this.lastYoloTip.x, tracked.y - this.lastYoloTip.y) >
          Math.min(vw, vh) * 0.1
      ) {
        tracked = { x: this.lastYoloTip.x, y: this.lastYoloTip.y };
        this.tipLk.reset();
      }
      if (
        Math.hypot(tracked.x - prev.x, tracked.y - prev.y) >
        Math.min(40, (this.geo?.side || 80) * 0.25)
      ) {
        this.clearEarLock("tip_lk_abort");
      } else if (tracked && prev && this.geo) {
        this.geo = {
          cx: this.geo.cx + (tracked.x - prev.x),
          cy: this.geo.cy + (tracked.y - prev.y),
          side: this.geo.side,
        };
        this.holdTip = tracked;
        this.tip = tracked;
        this.applyTipHold(
          this.holdTip || this.tip,
          this.side,
          this.geo,
          vw,
          vh,
          dt,
          this.tipSnap
        );
        this.tipSnap = false;
      }
    } else if (this.overlay && (!this.rawRel || this.firstLock)) {
      this.overlay = null;
    }

    this.emitFrame(ts);

    if (!this.processing && this.shgSession) {
      const now = performance.now();
      const minGap = this.smoothMode
        ? Math.max(80, Math.min(400, (this.lastHeavyMs || 200) * 0.4))
        : 0;
      if (this.lastInferStartTs && now - this.lastInferStartTs < minGap) return;
      this.lastInferStartTs = now;
      this.processing = true;
      this.processFrame(vw, vh, dt)
        .catch((e) => this.onStatus(`Infer error: ${e?.message || e}`))
        .finally(() => {
          this.processing = false;
        });
    }
  }

  emitFrame(ts) {
    if (this.lastDisplayTs) {
      const inst = 1000 / Math.max(1e-3, ts - this.lastDisplayTs);
      const ema = this.fpsEma ? this.fpsEma * 0.85 + inst * 0.15 : inst;
      this.fpsEma = this.clampFps(ema);
    }
    this.lastDisplayTs = ts;
    if (this.onOverlay) {
      this.onOverlay(this.overlay, {
        fps: this.fpsEma,
        pipeMs: this.pipeMsEma,
        inferMs: this.inferMsEma,
        profile: this.profileName,
        yoloEvery: this.yoloEvery,
        shgEvery: this.shgEvery,
        targetFps: this.targetFps,
        dt: this.lastTs ? DT_FALLBACK : DT_FALLBACK,
      });
    }
  }

  loopLive = (ts) => {
    if (!this.live) return;
    this.rafId = requestAnimationFrame(this.loopLive);
    const want = this.targetFps;
    const interval = 1000 / want;
    if (this.lastDrawTs && ts - this.lastDrawTs < interval - 0.5) return;
    if (this.lastDrawTs) {
      const elapsed = ts - this.lastDrawTs;
      const steps = Math.max(1, Math.floor(elapsed / interval));
      this.lastDrawTs += steps * interval;
      if (ts - this.lastDrawTs > interval) this.lastDrawTs = ts;
    } else {
      this.lastDrawTs = ts;
    }
    this.onDisplayTick(ts);
  };

  async startCamera() {
    if (this.live) return;
    if (!this.ready) {
      this.onStatus("Models not ready — load models first.");
      return;
    }
    const wantFps = this.targetFps;
    const w = this.camWidth;
    const h = this.camHeight;
    const sizeFps = {
      width: { ideal: w },
      height: { ideal: h },
      frameRate: { ideal: wantFps, max: this.fpsMax },
    };
    try {
      this.onStatus("Requesting front camera…");
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { exact: "user" }, ...sizeFps },
        });
      } catch (_) {
        try {
          this.stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: { ideal: "user" }, ...sizeFps },
          });
        } catch (_) {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const cams = devices.filter((d) => d.kind === "videoinput");
          const front =
            cams.find((d) => /front|user|face|selfie/i.test(d.label || "")) ||
            cams.find(
              (d) => !/back|rear|environment|world|ultra/i.test(d.label || "")
            );
          this.stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: front?.deviceId
              ? { deviceId: { exact: front.deviceId }, ...sizeFps }
              : { facingMode: "user", ...sizeFps },
          });
        }
      }
    } catch (e2) {
      this.onStatus(`Camera error: ${e2?.name || e2}`);
      throw e2;
    }
    this.video.srcObject = this.stream;
    this.video.style.display = "none";
    await this.video.play();
    try {
      const s = this.stream.getVideoTracks()[0]?.getSettings?.() || {};
      const facing = String(s.facingMode || "user").toLowerCase();
      this.mirrorFeed = facing !== "environment";
      if (facing === "environment") {
        for (const t of this.stream.getTracks()) t.stop();
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: "user" }, ...sizeFps },
        });
        this.video.srcObject = this.stream;
        await this.video.play();
        const s2 = this.stream.getVideoTracks()[0]?.getSettings?.() || {};
        this.mirrorFeed =
          String(s2.facingMode || "user").toLowerCase() !== "environment";
      }
    } catch (_) {
      this.mirrorFeed = true;
    }
    this.live = true;
    this.resetTracking();
    if (this.perfScaler) this.perfScaler.reset();
    this.onStatus(
      `Live · ${this.profileLabel || this.profileName} · ${wantFps} fps · ` +
        `Y/${this.yoloEvery} S/${this.shgEvery} — show SIDE PROFILE of one ear`
    );
    this.rafId = 0;
    this.rafId = requestAnimationFrame(this.loopLive);
  }

  stopCamera() {
    this.live = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
    this.onStatus("Camera stopped.");
  }
}
