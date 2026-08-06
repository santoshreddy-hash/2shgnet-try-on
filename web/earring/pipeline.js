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

const SHGNET_URL = "/models/shgnet/SHGNet-56.onnx";
const YOLO_URL = "/models/yolo/yolo26n-pose.onnx";
const WASM_PATH = "/vendor/onnxruntime-web/dist/";

const CROP_PAD = 1.65;
const REFINE_PAD = 1.35;
const YOLO_EVERY = 1;
const SHG_EVERY = 1;
// Must match ONNX fixed input [1,3,640,640] — other sizes break detection
const YOLO_IMGSZ = 640;
const YOLO_CONF = 0.22;
const YOLO_LOST_MAX = 8;
const CAM_FPS_MIN = 20;
const CAM_FPS_MAX = 30;
const CAM_FPS_DEFAULT = 25;
const CAM_WIDTH = 960;
const CAM_HEIGHT = 540;
const DT_FALLBACK = 1 / CAM_FPS_DEFAULT;
const MIRROR_FEED = true;
const NUM_LANDMARKS = 56;
export const PIERCING_INDEX = 55;

const ONE_EURO_DEFAULTS = {
  min_cutoff: 1.8,
  beta: 0.85,
  d_cutoff: 1.19,
  rest_speed_px: 1.5,
  rest_hold_frames: 2,
  rest_release_mult: 1.5,
  max_step_px: 42.0,
};
const BLEND_SMOOTH_REST = 0.9;
const BLEND_SMOOTH_MOVE = 0.68;
const BLEND_SPEED_LOW = 40; // px/s
const BLEND_SPEED_HIGH = 260; // px/s
const TIP_DEADZONE_PX = 0.7;
const TIP_MAX_STEP_PX = 22;

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
    this.stream = null;
    this.live = false;
    this.processing = false;
    this.rafId = 0;
    this.ortProxy = false;
    this.targetFps = CAM_FPS_DEFAULT;

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
    this.TIP_PATCH = 11;
    this.TIP_SEARCH = 18;
    this.geo = null;
    this.rawRel = null;
    this.smoothRel = null;
    this.firstLock = true;
    this.overlay = null;
    this.yoloLost = 0;
    this.transferring = false;
    this.tipVel = { x: 0, y: 0 };
    if (this.yoloPose) this.yoloPose.preferSide = null;
    this.smoother.reset();
  }

  clearEarLock() {
    this.beginTransfer();
    this.lastYolo = null;
    this.yoloLost = 0;
    if (this.yoloPose) this.yoloPose.preferSide = null;
  }

  beginTransfer() {
    this.transferring = true;
    this.side = null;
    this.tip = null;
    this.holdTip = null;
    this.geo = null;
    this.rawRel = null;
    this.smoothRel = null;
    this.overlay = null;
    this.tipPatch = null;
    this.firstLock = true;
    this.tipSnap = true;
    this.tipVel = { x: 0, y: 0 };
    this.smoother.reset();
  }

  oneEuroMaxStep() {
    return Math.max(1, Number(this.oneEuroCfg.max_step_px) || 42);
  }

  async loadOneEuroSettings() {
    try {
      const res = await fetch("/one_euro_settings.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.oneEuroCfg = { ...ONE_EURO_DEFAULTS, ...data };
      this.smoother.applySettings(this.oneEuroCfg);
    } catch {
      this.smoother.applySettings(ONE_EURO_DEFAULTS);
      this.oneEuroCfg = { ...ONE_EURO_DEFAULTS };
    }
  }

  configureOrt(useProxy) {
    ort.env.wasm.wasmPaths = WASM_PATH;
    const canSAB =
      typeof SharedArrayBuffer !== "undefined" &&
      (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
    ort.env.wasm.numThreads = canSAB
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;
    ort.env.wasm.proxy = !!useProxy;
    this.ortProxy = !!useProxy;
  }

  async createSession(url) {
    return ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  async loadModels() {
    this.onStatus("Loading ONNX models (YOLO + SHGNet)…");
    await this.loadOneEuroSettings();
    this.configureOrt(true);
    let shg;
    let yolo;
    try {
      shg = await this.createSession(SHGNET_URL);
      yolo = await this.createSession(YOLO_URL);
    } catch (firstErr) {
      console.warn("ORT proxy load failed; retrying without proxy", firstErr);
      this.configureOrt(false);
      shg = await this.createSession(SHGNET_URL);
      yolo = await this.createSession(YOLO_URL);
    }
    this.shgSession = shg;
    this.yoloPose = new YoloPoseBrowser(
      yolo,
      (data, dims) => new ort.Tensor("float32", data, dims),
      YOLO_IMGSZ,
      YOLO_CONF
    );
    this.onStatus(
      `Ready · YOLO + SHGNet · wasm${this.ortProxy ? "+proxy" : ""}`
    );
    return true;
  }

  get ready() {
    return !!(this.shgSession && this.yoloPose);
  }

  pinnaHeight(yolo, vw, vh) {
    const fmin = Math.min(vw, vh);
    const tip = yolo.tip;
    const cands = [];
    let tipNose = null;
    if (yolo.nose?.c >= 0.2) {
      tipNose = Math.hypot(tip.x - yolo.nose.x, tip.y - yolo.nose.y);
      if (tipNose > fmin * 0.03) cands.push(tipNose * 0.55);
    }
    if (yolo.eyeDist && yolo.eyeDist > fmin * 0.02) cands.push(yolo.eyeDist * 0.9);
    const [, y1, , y2] = yolo.bbox;
    const bh = y2 - y1;
    if (bh > 1) cands.push(bh * 0.12);
    if (!cands.length) return Math.max(40, fmin * 0.12);
    cands.sort((a, b) => a - b);
    let h =
      cands.length === 1 ? cands[0] : cands[Math.floor(cands.length / 2)];
    if (tipNose != null && tipNose > 1) h = Math.min(h, tipNose * 0.7);
    return Math.max(40, Math.min(fmin * 0.2, h));
  }

  isSideProfile(yolo) {
    if (yolo.earOtherConf != null && yolo.earConf != null) {
      if (yolo.earOtherConf >= 0.38 && yolo.earOtherConf >= yolo.earConf * 0.72)
        return false;
      if (yolo.earConf < 0.35) return false;
    }
    if (!yolo.nose || yolo.nose.c < 0.2) return true;
    const dx = Math.abs(yolo.tip.x - yolo.nose.x);
    const d = Math.hypot(yolo.tip.x - yolo.nose.x, yolo.tip.y - yolo.nose.y);
    if (dx < 22 || d < 28) return false;
    return true;
  }

  medial(yolo, tip, side, vw) {
    if (yolo.nose?.c >= 0.2) {
      const vx = yolo.nose.x - tip.x;
      const vy = yolo.nose.y - tip.y;
      const n = Math.hypot(vx, vy);
      if (n > 1e-3) return [vx / n, vy / n];
    }
    const vx = 0.5 * vw - tip.x;
    if (Math.abs(vx) > 1e-3) return [Math.sign(vx), 0];
    return [side === "LEFT" ? -1 : 1, 0];
  }

  updateGeoFromYolo(yolo, vw, vh) {
    if (!this.isSideProfile(yolo)) return false;
    const tipPt = yolo.tip;
    const pinna = this.pinnaHeight(yolo, vw, vh);
    const sideLen = pinna * CROP_PAD;
    const [mx] = this.medial(yolo, tipPt, yolo.side, vw);
    const ncx = tipPt.x + mx * (0.1 * pinna);
    const ncy = tipPt.y + 0.06 * pinna;
    if (!this.geo) {
      this.geo = { cx: ncx, cy: ncy, side: sideLen };
    } else {
      const a = 0.85;
      this.geo = {
        cx: (1 - a) * this.geo.cx + a * ncx,
        cy: (1 - a) * this.geo.cy + a * ncy,
        side: (1 - a) * this.geo.side + a * sideLen,
      };
    }
    this.side = yolo.side;
    this.tip = tipPt;
    this.holdTip = { x: tipPt.x, y: tipPt.y };
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
    this.cropCtx.setTransform(1, 0, 0, 1, 0, 0);
    this.cropCtx.clearRect(0, 0, 256, 256);
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

  landmarksOk(pts, tipPt, sidePx) {
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
    const bw = x1 - x0;
    const bh = y1 - y0;
    const span = Math.max(bw, bh);
    const ratio = span / Math.max(1, sidePx);
    if (ratio < 0.4 || ratio > 0.88) return false;
    if (Math.min(bw, bh) < span * 0.28) return false;
    let mx = 0,
      my = 0;
    for (const [x, y] of pts) {
      mx += x;
      my += y;
    }
    mx /= pts.length;
    my /= pts.length;
    if (Math.hypot(mx - tipPt.x, my - tipPt.y) > sidePx * 0.45) return false;
    const padX = 0.08 * bw;
    const padY = 0.08 * bh;
    if (tipPt.x < x0 - padX || tipPt.x > x1 + padX) return false;
    if (tipPt.y < y0 - padY || tipPt.y > y1 + padY) return false;
    return true;
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
    const out = await this.shgSession.run({
      [this.shgSession.inputNames[0]]: this.cropToTensor(),
    });
    const ms = performance.now() - t0;
    this.inferMsEma = this.inferMsEma ? this.inferMsEma * 0.85 + ms * 0.15 : ms;
    let pts256 = heatmapsToPointsSoft(out[this.shgSession.outputNames[0]], 256);
    const score = pts256.score ?? 0;
    if (needFlip) {
      pts256 = pts256.map(([x, y]) => [255 - x, y]);
    }
    const scale = sidePx / 256;
    const pts = pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
    return { pts, score };
  }

  async inferFromSnapshot(source, vw, vh, tipPt, sideNow, geoNow, yolo) {
    if (!geoNow || !tipPt || !sideNow || !this.shgSession) return null;

    let { cx, cy, side: sideLen } = geoNow;
    const half = sideLen * 0.5;
    if (
      Math.abs(tipPt.x - cx) > half * 0.55 ||
      Math.abs(tipPt.y - cy) > half * 0.55
    ) {
      if (yolo) {
        const pinna = sideLen / Math.max(CROP_PAD, 1e-3);
        const [mx] = this.medial(yolo, tipPt, sideNow, vw);
        cx = tipPt.x + mx * (0.1 * pinna);
        cy = tipPt.y + 0.06 * pinna;
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
    const ok1 = this.landmarksOk(pts, tipPt, sidePx);
    if (lock || !ok1) {
      this.drawSquareCrop(source, cx, cy, sideLen, !preferFlip);
      const alt = await this.runShg(!preferFlip, ox, oy, sidePx);
      if (
        alt.score > score ||
        (lock && this.landmarksOk(alt.pts, tipPt, sidePx) && !ok1)
      ) {
        pts = alt.pts;
        score = alt.score;
      }
    }

    if (lock && this.landmarksOk(pts, tipPt, sidePx) && score > 0.07) {
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
            this.landmarksOk(refined.pts, tipPt, c2.sidePx) &&
            refined.score >= score * 0.9
          ) {
            pts = refined.pts;
            score = refined.score;
          }
        }
      }
    }

    if (!(this.landmarksOk(pts, tipPt, sidePx) && score > 0.07)) return null;

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
    const rel = this.smoothRel || this.rawRel;
    if (!rel || !tipPt || !rel[PIERCING_INDEX]) return null;
    const [rx, ry] = rel[PIERCING_INDEX];
    return { x: tipPt.x + rx, y: tipPt.y + ry };
  }

  blendSmoothWeight(speedPxPerSec) {
    if (!Number.isFinite(speedPxPerSec)) return BLEND_SMOOTH_REST;
    if (speedPxPerSec <= BLEND_SPEED_LOW) return BLEND_SMOOTH_REST;
    if (speedPxPerSec >= BLEND_SPEED_HIGH) return BLEND_SMOOTH_MOVE;
    const t =
      (speedPxPerSec - BLEND_SPEED_LOW) / (BLEND_SPEED_HIGH - BLEND_SPEED_LOW);
    return BLEND_SMOOTH_REST + (BLEND_SMOOTH_MOVE - BLEND_SMOOTH_REST) * t;
  }

  blendRelativeCloud(relSmooth, relRaw, speedPxPerSec) {
    if (!relSmooth && !relRaw) return null;
    if (!relSmooth) return relRaw;
    if (!relRaw) return relSmooth;
    const ws = this.blendSmoothWeight(speedPxPerSec);
    const wr = 1 - ws;
    return relSmooth.map(([sx, sy], i) => {
      const [rx, ry] = relRaw[i] || [sx, sy];
      return [ws * sx + wr * rx, ws * sy + wr * ry];
    });
  }

  /** Rigid tip glue — zero lag. One Euro only on SHG shape updates. */
  applyTipHold(tipPt, sideNow, geoNow, vw, vh) {
    const relSmooth = this.smoothRel || this.rawRel;
    const relRaw = this.rawRel || this.smoothRel;
    const speed = Math.hypot(this.tipVel?.x || 0, this.tipVel?.y || 0);
    const rel = this.blendRelativeCloud(relSmooth, relRaw, speed);
    if (!rel || !tipPt || !sideNow || !geoNow) return false;
    const landmarks = rel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]);
    const pierce = this.stickyPierce(tipPt);
    if (pierce && landmarks[PIERCING_INDEX]) {
      landmarks[PIERCING_INDEX] = [pierce.x, pierce.y];
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
      landmarks,
      side: sideNow,
      pierce,
    };
    return true;
  }

  async processFrame(vw, vh, dt) {
    const tPipe = performance.now();
    this.snapCanvas.width = vw;
    this.snapCanvas.height = vh;
    if (MIRROR_FEED) {
      this.snapCtx.save();
      this.snapCtx.translate(vw, 0);
      this.snapCtx.scale(-1, 1);
      this.snapCtx.drawImage(this.video, 0, 0, vw, vh);
      this.snapCtx.restore();
    } else {
      this.snapCtx.drawImage(this.video, 0, 0, vw, vh);
    }

    const tick = this.inferTick++;
    const runYolo = tick % Math.max(1, YOLO_EVERY) === 0;
    const runShgNet = tick % Math.max(1, SHG_EVERY) === 0;
    let tipPt = this.tip;
    let sideNow = this.side;
    let geoNow = this.geo;

    if (runYolo && this.yoloPose) {
      const y = await this.yoloPose.detect(this.snapCanvas, {
        preferSide: this.transferring ? null : this.side,
      });
      if (y) {
        const sideSwitched = this.side && y.side !== this.side;
        const tipJumped =
          this.tip &&
          !sideSwitched &&
          Math.hypot(y.tip.x - this.tip.x, y.tip.y - this.tip.y) >
            Math.max(48, (this.geo?.side || 120) * 0.35);
        if (sideSwitched || tipJumped) this.beginTransfer();
        this.lastYolo = y;
        if (!this.updateGeoFromYolo(y, vw, vh)) {
          this.beginTransfer();
          this.yoloLost = (this.yoloLost || 0) + 1;
          if (this.yoloLost > YOLO_LOST_MAX) this.clearEarLock();
          this.onStatus("Turn head — clear SIDE PROFILE of one ear");
          return;
        }
        this.yoloLost = 0;
        this.tipSnap = true;
        tipPt = this.tip;
        sideNow = this.side;
        geoNow = this.geo;
      } else {
        this.yoloLost = (this.yoloLost || 0) + 1;
        if (this.yoloLost > 2) this.beginTransfer();
        if (this.yoloLost > YOLO_LOST_MAX) {
          this.clearEarLock();
          tipPt = sideNow = geoNow = null;
        }
      }
    }

    if (!geoNow || !tipPt || !sideNow) return;

    if (this.transferring || this.firstLock) {
      if (!runShgNet) return;
    }

    if (!runShgNet && (this.smoothRel || this.rawRel) && (this.holdTip || tipPt) && !this.transferring) {
      this.applyTipHold(
        this.holdTip || tipPt,
        sideNow,
        geoNow,
        vw,
        vh
      );
      this.tipSnap = false;
      const ms = performance.now() - tPipe;
      this.pipeMsEma = this.pipeMsEma ? this.pipeMsEma * 0.85 + ms * 0.15 : ms;
      return;
    }

    if (!runShgNet) return;

    const result = await this.inferFromSnapshot(
      this.snapCanvas,
      vw,
      vh,
      tipPt,
      sideNow,
      geoNow,
      this.lastYolo
    );
    if (!result) {
      if (this.transferring || this.firstLock) this.overlay = null;
      return;
    }

    const tx = result.tip.x;
    const ty = result.tip.y;
    this.holdTip = { x: tx, y: ty };
    this.tip = this.holdTip;
    if (!this.tipVel) this.tipVel = { x: 0, y: 0 };
    this.rawRel = result.pts.map(([x, y]) => [x - tx, y - ty]);

    const snap = this.firstLock || this.transferring;
    const sm = this.smoother.updateRelative(result.pts, this.holdTip, dt, result.side, {
      maxStepPx: this.oneEuroMaxStep(),
      snap,
    });
    this.smoothRel = sm.map(([x, y]) => [x - tx, y - ty]);
    this.firstLock = false;
    this.transferring = false;

    const relSmooth = this.smoothRel || this.rawRel;
    const relRaw = this.rawRel || this.smoothRel;
    const speed = Math.hypot(this.tipVel?.x || 0, this.tipVel?.y || 0);
    const rel = this.blendRelativeCloud(relSmooth, relRaw, speed);
    const tipRigid = rel.map(([x, y]) => [x + tx, y + ty]);
    const pierce = this.stickyPierce(this.holdTip);
    if (pierce && tipRigid[PIERCING_INDEX]) {
      tipRigid[PIERCING_INDEX] = [pierce.x, pierce.y];
    }

    this.overlay = {
      tip: this.holdTip,
      box: result.box,
      landmarks: tipRigid,
      side: result.side,
      pierce,
    };
    const ms = performance.now() - tPipe;
    this.pipeMsEma = this.pipeMsEma ? this.pipeMsEma * 0.85 + ms * 0.15 : ms;
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
    let best = Infinity;
    let bx = cx;
    let by = cy;
    const limX = x2 - x1;
    const limY = y2 - y1;
    for (let oy = 0; oy <= limY; oy++) {
      for (let ox = 0; ox <= limX; ox++) {
        let sad = 0;
        for (let py = 0; py < this.TIP_PATCH; py++) {
          for (let px = 0; px < this.TIP_PATCH; px++) {
            const i = ((oy + py) * rw + (ox + px)) * 4;
            const gv =
              0.299 * region.data[i] +
              0.587 * region.data[i + 1] +
              0.114 * region.data[i + 2];
            sad += Math.abs(gv - tpl[py * this.TIP_PATCH + px]);
          }
        }
        if (sad < best) {
          best = sad;
          bx = x1 + ox + half;
          by = y1 + oy + half;
        }
      }
    }
    let dx = bx - this.holdTip.x;
    let dy = by - this.holdTip.y;
    const step = Math.hypot(dx, dy);
    if (step > this.TIP_SEARCH) return this.holdTip;
    if (step < TIP_DEADZONE_PX) return this.holdTip;
    if (step > TIP_MAX_STEP_PX) {
      const s = TIP_MAX_STEP_PX / Math.max(step, 1e-6);
      dx *= s;
      dy *= s;
      bx = this.holdTip.x + dx;
      by = this.holdTip.y + dy;
    }
    if (!this.tipVel) this.tipVel = { x: 0, y: 0 };
    this.tipVel.x = 0.7 * this.tipVel.x + 0.3 * dx;
    this.tipVel.y = 0.7 * this.tipVel.y + 0.3 * dy;
    return {
      x: bx + this.tipVel.x * 0.18,
      y: by + this.tipVel.y * 0.18,
    };
  }

  paintVideo() {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh) return null;
    if (this.canvas.width !== vw || this.canvas.height !== vh) {
      this.canvas.width = vw;
      this.canvas.height = vh;
    }
    if (MIRROR_FEED) {
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
    return Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, v));
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

    if (
      !this.transferring &&
      (this.holdTip || this.tip) &&
      this.side &&
      this.geo &&
      (this.smoothRel || this.rawRel)
    ) {
      if (!this.tipPatch && this.holdTip)
        this.tipPatch = this.captureTipPatch(this.holdTip.x, this.holdTip.y);
      const prev = this.holdTip;
      const tracked = this.trackTipOnCanvas();
      if (tracked) {
        if (prev && this.geo) {
          this.geo = {
            cx: this.geo.cx + (tracked.x - prev.x),
            cy: this.geo.cy + (tracked.y - prev.y),
            side: this.geo.side,
          };
        }
        this.holdTip = tracked;
        this.tip = tracked;
      }
      this.applyTipHold(
        this.holdTip || this.tip,
        this.side,
        this.geo,
        vw,
        vh
      );
      this.tipSnap = false;
      if (this.frameIdx % 6 === 0 && this.holdTip)
        this.tipPatch = this.captureTipPatch(this.holdTip.x, this.holdTip.y);
    } else if (this.transferring) {
      this.overlay = null;
    }

    this.emitFrame(ts);

    const inferGap = (1000 / this.targetFps) * 0.9;
    if (
      !this.processing &&
      this.shgSession &&
      ts - (this.lastInferKickTs || 0) >= inferGap
    ) {
      this.lastInferKickTs = ts;
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
    this.lastDrawTs = ts;
    this.onDisplayTick(ts);
  };

  async startCamera() {
    if (this.live) return;
    if (!this.ready) {
      this.onStatus("Models not ready — load models first.");
      return;
    }
    const wantFps = this.targetFps;
    try {
      this.onStatus("Requesting camera…");
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: CAM_WIDTH },
          height: { ideal: CAM_HEIGHT },
          frameRate: { ideal: wantFps, min: CAM_FPS_MIN, max: CAM_FPS_MAX },
        },
      });
    } catch (e) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "user", width: { ideal: CAM_WIDTH }, height: { ideal: CAM_HEIGHT } },
        });
      } catch (e2) {
        this.onStatus(`Camera error: ${e2?.name || e2}`);
        throw e2;
      }
    }
    this.video.srcObject = this.stream;
    this.video.style.display = "none";
    await this.video.play();
    this.live = true;
    this.resetTracking();
    this.onStatus("Live — show a clear SIDE PROFILE of one ear");
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
