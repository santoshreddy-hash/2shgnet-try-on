/**
 * Browser live ear landmarks — aligned with desktop app.py:
 *
 *   webcam → mirror → YOLO ONNX tip → tip-centered full-ear crop
 *         → 2-SHGNet ONNX (flip LEFT; refine) → One Euro → overlay
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";
import { OneEuroLandmarks } from "./one_euro.js";
import { YoloPoseBrowser } from "./yolo_pose.js";
import { canvasRgbaToBgrChw, heatmapsToPointsSoft } from "./preprocess.js";

const SHGNET_URL = "/models/shgnet/SHGNet-56.onnx";
const YOLO_URL = "/models/yolo/yolo26n-pose.onnx";
const WASM_PATH = "/vendor/onnxruntime-web/dist/";

const CROP_PAD = 1.65;
const REFINE_PAD = 1.35;
const YOLO_EVERY = 2;
const SHG_EVERY = 2;
const YOLO_IMGSZ = 640;
const YOLO_CONF = 0.35;
const CAM_FPS_MIN = 20;
const CAM_FPS_MAX = 30;
const DT_FALLBACK = 1 / 30;
const MIRROR_FEED = true;
const NUM_LANDMARKS = 56;
const PIERCING_INDEX = 55;

const fpsSlider = document.getElementById("fpsSlider");
const fpsTargetVal = document.getElementById("fpsTargetVal");

function targetFps() {
  const v = Number(fpsSlider?.value);
  if (!Number.isFinite(v)) return 30;
  return Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, Math.round(v)));
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
let stream = null;
let live = false;
let processing = false;
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
let tipPatch = null; // {data, w, h, tw, th} grayscale template around tip
const TIP_PATCH = 11; // odd
const TIP_SEARCH = 18;
let geo = null; // {cx, cy, side}
let rawPts = null;
let lastBox = null;
let firstLock = true;
let overlay = null; // committed {tip, box, landmarks, side} — same frame

const snapCanvas = document.createElement("canvas");
const snapCtx = snapCanvas.getContext("2d", { willReadFrequently: true });

// Saved One Euro landmark values (match config.py / one_euro_settings.json)
const ONE_EURO_MIN_CUTOFF = 1.2;
const ONE_EURO_BETA = 0.25;
const ONE_EURO_D_CUTOFF = 1.19;
const smoother = new OneEuroLandmarks(
  NUM_LANDMARKS,
  ONE_EURO_MIN_CUTOFF,
  ONE_EURO_BETA,
  ONE_EURO_D_CUTOFF,
  0, // disable rest freeze — was causing landmarks to stick
  3,
  2.0
);

const cropCanvas = document.createElement("canvas");
cropCanvas.width = 256;
cropCanvas.height = 256;
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

const padCanvas = document.createElement("canvas");
const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });

function setStatus(msg) {
  statusEl.textContent = msg;
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
        <tr><td><strong>Active EP</strong></td><td><strong>wasm</strong></td><td>WASM only</td></tr>
      </tbody>
    </table>
  `;
}

let ortProxy = true;
function configureOrt(useProxy) {
  ort.env.wasm.wasmPaths = WASM_PATH;
  const canSAB =
    typeof SharedArrayBuffer !== "undefined" &&
    (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
  ort.env.wasm.numThreads = canSAB
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;
  // Prefer worker proxy so SHGNet never blocks the display loop
  ort.env.wasm.proxy = !!useProxy;
  ortProxy = !!useProxy;
}

async function createSession(url, label) {
  setStatus(`Loading ${label} (wasm${ortProxy ? "+proxy" : ""})…`);
  return ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

async function loadModels() {
  loadBtn.disabled = true;
  setStatus("Loading ONNX models (YOLO + SHGNet)…");
  try {
    configureOrt(true);
    let shg;
    let yolo;
    try {
      shg = await createSession(SHGNET_URL, "SHGNet ONNX (~26 MB)");
      yolo = await createSession(YOLO_URL, "YOLO pose ONNX (~12 MB)");
    } catch (proxyErr) {
      console.warn("ORT wasm.proxy failed; retrying without proxy", proxyErr);
      configureOrt(false);
      shg = await createSession(SHGNET_URL, "SHGNet ONNX (~26 MB)");
      yolo = await createSession(YOLO_URL, "YOLO pose ONNX (~12 MB)");
    }
    shgSession = shg;
    yoloPose = new YoloPoseBrowser(
      yolo,
      (data, dims) => new ort.Tensor("float32", data, dims),
      YOLO_IMGSZ,
      YOLO_CONF
    );
    setStatus(
      `Ready · YOLO + SHGNet ONNX · EP: wasm${ortProxy ? "+proxy" : ""}\n` +
        "Click Start live cam — allow camera when prompted."
    );
    updateButtons();
    reportSizes();
  } catch (e) {
    console.error(e);
    setStatus(`Load failed: ${e?.message || e}\nRetry Load models in Chrome/Edge.`);
    loadBtn.disabled = false;
    updateButtons();
  }
}

function pinnaHeight(yolo, vw, vh) {
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
    cands.length === 1
      ? cands[0]
      : cands[Math.floor(cands.length / 2)];
  if (tipNose != null && tipNose > 1) h = Math.min(h, tipNose * 0.7);
  return Math.max(40, Math.min(fmin * 0.2, h));
}

function isSideProfile(yolo) {
  if (!yolo.nose || yolo.nose.c < 0.2) return true;
  const dx = Math.abs(yolo.tip.x - yolo.nose.x);
  const d = Math.hypot(yolo.tip.x - yolo.nose.x, yolo.tip.y - yolo.nose.y);
  if (dx < 22 || d < 28) return false;
  if (yolo.earOtherConf != null && yolo.earConf != null) {
    if (yolo.earOtherConf >= 0.5 && yolo.earOtherConf >= yolo.earConf * 0.85)
      return false;
  }
  return true;
}

function sideLabel(side) {
  if (side === "LEFT") return "Left";
  if (side === "RIGHT") return "Right";
  return side || "—";
}

/** Anatomical side → on-screen label when preview is mirrored. */
function displayEarLabel(side) {
  if (!side) return "—";
  if (MIRROR_FEED) {
    if (side === "LEFT") return "Right";
    if (side === "RIGHT") return "Left";
  }
  return sideLabel(side);
}

function medial(yolo, tip, side, vw) {
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

function updateGeoFromYolo(yolo, vw, vh) {
  if (!isSideProfile(yolo)) return false;
  const tipPt = yolo.tip;
  const pinna = pinnaHeight(yolo, vw, vh);
  const sideLen = pinna * CROP_PAD;
  const [mx] = medial(yolo, tipPt, yolo.side, vw);
  const ncx = tipPt.x + mx * (0.1 * pinna);
  const ncy = tipPt.y + 0.06 * pinna;
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
  side = yolo.side;
  tip = tipPt;
  // Hold tip IS the YOLO tip — rawRel is always vs this tip (jewellery style)
  holdTip = { x: tipPt.x, y: tipPt.y };
  lastYoloTip = { x: tipPt.x, y: tipPt.y };
  tipPatch = null; // refresh after paint
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

function cropToTensor() {
  const img = cropCtx.getImageData(0, 0, 256, 256);
  return new ort.Tensor("float32", canvasRgbaToBgrChw(img), [1, 3, 256, 256]);
}

function landmarksOk(pts, tipPt, sidePx) {
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
  // Tip near landmark cloud (match app.py)
  const padX = 0.08 * bw;
  const padY = 0.08 * bh;
  if (tipPt.x < x0 - padX || tipPt.x > x1 + padX) return false;
  if (tipPt.y < y0 - padY || tipPt.y > y1 + padY) return false;
  return true;
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

function paintVideo() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  if (MIRROR_FEED) {
    ctx.save();
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, vw, vh);
    ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, vw, vh);
  }
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
    ctx.fillText("YOLO+SHGNet-56 ONNX (WASM)", 16, 42);
  }
}

async function runShg(needFlip, ox, oy, sidePx) {
  const t0 = performance.now();
  const out = await shgSession.run({
    [shgSession.inputNames[0]]: cropToTensor(),
  });
  const ms = performance.now() - t0;
  inferMsEma = inferMsEma ? inferMsEma * 0.85 + ms * 0.15 : ms;
  let pts256 = heatmapsToPointsSoft(out[shgSession.outputNames[0]], 256);
  const score = pts256.score ?? 0;
  if (needFlip) {
    pts256 = pts256.map(([x, y]) => [255 - x, y]);
  }
  const scale = sidePx / 256;
  const pts = pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
  return { pts, score };
}

async function inferFromSnapshot(source, vw, vh, tipPt, sideNow, geoNow, yolo) {
  if (!geoNow || !tipPt || !sideNow || !shgSession) return null;

  let { cx, cy, side: sideLen } = geoNow;
  const half = sideLen * 0.5;
  if (
    Math.abs(tipPt.x - cx) > half * 0.55 ||
    Math.abs(tipPt.y - cy) > half * 0.55
  ) {
    // Match desktop: re-center with medial offset when tip drifts
    if (yolo) {
      const pinna = sideLen / Math.max(CROP_PAD, 1e-3);
      const [mx] = medial(yolo, tipPt, sideNow, vw);
      cx = tipPt.x + mx * (0.1 * pinna);
      cy = tipPt.y + 0.06 * pinna;
    } else {
      cx = tipPt.x;
      cy = tipPt.y;
    }
    geo = { cx, cy, side: sideLen };
  }

  const preferFlip = sideNow === "LEFT";
  let { ox, oy, sidePx } = drawSquareCrop(source, cx, cy, sideLen, preferFlip);
  let box = [
    Math.max(0, Math.round(cx - sidePx * 0.5)),
    Math.max(0, Math.round(cy - sidePx * 0.5)),
    Math.min(vw, Math.round(cx + sidePx * 0.5)),
    Math.min(vh, Math.round(cy + sidePx * 0.5)),
  ];

  let { pts, score } = await runShg(preferFlip, ox, oy, sidePx);
  const lock = firstLock;
  const ok1 = landmarksOk(pts, tipPt, sidePx);
  // Flip retry only when needed (avoid 2× SHGNet every frame)
  if (lock || !ok1) {
    drawSquareCrop(source, cx, cy, sideLen, !preferFlip);
    const alt = await runShg(!preferFlip, ox, oy, sidePx);
    if (
      alt.score > score ||
      (lock && landmarksOk(alt.pts, tipPt, sidePx) && !ok1)
    ) {
      pts = alt.pts;
      score = alt.score;
    }
  }

  // Refine only on first lock (browser speed: skip ongoing refine)
  if (lock && landmarksOk(pts, tipPt, sidePx) && score > 0.07) {
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
        }
      }
    }
  }

  if (!(landmarksOk(pts, tipPt, sidePx) && score > 0.07)) return null;

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

async function processFrame(vw, vh, dt) {
  const tPipe = performance.now();
  // Match desktop: flip first, then YOLO + SHGNet in mirrored space
  snapCanvas.width = vw;
  snapCanvas.height = vh;
  if (MIRROR_FEED) {
    snapCtx.save();
    snapCtx.translate(vw, 0);
    snapCtx.scale(-1, 1);
    snapCtx.drawImage(video, 0, 0, vw, vh);
    snapCtx.restore();
  } else {
    snapCtx.drawImage(video, 0, 0, vw, vh);
  }

  const tick = inferTick++;
  // Stagger: never run YOLO + SHGNet on the same tick (WASM budget)
  const runYolo = tick % Math.max(1, YOLO_EVERY) === 0;
  const runShgNet = tick % Math.max(1, SHG_EVERY) === 1;
  let tipPt = tip;
  let sideNow = side;
  let geoNow = geo;

  if (runYolo && yoloPose) {
    const y = await yoloPose.detect(snapCanvas);
    if (y) {
      if (side && y.side !== side) {
        smoother.reset();
        rawPts = null;
        rawRel = null;
        firstLock = true;
        geo = null;
        overlay = null;
        holdTip = null;
        lastYoloTip = null;
        tipSnap = true;
      }
      lastYolo = y;
      if (!updateGeoFromYolo(y, vw, vh)) {
        setStatus("Turn head — clear SIDE PROFILE of one ear");
        return;
      }
      tipSnap = true;
      tipPt = tip;
      sideNow = side;
      geoNow = geo;
    }
  }

  if (!geoNow || !tipPt || !sideNow) return;

  // Tip-only frames: move cloud with YOLO tip (rawRel fixed)
  if (!runShgNet && rawRel && (holdTip || tipPt)) {
    applyTipHold(holdTip || tipPt, sideNow, geoNow, vw, vh, dt, tipSnap);
    tipSnap = false;
    const ms = performance.now() - tPipe;
    pipeMsEma = pipeMsEma ? pipeMsEma * 0.85 + ms * 0.15 : ms;
    return;
  }

  const result = await inferFromSnapshot(
    snapCanvas,
    vw,
    vh,
    tipPt,
    sideNow,
    geoNow,
    lastYolo
  );
  if (!result) return;

  rawPts = result.pts;
  // rawRel vs YOLO tip (same tip used in tip-hold) — points stick when tip moves
  const tx = result.tip.x;
  const ty = result.tip.y;
  holdTip = { x: tx, y: ty };
  tip = holdTip;
  rawRel = result.pts.map(([x, y]) => [x - tx, y - ty]);
  lastBox = result.box;

  const snap = firstLock;
  const sm = smoother.updateRelative(result.pts, holdTip, dt, result.side, {
    maxStepPx: 24,
    snap,
  });
  if (snap) firstLock = false;

  overlay = {
    tip: holdTip,
    box: result.box,
    landmarks: sm,
    side: result.side,
  };
  const ms = performance.now() - tPipe;
  pipeMsEma = pipeMsEma ? pipeMsEma * 0.85 + ms * 0.15 : ms;
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
 * SAD match of last tip patch in a small search window.
 */
function trackTipOnCanvas() {
  if (!holdTip || !tipPatch || !canvas.width) return holdTip;
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
  for (let oy = 0; oy <= limY; oy++) {
    for (let ox = 0; ox <= limX; ox++) {
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
  const dx = bx - holdTip.x;
  const dy = by - holdTip.y;
  if (Math.hypot(dx, dy) > TIP_SEARCH) return holdTip;
  if (Math.hypot(dx, dy) > 0.4) tipSnap = true;
  return { x: bx, y: by };
}
function applyTipHold(tipPt, sideNow, geoNow, vw, vh, dt, snap) {
  if (!rawRel || !tipPt || !sideNow || !geoNow) return false;
  const pts = rawRel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]);
  const sm = smoother.updateRelative(pts, tipPt, dt, sideNow, {
    maxStepPx: 24,
    snap: !!snap,
  });
  const sidePx = geoNow.side;
  overlay = {
    tip: { x: tipPt.x, y: tipPt.y },
    box: [
      Math.max(0, Math.round(geoNow.cx - sidePx * 0.5)),
      Math.max(0, Math.round(geoNow.cy - sidePx * 0.5)),
      Math.min(vw, Math.round(geoNow.cx + sidePx * 0.5)),
      Math.min(vh, Math.round(geoNow.cy + sidePx * 0.5)),
    ],
    landmarks: sm,
    side: sideNow,
  };
  return true;
}

function noteDisplayFps(ts) {
  if (lastDisplayTs) {
    const inst = 1000 / Math.max(1e-3, ts - lastDisplayTs);
    // True measured rAF/display FPS — no min floor, no max ceiling
    fpsEma = fpsEma ? fpsEma * 0.85 + inst * 0.15 : inst;
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
  const shownFps = fpsEma || 0;
  const epLabel = ortProxy ? `${ortEp}+proxy` : ortEp;
  drawHud(
    lmDisp,
    boxDisp,
    tipDisp,
    `LIVE ${shownFps.toFixed(0)}/${targetFps()} fps · pipe ${pipeMsEma.toFixed(0)} ms · ${epLabel}`
  );
  setStatus(
    `LIVE ${shownFps.toFixed(0)} FPS · target ${targetFps()} · pipe ${pipeMsEma.toFixed(0)} ms · SHG ${inferMsEma.toFixed(0)} ms (${epLabel})\n` +
      `Ear: ${earLabel} · 56 pts ${lmDisp ? "on" : "…"} · YOLO/${YOLO_EVERY} SHG/${SHG_EVERY}`
  );
}

/**
 * Every rAF: paint + tip-hold. Inference is fire-and-forget (never blocks HUD FPS).
 */
function onDisplayTick(ts) {
  if (!live) return;

  const frame = paintVideo();
  if (!frame) {
    noteDisplayFps(ts);
    paintOverlayHud();
    return;
  }
  const { vw, vh } = frame;
  // dt for One Euro only (cap large stalls); does not clamp reported FPS
  const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : DT_FALLBACK;
  lastTs = ts;
  frameIdx++;

  // Tip-hold every display tick: track tip on video, then glue landmarks to it
  if ((holdTip || tip) && side && geo && rawRel) {
    if (!tipPatch && holdTip) tipPatch = captureTipPatch(holdTip.x, holdTip.y);
    const prev = holdTip;
    const tracked = trackTipOnCanvas();
    if (tracked) {
      if (prev && geo) {
        geo = {
          cx: geo.cx + (tracked.x - prev.x),
          cy: geo.cy + (tracked.y - prev.y),
          side: geo.side,
        };
      }
      holdTip = tracked;
      tip = tracked;
    }
    applyTipHold(holdTip || tip, side, geo, vw, vh, dt, tipSnap);
    tipSnap = false;
    // Refresh patch occasionally so lighting changes don't break track
    if (frameIdx % 8 === 0 && holdTip) tipPatch = captureTipPatch(holdTip.x, holdTip.y);
  }

  noteDisplayFps(ts);
  paintOverlayHud();

  // Fire-and-forget inference — never await here
  if (!processing && shgSession) {
    processing = true;
    processFrame(vw, vh, dt)
      .catch((e) => setStatus(`Infer error: ${e?.message || e}`))
      .finally(() => {
        processing = false;
      });
  }
}

function loopLive(ts) {
  if (!live) return;
  rafId = requestAnimationFrame(loopLive);

  const interval = 1000 / targetFps();
  if (lastDrawTs && ts - lastDrawTs < interval) return;
  lastDrawTs = lastDrawTs
    ? ts - ((ts - lastDrawTs) % interval)
    : ts;

  onDisplayTick(ts);
}

function scheduleVideoFrames() {
  // Full rAF display clock — independent of slow WASM inference
  if (!rafId) rafId = requestAnimationFrame(loopLive);
}

async function applyCamFpsConstraint(idealFps) {
  if (!stream) return;
  const track = stream.getVideoTracks()[0];
  if (!track) return;
  const fps = Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, idealFps));
  try {
    await track.applyConstraints({
      frameRate: { ideal: fps, max: fps },
    });
  } catch (_) {
    /* optional — display throttle still applies */
  }
}

async function startCamera() {
  if (live) return;
  if (!shgSession || !yoloPose) {
    setStatus("Models not ready — wait for Ready or click Load models.");
    return;
  }
  const wantFps = targetFps();
  try {
    setStatus("Requesting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 540 },
        frameRate: { ideal: wantFps, min: CAM_FPS_MIN, max: CAM_FPS_MAX },
      },
    });
  } catch (e) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 960 },
          height: { ideal: 540 },
        },
      });
    } catch (e2) {
      setStatus(`Camera error: ${e2?.name || e2}`);
      return;
    }
  }
  video.srcObject = stream;
  video.style.display = "none";
  await video.play();
  await applyCamFpsConstraint(wantFps);
  try {
    const track = stream.getVideoTracks()[0];
    const s = track.getSettings?.() || {};
    console.log(
      `[cam] feed ${s.width}x${s.height} @ ${s.frameRate} fps (target ${wantFps})`
    );
  } catch (_) {
    /* settings optional */
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
  frameIdx = 0;
  inferTick = 0;
  smoother.reset();
  lastTs = 0;
  lastDisplayTs = 0;
  lastDrawTs = 0;
  pipeMsEma = 0;
  rawRel = null;
  fpsEma = 0;
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
  updateButtons();
  setStatus("Camera stopped.");
}

loadBtn.addEventListener("click", () => loadModels());
startCamBtn.addEventListener("click", () =>
  startCamera().catch((e) => setStatus(String(e)))
);
stopCamBtn.addEventListener("click", () => stopCamera());
window.addEventListener("beforeunload", () => stopCamera());

if (fpsSlider && fpsTargetVal) {
  fpsTargetVal.textContent = String(targetFps());
  fpsSlider.addEventListener("input", () => {
    const fps = targetFps();
    fpsTargetVal.textContent = String(fps);
    lastDrawTs = 0;
    if (live) applyCamFpsConstraint(fps);
  });
}

reportSizes();
updateButtons();
loadModels();
