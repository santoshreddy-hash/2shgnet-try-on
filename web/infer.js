/**
 * Browser live ear landmarks — one Web Worker runs YOLO + SHGNet.
 * UI thread: tip-rigid glue + Lucas–Kanade stick (desktop parity).
 */
import { OneEuroLandmarks } from "./one_euro.js";
import {
  rgbaToGray,
  trackTipLk,
  LandmarkStickTracker,
} from "./lk_track.js";

const SHGNET_URL = "/models/shgnet/SHGNet-56.onnx";
const YOLO_URL = "/models/yolo/yolo26n-pose.onnx";
const WASM_PATH = "/vendor/onnxruntime-web/dist/";
const WORKER_URL = new URL("./infer_worker.js", import.meta.url);

const YOLO_IMGSZ = 640;
const YOLO_CONF = 0.22;
const CAM_FPS_MIN = 20;
const CAM_FPS_MAX = 30;
const CAM_FPS_DEFAULT = 25;
const DT_FALLBACK = 1 / CAM_FPS_DEFAULT;
const MIRROR_FEED = true;
const NUM_LANDMARKS = 56;
const PIERCING_INDEX = 55;
const CAM_WIDTH = 960;
const CAM_HEIGHT = 540;
const YOLO_LOST_MAX = 8;
/** WASM SHG ~400ms — run every N frames; YOLO tip every frame keeps glue accurate */
const SHG_EVERY = 3;

const statusEl = document.getElementById("status");
const sizesEl = document.getElementById("sizes");
const loadBtn = document.getElementById("loadModel");
const startCamBtn = document.getElementById("startCam");
const stopCamBtn = document.getElementById("stopCam");
const video = document.getElementById("video");
const canvas = document.getElementById("out");
const ctx = canvas.getContext("2d", { alpha: false });
const fpsSlider = document.getElementById("fpsSlider");
const fpsTargetVal = document.getElementById("fpsTargetVal");

const snapCanvas = document.createElement("canvas");
const snapCtx = snapCanvas.getContext("2d", { willReadFrequently: true });

const ONE_EURO_DEFAULTS = {
  min_cutoff: 1.8,
  beta: 0.85,
  d_cutoff: 1.19,
  rest_speed_px: 1.5,
  rest_hold_frames: 2,
  rest_release_mult: 1.5,
  max_step_px: 42.0,
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

let inferWorker = null;
let workerThreads = 1;
let modelsReady = false;
let stream = null;
let live = false;
let processing = false;
let rafId = 0;
let inferReqId = 0;
let lastTs = 0;
let lastDrawTs = 0;
let lastDisplayTs = 0;
let lastInferKickTs = 0;
let fpsEma = 0;
let pipeMsEma = 0;
let inferMsEma = 0;
let frameIdx = 0;

let side = null;
let tip = null;
let holdTip = null;
let geo = null;
let rawRel = null;
let smoothRel = null;
let firstLock = true;
let transferring = false;
let yoloLost = 0;
let overlay = null;
let ptsGen = 0;
let shgTick = 0;
const stick = new LandmarkStickTracker(20);
let prevGray = null;
let grayW = 0;
let grayH = 0;

function targetFps() {
  const v = Number(fpsSlider?.value);
  const raw = Number.isFinite(v) ? v : CAM_FPS_DEFAULT;
  return Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, Math.round(raw)));
}

function clampFps(v) {
  if (!Number.isFinite(v) || v <= 0) return 0;
  return Math.max(CAM_FPS_MIN, Math.min(CAM_FPS_MAX, v));
}

function oneEuroMaxStep() {
  return Math.max(1, Number(oneEuroCfg.max_step_px) || 42);
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateButtons() {
  startCamBtn.disabled = !(modelsReady && !live);
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
        <tr><td><strong>Active EP</strong></td><td><strong>wasm+worker</strong></td><td>${workerThreads} thread(s)</td></tr>
      </tbody>
    </table>
  `;
}

async function loadOneEuroSettings() {
  try {
    const res = await fetch("/one_euro_settings.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    oneEuroCfg = { ...ONE_EURO_DEFAULTS, ...data };
    smoother.applySettings(oneEuroCfg);
  } catch {
    smoother.applySettings(ONE_EURO_DEFAULTS);
    oneEuroCfg = { ...ONE_EURO_DEFAULTS };
  }
}

function beginTransfer() {
  transferring = true;
  side = null;
  tip = null;
  holdTip = null;
  geo = null;
  rawRel = null;
  smoothRel = null;
  overlay = null;
  firstLock = true;
  ptsGen = 0;
  prevGray = null;
  stick.reset();
  smoother.reset();
}

function clearEarLock() {
  beginTransfer();
  yoloLost = 0;
}

function loadModelsViaWorker() {
  return new Promise((resolve, reject) => {
    if (inferWorker) {
      try {
        inferWorker.terminate();
      } catch (_) {}
    }
    const w = new Worker(WORKER_URL, { type: "module" });
    inferWorker = w;
    const onMsg = (ev) => {
      const msg = ev.data || {};
      if (msg.type === "ready") {
        workerThreads = msg.threads || 1;
        w.removeEventListener("message", onMsg);
        resolve(w);
        return;
      }
      if (msg.type === "error" && !msg.id) {
        w.removeEventListener("message", onMsg);
        reject(new Error(msg.message || "Worker init failed"));
      }
    };
    w.addEventListener("message", onMsg);
    w.addEventListener("error", (e) => {
      reject(new Error(e.message || "Worker error"));
    });
    w.postMessage({
      type: "init",
      shgUrl: SHGNET_URL,
      yoloUrl: YOLO_URL,
      wasmPath: WASM_PATH,
      imgsz: YOLO_IMGSZ,
      conf: YOLO_CONF,
    });
  });
}

async function loadModels() {
  loadBtn.disabled = true;
  setStatus("Loading ONNX in 1 Web Worker (YOLO + SHGNet)…");
  try {
    await loadOneEuroSettings();
    await loadModelsViaWorker();
    modelsReady = true;
    setStatus(
      `Ready · YOLO + SHGNet in 1 Web Worker · ${workerThreads} WASM thread(s)\n` +
        "Click Start live cam — allow camera when prompted."
    );
    updateButtons();
    reportSizes();
  } catch (e) {
    console.error(e);
    modelsReady = false;
    setStatus(`Load failed: ${e?.message || e}\nRetry Load models in Chrome/Edge.`);
    loadBtn.disabled = false;
    updateButtons();
  }
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

function drawHud(landmarks, box, tipPt, info) {
  if (box) {
    const [x1, y1, x2, y2] = box;
    ctx.strokeStyle = "rgba(80, 200, 120, 0.95)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (tipPt) {
    ctx.fillStyle = "rgb(0,140,255)";
    ctx.beginPath();
    ctx.arc(tipPt.x, tipPt.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  if (landmarks) {
    ctx.fillStyle = "rgb(0,220,255)";
    for (let i = 0; i < landmarks.length; i++) {
      if (i === PIERCING_INDEX) continue;
      const [x, y] = landmarks[i];
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    if (landmarks[PIERCING_INDEX]) {
      const [px, py] = landmarks[PIERCING_INDEX];
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
    ctx.fillRect(8, 8, 320, 40);
    ctx.fillStyle = "#f0f0f0";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(info, 16, 26);
    ctx.fillText("YOLO+SHGNet-56 · 1 Web Worker", 16, 42);
  }
}

/** Tip-rigid display — landmarks stay in an ear disk around the tip only. */
function applyTipHold(tipPt, sideNow, geoNow, vw, vh) {
  const rel = rawRel || smoothRel;
  if (!rel || !tipPt || !sideNow || !geoNow) return false;
  if (transferring || firstLock) {
    overlay = null;
    return false;
  }
  const sidePx = Math.max(40, geoNow.side);
  // Always center the ear box on the live tip (never a stale geo that pulls dots to face)
  const cx = tipPt.x;
  const cy = tipPt.y;
  const maxR = sidePx * 0.48;
  let far = 0;
  const landmarks = rel.map(([x, y]) => {
    let rx = x;
    let ry = y;
    const r = Math.hypot(rx, ry);
    if (r > maxR && r > 1e-6) {
      const s = maxR / r;
      rx *= s;
      ry *= s;
      far += 1;
    }
    return [rx + tipPt.x, ry + tipPt.y];
  });
  // Corrupted / face-sized cloud → hide
  if (far > landmarks.length * 0.35) {
    overlay = null;
    return false;
  }
  const box = [
    Math.max(0, Math.round(cx - sidePx * 0.5)),
    Math.max(0, Math.round(cy - sidePx * 0.5)),
    Math.min(vw, Math.round(cx + sidePx * 0.5)),
    Math.min(vh, Math.round(cy + sidePx * 0.5)),
  ];
  overlay = {
    tip: { x: tipPt.x, y: tipPt.y },
    box,
    landmarks,
    side: sideNow,
  };
  return true;
}

function currentCropBox(vw, vh) {
  if (!geo) return null;
  const sidePx = geo.side;
  return [
    Math.max(0, Math.round(geo.cx - sidePx * 0.5)),
    Math.max(0, Math.round(geo.cy - sidePx * 0.5)),
    Math.min(vw, Math.round(geo.cx + sidePx * 0.5)),
    Math.min(vh, Math.round(geo.cy + sidePx * 0.5)),
  ];
}

/** Re-anchor stick cloud to tip so landmarks never leave the ear tip. */
function reanchorStickToTip(stuck, tipPt) {
  if (!stuck || !tipPt || !geo) return null;
  const maxR = Math.max(40, geo.side) * 0.5;
  let far = 0;
  const next = stuck.map(([x, y]) => {
    const dx = x - tipPt.x;
    const dy = y - tipPt.y;
    if (Math.hypot(dx, dy) > maxR) far += 1;
    return [dx, dy];
  });
  // Stick drifted onto face — drop stick, keep previous tip-relative shape
  if (far > stuck.length * 0.25) {
    stick.reset();
    return null;
  }
  rawRel = next;
  const pts = rawRel.map(([x, y]) => [x + tipPt.x, y + tipPt.y]);
  stick.absPts = pts.map((p) => [p[0], p[1]]);
  return pts;
}

/** Fresh YOLO tip while SHG still running — tip-rigid glue, no shape rewrite. */
function applyTipUpdate(msg) {
  if (!msg?.tip) return;
  const vw = canvas.width || 1;
  const vh = canvas.height || 1;
  const newSide = msg.side;
  const newTip = { x: msg.tip.x, y: msg.tip.y };
  if (side && newSide && newSide !== side) {
    stick.reset();
    smoother.reset();
    rawRel = null;
    smoothRel = null;
    firstLock = true;
    transferring = true;
    overlay = null;
  }
  const prev = holdTip;
  if (
    prev &&
    geo &&
    Math.hypot(newTip.x - prev.x, newTip.y - prev.y) > Math.max(40, geo.side * 0.35)
  ) {
    // Tip jumped (face flick / false keypoint) — clear shape, don't trail across face
    stick.reset();
    smoother.reset();
    rawRel = null;
    smoothRel = null;
    firstLock = true;
    transferring = true;
    overlay = null;
  }
  if (msg.geo) {
    // Snap geo to tip-centered crop from YOLO (no lagging box on the face)
    geo = {
      cx: newTip.x + (msg.geo.cx - msg.tip.x),
      cy: newTip.y + (msg.geo.cy - msg.tip.y),
      side: msg.geo.side,
    };
  } else if (geo && prev) {
    geo = {
      cx: geo.cx + (newTip.x - prev.x),
      cy: geo.cy + (newTip.y - prev.y),
      side: geo.side,
    };
  }
  side = newSide || side;
  tip = newTip;
  holdTip = newTip;
  if (rawRel && !transferring && !firstLock) {
    applyTipHold(holdTip, side, geo, vw, vh);
  } else {
    overlay = null;
  }
}

function applyWorkerResult(msg, dt) {
  const pipeMs = Number(msg.pipeMs) || 0;
  if (pipeMs) pipeMsEma = pipeMsEma ? pipeMsEma * 0.85 + pipeMs * 0.15 : pipeMs;
  if (msg.shgMs != null && Number(msg.shgMs) > 0) {
    inferMsEma = inferMsEma
      ? inferMsEma * 0.85 + Number(msg.shgMs) * 0.15
      : Number(msg.shgMs);
  }

  const vw = canvas.width || 1;
  const vh = canvas.height || 1;

  if (msg.type === "error") {
    setStatus(`Worker error: ${msg.message || "unknown"}`);
    return;
  }

  if (msg.type === "miss") {
    if (msg.reason === "busy") return;
    if (msg.reason === "not_profile" || msg.reason === "no_ear") {
      beginTransfer();
      overlay = null;
      yoloLost += 1;
      if (yoloLost > YOLO_LOST_MAX) clearEarLock();
      if (msg.reason === "not_profile") {
        setStatus("Front face / mid-turn — landmarks hidden · show SIDE PROFILE");
      }
      return;
    }
    // bad_shg: keep tip-rigid hold if we already locked (desktop path)
    if (msg.reason === "bad_shg") {
      yoloLost += 1;
      if (msg.tip && rawRel && !transferring && !firstLock) {
        applyTipUpdate(msg);
      } else if (transferring || firstLock) {
        overlay = null;
      }
      if (yoloLost > YOLO_LOST_MAX) clearEarLock();
      return;
    }
    yoloLost += 1;
    if (yoloLost > YOLO_LOST_MAX) clearEarLock();
    return;
  }

  yoloLost = 0;

  // YOLO-only frame: tip already applied via tip msg; nothing else
  if (msg.tipOnly || !msg.pts) {
    applyTipUpdate(msg);
    return;
  }

  const newSide = msg.side;
  const shgTip = { x: msg.tip.x, y: msg.tip.y };
  const sideSwitched = side && newSide && newSide !== side;
  if (sideSwitched) {
    stick.reset();
    smoother.reset();
    rawRel = null;
    smoothRel = null;
    firstLock = true;
    transferring = true;
    overlay = null;
  }

  if (msg.geo) {
    if (geo && !firstLock && !transferring) {
      geo = {
        cx: 0.85 * geo.cx + 0.15 * msg.geo.cx,
        cy: 0.85 * geo.cy + 0.15 * msg.geo.cy,
        side: 0.85 * geo.side + 0.15 * msg.geo.side,
      };
    } else {
      geo = msg.geo;
    }
  }
  side = newSide;

  // Critical: SHG tip is ~400ms stale — keep live tip (interim/LK), only refresh shape
  const liveTip =
    holdTip && !firstLock && !transferring && !sideSwitched
      ? { x: holdTip.x, y: holdTip.y }
      : shgTip;
  tip = liveTip;
  holdTip = { x: liveTip.x, y: liveTip.y };

  const snap = firstLock || transferring || sideSwitched;
  const sm = smoother.updateRelative(msg.pts, shgTip, dt, newSide, {
    maxStepPx: oneEuroMaxStep(),
    snap,
  });
  let ptsAtShg = snap ? msg.pts.map(([x, y]) => [x, y]) : sm;
  // Shape relative to SHG-frame tip, displayed at live tip
  rawRel = ptsAtShg.map(([x, y]) => [x - shgTip.x, y - shgTip.y]);
  let ptsDraw = rawRel.map(([x, y]) => [x + liveTip.x, y + liveTip.y]);
  ptsGen += 1;

  if (prevGray && grayW === vw && grayH === vh) {
    const stuck = stick.update(
      prevGray,
      grayW,
      grayH,
      ptsDraw,
      ptsGen,
      currentCropBox(vw, vh)
    );
    if (stuck) ptsDraw = reanchorStickToTip(stuck, liveTip) || ptsDraw;
  } else {
    stick.reset();
  }

  smoothRel = rawRel.map(([x, y]) => [x, y]);
  firstLock = false;
  transferring = false;
  applyTipHold(holdTip, side, geo, vw, vh);
}

async function processFrame(vw, vh, dt) {
  if (!inferWorker || processing) return;
  processing = true;
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

  let bitmap;
  try {
    bitmap = await createImageBitmap(snapCanvas);
  } catch (e) {
    processing = false;
    console.warn("createImageBitmap failed", e);
    return;
  }

  const id = ++inferReqId;
  const prefer = transferring ? null : side;
  const wantLock = firstLock || transferring;
  shgTick += 1;
  const wantShg = wantLock || shgTick % SHG_EVERY === 0;

  await new Promise((resolve) => {
    const onMsg = (ev) => {
      const msg = ev.data || {};
      if (msg.id !== id) return;
      if (msg.type === "tip") {
        applyTipUpdate(msg);
        return;
      }
      if (msg.type !== "result" && msg.type !== "miss" && msg.type !== "error")
        return;
      inferWorker.removeEventListener("message", onMsg);
      applyWorkerResult(msg, dt);
      resolve();
    };
    inferWorker.addEventListener("message", onMsg);
    try {
      inferWorker.postMessage(
        {
          type: "infer",
          id,
          bitmap,
          preferSide: prefer,
          firstLock: wantLock,
          wantShg,
        },
        [bitmap]
      );
    } catch (e) {
      inferWorker.removeEventListener("message", onMsg);
      if (bitmap?.close) bitmap.close();
      setStatus(`Worker post failed: ${e?.message || e}`);
      resolve();
    }
  });
  processing = false;
}

function paintOverlayHud() {
  const want = targetFps();
  const shownFps = fpsEma ? clampFps(fpsEma) : 0;
  const ear = overlay?.side || side || "…";
  drawHud(
    overlay?.landmarks || null,
    overlay?.box || null,
    overlay?.tip || null,
    `LIVE ${shownFps.toFixed(0)}/${want} fps · pipe ${pipeMsEma.toFixed(0)} ms · ${ear}`
  );
  setStatus(
    `LIVE ${shownFps.toFixed(0)} FPS · target ${want} · pipe ${pipeMsEma.toFixed(0)} ms · SHG ${inferMsEma.toFixed(0)} ms (wasm+worker×${workerThreads})\n` +
      `Ear: ${ear} · 56 pts ${overlay?.landmarks ? "on" : transferring ? "switching…" : "…"} · YOLO/1 SHG/${SHG_EVERY} · 1 worker`
  );
}

function onDisplayTick(ts) {
  if (!live) return;
  const frame = paintVideo();
  if (!frame) {
    paintOverlayHud();
    return;
  }
  const { vw, vh } = frame;
  const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : DT_FALLBACK;
  lastTs = ts;
  frameIdx++;

  if (lastDisplayTs) {
    const inst = 1000 / Math.max(1e-3, ts - lastDisplayTs);
    fpsEma = clampFps(fpsEma ? fpsEma * 0.85 + inst * 0.15 : inst);
  }
  lastDisplayTs = ts;

  // Gray buffer for Lucas–Kanade (desktop track_tip_lk + LandmarkStickTracker)
  let gray = null;
  try {
    const img = ctx.getImageData(0, 0, vw, vh);
    const g = rgbaToGray(img);
    gray = g.gray;
    grayW = g.width;
    grayH = g.height;
  } catch (_) {
    gray = null;
  }

  if (!transferring && holdTip && side && geo && rawRel) {
    if (gray && prevGray && prevGray.length === gray.length) {
      const { tip: tracked, moved } = trackTipLk(
        prevGray,
        gray,
        grayW,
        grayH,
        holdTip
      );
      if (moved) {
        geo = {
          cx: geo.cx + (tracked.x - holdTip.x),
          cy: geo.cy + (tracked.y - holdTip.y),
          side: geo.side,
        };
        holdTip = tracked;
        tip = tracked;
      }
    }
    const stuck = gray
      ? stick.update(
          gray,
          grayW,
          grayH,
          null,
          ptsGen,
          currentCropBox(vw, vh)
        )
      : null;
    if (stuck) reanchorStickToTip(stuck, holdTip);
    applyTipHold(holdTip, side, geo, vw, vh);
  } else if (transferring) {
    overlay = null;
  }

  if (gray) prevGray = gray;

  paintOverlayHud();

  if (!processing && modelsReady) {
    lastInferKickTs = ts;
    processFrame(vw, vh, dt).catch((e) =>
      setStatus(`Infer error: ${e?.message || e}`)
    );
  }
}

function loopLive(ts) {
  if (!live) return;
  rafId = requestAnimationFrame(loopLive);
  const want = targetFps();
  const interval = 1000 / want;
  if (lastDrawTs && ts - lastDrawTs < interval - 0.5) return;
  lastDrawTs = ts;
  onDisplayTick(ts);
}

async function startCamera() {
  if (live) return;
  if (!modelsReady) {
    setStatus("Models not ready — click Load models.");
    return;
  }
  const wantFps = targetFps();
  try {
    setStatus("Requesting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
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
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: CAM_WIDTH },
          height: { ideal: CAM_HEIGHT },
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
  live = true;
  clearEarLock();
  frameIdx = 0;
  lastTs = 0;
  lastDisplayTs = 0;
  lastDrawTs = 0;
  lastInferKickTs = 0;
  pipeMsEma = 0;
  fpsEma = 0;
  updateButtons();
  rafId = requestAnimationFrame(loopLive);
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
  setStatus("Camera stopped.");
  updateButtons();
}

fpsSlider?.addEventListener("input", () => {
  const v = targetFps();
  if (fpsTargetVal) fpsTargetVal.textContent = String(v);
});

loadBtn.addEventListener("click", () => loadModels());
startCamBtn.addEventListener("click", () => startCamera());
stopCamBtn.addEventListener("click", () => stopCamera());

updateButtons();
loadModels();
