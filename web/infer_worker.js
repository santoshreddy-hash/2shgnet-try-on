/**
 * Inference Web Worker — YOLO + SHGNet run entirely off the UI thread.
 * Uses threaded WASM (SharedArrayBuffer) when available; no ORT proxy nesting.
 *
 * Protocol:
 *   main → { type:'init', shgUrl, yoloUrl, wasmPath, imgsz, conf }
 *   ← { type:'ready', threads } | { type:'error', message }
 *   main → { type:'infer', id, bitmap, preferSide, firstLock }
 *   ← { type:'result', id, ... } | { type:'miss', id, reason, pipeMs }
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";
import { canvasRgbaToBgrChw, heatmapsToPointsSoft } from "./preprocess.js";

const CROP_PAD = 1.55;
const LEFT_EAR = 3;
const RIGHT_EAR = 4;
const NOSE = 0;
const LEFT_EYE = 1;
const RIGHT_EYE = 2;
const EAR_MIN_CONF = 0.3;

let shgSession = null;
let yoloSession = null;
let yoloInput = null;
let shgInput = null;
let imgsz = 640;
let conf = 0.22;
let preferSide = null;
let busy = false;

const yoloCanvas = new OffscreenCanvas(640, 640);
const yoloCtx = yoloCanvas.getContext("2d", { willReadFrequently: true });
const padCanvas = new OffscreenCanvas(256, 256);
const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });
const cropCanvas = new OffscreenCanvas(256, 256);
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

function configureOrt(wasmPath) {
  ort.env.wasm.wasmPaths = wasmPath;
  const canSAB =
    typeof SharedArrayBuffer !== "undefined" &&
    (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
  const threads = canSAB
    ? Math.min(4, self.navigator?.hardwareConcurrency || 2)
    : 1;
  // Run WASM in THIS worker with threads — do not nest ORT proxy workers
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false;
  return threads;
}

function isSideProfile(ear, other, nose) {
  // Match train/crop.is_side_profile — reject front / mid-turn only
  if (ear.c < 0.40) return false;
  if (other.c >= 0.38 && other.c >= ear.c * 0.72) return false;
  if (nose?.c >= 0.2) {
    const dx = Math.abs(ear.x - nose.x);
    const d = Math.hypot(ear.x - nose.x, ear.y - nose.y);
    if (dx < 28 || d < 36) return false;
  }
  return true;
}

function pickEar(left, right, nose) {
  const prefer = (preferSide || "").toUpperCase();
  const cands = [];
  for (const [ear, other, name] of [
    [left, right, "LEFT"],
    [right, left, "RIGHT"],
  ]) {
    if (!isSideProfile(ear, other, nose)) continue;
    let score = ear.c;
    if (prefer === name) score += 0.08;
    cands.push({ score, side: name, ear, other });
  }
  if (!cands.length) {
    for (const [ear, other, name] of [
      [left, right, "LEFT"],
      [right, left, "RIGHT"],
    ]) {
      if (ear.c < EAR_MIN_CONF) continue;
      let score = ear.c;
      if (prefer === name) score += 0.05;
      cands.push({ score, side: name, ear, other });
    }
  }
  if (!cands.length) return null;
  cands.sort((a, b) => b.score - a.score);
  return cands[0];
}

function parseYolo(arr, dims, padX, padY, r) {
  const un = (x, y) => [(x - padX) / r, (y - padY) / r];
  let best = null;
  if (dims.length === 3 && dims[1] === 56) {
    const nDet = dims[2];
    const get = (c, i) => arr[c * nDet + i];
    let bestOff = -1;
    let bestScore = conf;
    for (let i = 0; i < nDet; i++) {
      const score = get(4, i);
      if (score > bestScore) {
        bestScore = score;
        bestOff = i;
      }
    }
    if (bestOff >= 0) {
      const cx = get(0, bestOff);
      const cy = get(1, bestOff);
      const bw = get(2, bestOff);
      const bh = get(3, bestOff);
      const [x1, y1] = un(cx - bw * 0.5, cy - bh * 0.5);
      const [x2, y2] = un(cx + bw * 0.5, cy + bh * 0.5);
      const kpt = (idx) => {
        const base = 5 + idx * 3;
        const [x, y] = un(get(base, bestOff), get(base + 1, bestOff));
        return { x, y, c: get(base + 2, bestOff) };
      };
      best = { kpt, bbox: [x1, y1, x2, y2], conf: bestScore };
    }
  } else {
    let nDet = 300;
    let get = (i, c) => arr[i * 57 + c];
    if (dims.length === 3 && dims[1] === 57) {
      nDet = dims[2];
      get = (i, c) => arr[c * nDet + i];
    } else if (dims.length === 3 && dims[2] === 57) {
      nDet = dims[1];
      get = (i, c) => arr[i * 57 + c];
    }
    let bestOff = -1;
    let bestScore = conf;
    for (let i = 0; i < nDet; i++) {
      const score = get(i, 4);
      if (score > bestScore) {
        bestScore = score;
        bestOff = i;
      }
    }
    if (bestOff >= 0) {
      const [x1, y1] = un(get(bestOff, 0), get(bestOff, 1));
      const [x2, y2] = un(get(bestOff, 2), get(bestOff, 3));
      const kpt = (idx) => {
        const base = 6 + idx * 3;
        const [x, y] = un(get(bestOff, base), get(bestOff, base + 1));
        return { x, y, c: get(bestOff, base + 2) };
      };
      best = { kpt, bbox: [x1, y1, x2, y2], conf: bestScore };
    }
  }
  if (!best) return null;
  const left = best.kpt(LEFT_EAR);
  const right = best.kpt(RIGHT_EAR);
  const nose = best.kpt(NOSE);
  const leftEye = best.kpt(LEFT_EYE);
  const rightEye = best.kpt(RIGHT_EYE);
  const picked = pickEar(left, right, nose);
  if (!picked) return null;
  let eyeDist = null;
  if (leftEye.c >= 0.2 && rightEye.c >= 0.2) {
    eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
  }
  preferSide = picked.side;
  return {
    side: picked.side,
    tip: { x: picked.ear.x, y: picked.ear.y },
    bbox: best.bbox,
    nose,
    leftEye,
    rightEye,
    eyeDist,
    earConf: picked.ear.c,
    earOtherConf: picked.other.c,
    conf: best.conf,
  };
}

async function runYolo(bitmap, vw, vh) {
  const r = Math.min(imgsz / vh, imgsz / vw);
  const nw = Math.round(vw * r);
  const nh = Math.round(vh * r);
  const padX = (imgsz - nw) / 2;
  const padY = (imgsz - nh) / 2;
  if (yoloCanvas.width !== imgsz) {
    yoloCanvas.width = imgsz;
    yoloCanvas.height = imgsz;
  }
  yoloCtx.fillStyle = "#727272";
  yoloCtx.fillRect(0, 0, imgsz, imgsz);
  yoloCtx.drawImage(bitmap, padX, padY, nw, nh);
  const { data } = yoloCtx.getImageData(0, 0, imgsz, imgsz);
  const plane = imgsz * imgsz;
  const chw = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    chw[p] = data[i] / 255;
    chw[plane + p] = data[i + 1] / 255;
    chw[2 * plane + p] = data[i + 2] / 255;
  }
  const t0 = performance.now();
  const out = await yoloSession.run({
    [yoloInput]: new ort.Tensor("float32", chw, [1, 3, imgsz, imgsz]),
  });
  const yoloMs = performance.now() - t0;
  const tensorOut = out[yoloSession.outputNames[0]];
  const det = parseYolo(tensorOut.data, tensorOut.dims || [], padX, padY, r);
  return { det, yoloMs };
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
  let h = cands.length === 1 ? cands[0] : cands[Math.floor(cands.length / 2)];
  if (tipNose != null && tipNose > 1) h = Math.min(h, tipNose * 0.7);
  return Math.max(40, Math.min(fmin * 0.2, h));
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

function profileOk(yolo) {
  // Same thresholds as train/crop.is_side_profile
  if (yolo.earOtherConf != null && yolo.earConf != null) {
    if (yolo.earConf < 0.40) return false;
    if (yolo.earOtherConf >= 0.38 && yolo.earOtherConf >= yolo.earConf * 0.72)
      return false;
  }
  // Reject eye-as-ear: tip must not sit on an eye keypoint
  for (const eye of [yolo.leftEye, yolo.rightEye]) {
    if (!eye || eye.c < 0.25) continue;
    const dEye = Math.hypot(yolo.tip.x - eye.x, yolo.tip.y - eye.y);
    if (dEye < 32) return false;
  }
  if (!yolo.nose || yolo.nose.c < 0.2) return true;
  const dx = Math.abs(yolo.tip.x - yolo.nose.x);
  const d = Math.hypot(yolo.tip.x - yolo.nose.x, yolo.tip.y - yolo.nose.y);
  // Mid-turn / frontal: tip too close to nose
  if (dx < 28 || d < 36) return false;
  // Tip should be clearly lateral vs nose (side profile)
  if (dx < d * 0.45) return false;
  return true;
}

function landmarksOk(pts, tipPt, sidePx, cropBox) {
  // Match train/crop.landmarks_ok + ear-only containment
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
  const span = Math.max(bw, bh, 1);
  const ratio = span / Math.max(1, sidePx);
  if (ratio < 0.35 || ratio > 0.88) return false;
  if (Math.min(bw, bh) < span * 0.25) return false;
  let sx = 0,
    sy = 0;
  for (const [x, y] of pts) {
    sx += x;
    sy += y;
  }
  const mx = sx / pts.length;
  const my = sy / pts.length;
  if (Math.hypot(mx - tipPt.x, my - tipPt.y) > sidePx * 0.42) return false;
  const padX = 0.1 * bw;
  const padY = 0.1 * bh;
  if (tipPt.x < x0 - padX || tipPt.x > x1 + padX) return false;
  if (tipPt.y < y0 - padY || tipPt.y > y1 + padY) return false;
  const pierce = pts[Math.min(55, pts.length - 1)];
  const tipD = Math.hypot(pierce[0] - tipPt.x, pierce[1] - tipPt.y);
  if (tipD < 0.1 * sidePx || tipD > 0.55 * sidePx) return false;
  if (pierce[1] < tipPt.y + 0.05 * sidePx) return false;

  // Ear-only: every landmark must stay near tip / inside crop (no cheek/face/hair)
  const maxR = sidePx * 0.52;
  let far = 0;
  for (const [x, y] of pts) {
    if (Math.hypot(x - tipPt.x, y - tipPt.y) > maxR) far += 1;
  }
  if (far > pts.length * 0.12) return false;

  if (cropBox) {
    const [bx0, by0, bx1, by1] = cropBox;
    const inset = Math.max(4, sidePx * 0.04);
    let out = 0;
    for (const [x, y] of pts) {
      if (
        x < bx0 + inset ||
        x > bx1 - inset ||
        y < by0 + inset ||
        y > by1 - inset
      )
        out += 1;
    }
    if (out > pts.length * 0.15) return false;
  }
  return true;
}

function drawSquareCrop(bitmap, cx, cy, sidePx, needFlip) {
  const s = Math.max(32, Math.round(sidePx));
  const ox = Math.round(cx - s * 0.5);
  const oy = Math.round(cy - s * 0.5);
  if (padCanvas.width !== s) {
    padCanvas.width = s;
    padCanvas.height = s;
  }
  padCtx.fillStyle = "rgb(114,114,114)";
  padCtx.fillRect(0, 0, s, s);
  const sw = bitmap.width;
  const sh = bitmap.height;
  const sx1 = Math.max(0, ox);
  const sy1 = Math.max(0, oy);
  const sx2 = Math.min(sw, ox + s);
  const sy2 = Math.min(sh, oy + s);
  const dx = sx1 - ox;
  const dy = sy1 - oy;
  if (sx2 > sx1 && sy2 > sy1) {
    padCtx.drawImage(bitmap, sx1, sy1, sx2 - sx1, sy2 - sy1, dx, dy, sx2 - sx1, sy2 - sy1);
  }
  cropCtx.save();
  cropCtx.setTransform(1, 0, 0, 1, 0, 0);
  if (needFlip) {
    cropCtx.translate(256, 0);
    cropCtx.scale(-1, 1);
  }
  cropCtx.drawImage(padCanvas, 0, 0, s, s, 0, 0, 256, 256);
  cropCtx.restore();
  return { ox, oy, sidePx: s };
}

async function runShg(needFlip, ox, oy, sidePx) {
  const img = cropCtx.getImageData(0, 0, 256, 256);
  const t0 = performance.now();
  const out = await shgSession.run({
    [shgInput]: new ort.Tensor("float32", canvasRgbaToBgrChw(img), [1, 3, 256, 256]),
  });
  const ms = performance.now() - t0;
  let pts256 = heatmapsToPointsSoft(out[shgSession.outputNames[0]], 256);
  const score = pts256.score ?? 0;
  if (needFlip) pts256 = pts256.map(([x, y]) => [255 - x, y]);
  const scale = sidePx / 256;
  const pts = pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
  return { pts, score, ms };
}

async function runPipeline(bitmap, prefer, firstLock, wantShg, emitTip) {
  const tPipe = performance.now();
  const vw = bitmap.width;
  const vh = bitmap.height;
  if (prefer !== undefined) preferSide = prefer;

  const { det, yoloMs } = await runYolo(bitmap, vw, vh);
  if (!det) {
    return { ok: false, reason: "no_ear", pipeMs: performance.now() - tPipe, yoloMs };
  }
  if (!profileOk(det)) {
    return {
      ok: false,
      reason: "not_profile",
      tip: det.tip,
      side: det.side,
      pipeMs: performance.now() - tPipe,
      yoloMs,
    };
  }

  const tip = det.tip;
  const side = det.side;
  const pinna = pinnaHeight(det, vw, vh);
  const sideLen = pinna * CROP_PAD;
  const [mx] = medial(det, tip, side, vw);
  const cx = tip.x + mx * (0.1 * pinna);
  const cy = tip.y + 0.06 * pinna;
  const box = [
    Math.max(0, Math.round(cx - sideLen * 0.5)),
    Math.max(0, Math.round(cy - sideLen * 0.5)),
    Math.min(vw, Math.round(cx + sideLen * 0.5)),
    Math.min(vh, Math.round(cy + sideLen * 0.5)),
  ];
  const geo = { cx, cy, side: sideLen };

  // Push tip ASAP so UI can tip-rigid while SHG (~400ms) runs
  if (typeof emitTip === "function") {
    emitTip({
      tip,
      side,
      geo,
      box,
      yoloMs,
      pipeMs: performance.now() - tPipe,
    });
  }

  if (!wantShg) {
    return {
      ok: true,
      tipOnly: true,
      tip,
      side,
      geo,
      box,
      pipeMs: performance.now() - tPipe,
      yoloMs,
      shgMs: 0,
    };
  }

  const preferFlip = side === "LEFT";
  let { ox, oy, sidePx } = drawSquareCrop(bitmap, cx, cy, sideLen, preferFlip);
  const cropBox = [ox, oy, ox + sidePx, oy + sidePx];
  let { pts, score, ms: shgMs } = await runShg(preferFlip, ox, oy, sidePx);
  let ok1 = landmarksOk(pts, tip, sidePx, cropBox);
  if (firstLock && !ok1) {
    drawSquareCrop(bitmap, cx, cy, sideLen, !preferFlip);
    const alt = await runShg(!preferFlip, ox, oy, sidePx);
    shgMs += alt.ms;
    if (
      alt.score > score ||
      landmarksOk(alt.pts, tip, sidePx, cropBox)
    ) {
      pts = alt.pts;
      score = alt.score;
      ok1 = landmarksOk(pts, tip, sidePx, cropBox);
    }
  }
  // Never accept a cloud that spills onto face/cheek/hair
  if (!landmarksOk(pts, tip, sidePx, cropBox)) {
    return {
      ok: false,
      reason: "bad_shg",
      tip,
      side,
      score,
      pipeMs: performance.now() - tPipe,
      yoloMs,
      shgMs,
    };
  }

  // Reject if landmark mass sits toward the face (nose) instead of the pinna
  if (det.nose && det.nose.c >= 0.2) {
    let sx = 0,
      sy = 0;
    for (const [x, y] of pts) {
      sx += x;
      sy += y;
    }
    const mx = sx / pts.length;
    const my = sy / pts.length;
    const dTip = Math.hypot(mx - tip.x, my - tip.y);
    const dNose = Math.hypot(mx - det.nose.x, my - det.nose.y);
    if (dNose + 8 < dTip || dNose < sidePx * 0.28) {
      return {
        ok: false,
        reason: "bad_shg",
        tip,
        side,
        score,
        pipeMs: performance.now() - tPipe,
        yoloMs,
        shgMs,
      };
    }
  }
  // Reject clouds that land on an eye
  for (const eye of [det.leftEye, det.rightEye]) {
    if (!eye || eye.c < 0.25) continue;
    let nearEye = 0;
    for (const [x, y] of pts) {
      if (Math.hypot(x - eye.x, y - eye.y) < 22) nearEye += 1;
    }
    if (nearEye >= 4) {
      return {
        ok: false,
        reason: "bad_shg",
        tip,
        side,
        score,
        pipeMs: performance.now() - tPipe,
        yoloMs,
        shgMs,
      };
    }
  }

  return {
    ok: true,
    tipOnly: false,
    tip,
    side,
    pts,
    box,
    score,
    geo,
    pipeMs: performance.now() - tPipe,
    yoloMs,
    shgMs,
  };
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === "init") {
      imgsz = msg.imgsz || 640;
      conf = msg.conf || 0.22;
      const threads = configureOrt(msg.wasmPath);
      shgSession = await ort.InferenceSession.create(msg.shgUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      yoloSession = await ort.InferenceSession.create(msg.yoloUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      shgInput = shgSession.inputNames[0];
      yoloInput = yoloSession.inputNames[0];
      self.postMessage({ type: "ready", threads });
      return;
    }

    if (msg.type === "infer") {
      if (busy) {
        if (msg.bitmap?.close) msg.bitmap.close();
        self.postMessage({ type: "miss", id: msg.id, reason: "busy", pipeMs: 0 });
        return;
      }
      busy = true;
      try {
        const wantShg = msg.wantShg !== false;
        const result = await runPipeline(
          msg.bitmap,
          msg.preferSide,
          !!msg.firstLock,
          wantShg,
          (partial) =>
            self.postMessage({ type: "tip", id: msg.id, ...partial })
        );
        if (msg.bitmap?.close) msg.bitmap.close();
        if (result.ok) {
          self.postMessage({ type: "result", id: msg.id, ...result });
        } else {
          self.postMessage({ type: "miss", id: msg.id, ...result });
        }
      } finally {
        busy = false;
      }
      return;
    }
  } catch (e) {
    if (msg.bitmap?.close) try { msg.bitmap.close(); } catch (_) {}
    busy = false;
    self.postMessage({
      type: "error",
      id: msg.id,
      message: e?.message || String(e),
    });
  }
};
