/**
 * YOLO pose Web Worker — ear tip / side only.
 *
 *   main → { type:'init', yoloUrl, wasmPath, imgsz, conf }
 *   ← { type:'ready', threads }
 *   main → { type:'infer', id, bitmap, preferSide }
 *   ← { type:'result', id, tip, side, geo, ... } | { type:'miss', id, reason }
 */
import { configureOrt, ort } from "./worker_ort.js";

const CROP_PAD = 1.65;
const LEFT_EAR = 3;
const RIGHT_EAR = 4;
const NOSE = 0;
const LEFT_EYE = 1;
const RIGHT_EYE = 2;
const EAR_MIN_CONF = 0.3;

let yoloSession = null;
let yoloInput = null;
let imgsz = 640;
let conf = 0.22;
let preferSide = null;
let busy = false;

const yoloCanvas = new OffscreenCanvas(640, 640);
const yoloCtx = yoloCanvas.getContext("2d", { willReadFrequently: true });

function isSideProfile(ear, other, nose) {
  if (ear.c < EAR_MIN_CONF) return false;
  if (other.c >= 0.38 && other.c >= ear.c * 0.72) return false;
  if (nose?.c >= 0.2) {
    const dx = Math.abs(ear.x - nose.x);
    const d = Math.hypot(ear.x - nose.x, ear.y - nose.y);
    if (dx < 22 || d < 28) return false;
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
    eyeDist,
    earConf: picked.ear.c,
    earOtherConf: picked.other.c,
    conf: best.conf,
  };
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
  if (yolo.earOtherConf != null && yolo.earConf != null) {
    if (yolo.earOtherConf >= 0.38 && yolo.earOtherConf >= yolo.earConf * 0.72)
      return false;
    if (yolo.earConf < 0.35) return false;
  }
  if (!yolo.nose || yolo.nose.c < 0.2) return true;
  const dx = Math.abs(yolo.tip.x - yolo.nose.x);
  const d = Math.hypot(yolo.tip.x - yolo.nose.x, yolo.tip.y - yolo.nose.y);
  return !(dx < 22 || d < 28);
}

async function runYolo(bitmap) {
  const vw = bitmap.width;
  const vh = bitmap.height;
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
  return { det, yoloMs, vw, vh };
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === "init") {
      imgsz = msg.imgsz || 640;
      conf = msg.conf || 0.22;
      const threads = configureOrt(msg.wasmPath);
      yoloSession = await ort.InferenceSession.create(msg.yoloUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      yoloInput = yoloSession.inputNames[0];
      self.postMessage({ type: "ready", threads, model: "yolo" });
      return;
    }

    if (msg.type === "infer") {
      if (busy) {
        if (msg.bitmap?.close) msg.bitmap.close();
        self.postMessage({ type: "miss", id: msg.id, reason: "busy", yoloMs: 0 });
        return;
      }
      busy = true;
      const t0 = performance.now();
      try {
        if (msg.preferSide !== undefined) preferSide = msg.preferSide;
        const { det, yoloMs, vw, vh } = await runYolo(msg.bitmap);
        if (msg.bitmap?.close) msg.bitmap.close();
        if (!det) {
          self.postMessage({
            type: "miss",
            id: msg.id,
            reason: "no_ear",
            yoloMs,
            pipeMs: performance.now() - t0,
          });
          return;
        }
        if (!profileOk(det)) {
          self.postMessage({
            type: "miss",
            id: msg.id,
            reason: "not_profile",
            tip: det.tip,
            side: det.side,
            yoloMs,
            pipeMs: performance.now() - t0,
          });
          return;
        }
        const tip = det.tip;
        const side = det.side;
        const pinna = pinnaHeight(det, vw, vh);
        const sideLen = pinna * CROP_PAD;
        const [mx] = medial(det, tip, side, vw);
        const cx = tip.x + mx * (0.1 * pinna);
        const cy = tip.y + 0.06 * pinna;
        self.postMessage({
          type: "result",
          id: msg.id,
          tip,
          side,
          geo: { cx, cy, side: sideLen },
          bbox: det.bbox,
          earConf: det.earConf,
          earOtherConf: det.earOtherConf,
          yoloMs,
          pipeMs: performance.now() - t0,
        });
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
