/**
 * YOLO pose Web Worker — owns its own ORT WASM + yolo26n-pose session.
 * Isolated from SHGNet so both models never share one worker thread.
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";

let IMGSZ = 640;
const confThr = 0.25;
const NOSE = 0;
const LEFT_EYE = 1;
const RIGHT_EYE = 2;
const LEFT_EAR = 3;
const RIGHT_EAR = 4;

let yoloSession = null;
let letterCanvas = null;
let letterCtx = null;
let chw = null;

function ensureLetterbox() {
  if (!letterCanvas || letterCanvas.width !== IMGSZ) {
    letterCanvas = new OffscreenCanvas(IMGSZ, IMGSZ);
    letterCtx = letterCanvas.getContext("2d", { willReadFrequently: true });
    chw = new Float32Array(3 * IMGSZ * IMGSZ);
  }
}

function parseYolo(arr, dims, padX, padY, r) {
  const un = (x, y) => [(x - padX) / r, (y - padY) / r];
  let best = null;

  if (dims.length === 3 && dims[1] === 56) {
    const nDet = dims[2];
    const get = (c, i) => arr[c * nDet + i];
    let bestOff = -1;
    let bestScore = confThr;
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
    let bestScore = confThr;
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
  const [x1, y1, x2, y2] = best.bbox;

  let side;
  if (left.c >= 0.25 || right.c >= 0.25) {
    side = left.c >= right.c ? "LEFT" : "RIGHT";
  } else {
    side =
      Math.abs(left.x - nose.x) >= Math.abs(right.x - nose.x) ? "LEFT" : "RIGHT";
  }
  const ear = side === "LEFT" ? left : right;
  const other = side === "LEFT" ? right : left;
  if (ear.c < 0.25) return null;
  if (other.c >= 0.5 && other.c >= ear.c * 0.85) return null;
  if (nose.c >= 0.2) {
    const dx = Math.abs(ear.x - nose.x);
    const d = Math.hypot(ear.x - nose.x, ear.y - nose.y);
    if (dx < 22 || d < 28) return null;
  }

  let eyeDist = null;
  if (leftEye.c >= 0.2 && rightEye.c >= 0.2) {
    eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
  }

  return {
    side,
    ear,
    tip: { x: ear.x, y: ear.y },
    bbox: [x1, y1, x2, y2],
    nose,
    eyeDist,
    left,
    right,
    earConf: ear.c,
    earOtherConf: other.c,
    conf: best.conf,
  };
}

async function init(msg) {
  const { modelUrl, wasmPaths } = msg;
  // ONNX export is fixed [1,3,640,640] — ignore any other imgsz
  IMGSZ = 640;
  ort.env.wasm.wasmPaths = wasmPaths || "/vendor/onnxruntime-web/dist/";
  const cores =
    typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 2 : 2;
  ort.env.wasm.numThreads =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated
      ? Math.min(4, Math.max(1, cores >> 1))
      : 1;
  ort.env.wasm.proxy = false;

  yoloSession = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  letterCanvas = null;
  ensureLetterbox();
  return {
    input: yoloSession.inputNames[0],
    output: yoloSession.outputNames[0],
    imgsz: IMGSZ,
  };
}

async function runYoloFromBitmap(bitmap) {
  ensureLetterbox();
  const vw = bitmap.width;
  const vh = bitmap.height;
  const r = Math.min(IMGSZ / vh, IMGSZ / vw);
  const nw = Math.round(vw * r);
  const nh = Math.round(vh * r);
  const padX = (IMGSZ - nw) / 2;
  const padY = (IMGSZ - nh) / 2;
  letterCtx.fillStyle = "#727272";
  letterCtx.fillRect(0, 0, IMGSZ, IMGSZ);
  letterCtx.drawImage(bitmap, padX, padY, nw, nh);
  bitmap.close?.();

  const { data } = letterCtx.getImageData(0, 0, IMGSZ, IMGSZ);
  const plane = IMGSZ * IMGSZ;
  const inv = 1 / 255;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    chw[p] = data[i] * inv;
    chw[plane + p] = data[i + 1] * inv;
    chw[2 * plane + p] = data[i + 2] * inv;
  }

  const tensor = new ort.Tensor("float32", chw, [1, 3, IMGSZ, IMGSZ]);
  const t0 = performance.now();
  const out = await yoloSession.run({ [yoloSession.inputNames[0]]: tensor });
  const ms = performance.now() - t0;
  const tensorOut = out[yoloSession.outputNames[0]];
  const det = parseYolo(tensorOut.data, tensorOut.dims || [], padX, padY, r);
  return { det, ms };
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  const { id, type } = msg;
  try {
    if (type === "init") {
      const meta = await init(msg);
      self.postMessage({ id, ok: true, type: "init", meta });
      return;
    }
    if (type === "setImgsz") {
      IMGSZ = 640;
      letterCanvas = null;
      ensureLetterbox();
      self.postMessage({ id, ok: true, type: "setImgsz", imgsz: IMGSZ });
      return;
    }
    if (type === "yolo") {
      if (!yoloSession) throw new Error("YOLO worker not initialized");
      const { det, ms } = await runYoloFromBitmap(msg.bitmap);
      self.postMessage({ id, ok: true, type: "yolo", det, ms });
      return;
    }
    throw new Error(`Unknown YOLO worker message: ${type}`);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      type,
      error: String(err?.message || err),
    });
  }
};
