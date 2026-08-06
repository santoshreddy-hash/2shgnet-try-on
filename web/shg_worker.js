/**
 * SHGNet-56 Web Worker — ear landmarks from tip-centered crop.
 *
 *   main → { type:'init', shgUrl, wasmPath }
 *   ← { type:'ready', threads }
 *   main → { type:'infer', id, bitmap, tip, side, geo, firstLock }
 *   ← { type:'result', id, pts, box, score, ... } | { type:'miss', id, reason }
 */
import { configureOrt, ort } from "./worker_ort.js";
import { canvasRgbaToBgrChw, heatmapsToPointsSoft } from "./preprocess.js";

let shgSession = null;
let shgInput = null;
let busy = false;

const padCanvas = new OffscreenCanvas(256, 256);
const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });
const cropCanvas = new OffscreenCanvas(256, 256);
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

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
  const span = Math.max(x1 - x0, y1 - y0, 1);
  const ratio = span / Math.max(1, sidePx);
  if (ratio < 0.28 || ratio > 0.95) return false;
  let sx = 0,
    sy = 0;
  for (const [x, y] of pts) {
    sx += x;
    sy += y;
  }
  const mx = sx / pts.length;
  const my = sy / pts.length;
  if (Math.hypot(mx - tipPt.x, my - tipPt.y) > sidePx * 0.45) return false;
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
    padCtx.drawImage(
      bitmap,
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
    [shgInput]: new ort.Tensor("float32", canvasRgbaToBgrChw(img), [
      1, 3, 256, 256,
    ]),
  });
  const ms = performance.now() - t0;
  let pts256 = heatmapsToPointsSoft(out[shgSession.outputNames[0]], 256);
  const score = pts256.score ?? 0;
  if (needFlip) pts256 = pts256.map(([x, y]) => [255 - x, y]);
  const scale = sidePx / 256;
  const pts = pts256.map(([x, y]) => [ox + x * scale, oy + y * scale]);
  return { pts, score, ms };
}

async function runShgPipeline(bitmap, tip, side, geo, firstLock) {
  const vw = bitmap.width;
  const vh = bitmap.height;
  const cx = geo.cx;
  const cy = geo.cy;
  const sideLen = geo.side;
  const preferFlip = side === "LEFT";
  let { ox, oy, sidePx } = drawSquareCrop(bitmap, cx, cy, sideLen, preferFlip);
  let { pts, score, ms: shgMs } = await runShg(preferFlip, ox, oy, sidePx);
  const ok1 = landmarksOk(pts, tip, sidePx);
  if (firstLock && !ok1) {
    drawSquareCrop(bitmap, cx, cy, sideLen, !preferFlip);
    const alt = await runShg(!preferFlip, ox, oy, sidePx);
    shgMs += alt.ms;
    if (alt.score > score || landmarksOk(alt.pts, tip, sidePx)) {
      pts = alt.pts;
      score = alt.score;
    }
  }
  if (!(landmarksOk(pts, tip, sidePx) && score > 0.07)) {
    return { ok: false, reason: "bad_shg", tip, side, shgMs };
  }
  const box = [
    Math.max(0, Math.round(cx - sidePx * 0.5)),
    Math.max(0, Math.round(cy - sidePx * 0.5)),
    Math.min(vw, Math.round(cx + sidePx * 0.5)),
    Math.min(vh, Math.round(cy + sidePx * 0.5)),
  ];
  return { ok: true, tip, side, pts, box, score, geo, shgMs };
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === "init") {
      const threads = configureOrt(msg.wasmPath);
      shgSession = await ort.InferenceSession.create(msg.shgUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      shgInput = shgSession.inputNames[0];
      self.postMessage({ type: "ready", threads, model: "shg" });
      return;
    }

    if (msg.type === "infer") {
      if (busy) {
        if (msg.bitmap?.close) msg.bitmap.close();
        self.postMessage({ type: "miss", id: msg.id, reason: "busy", shgMs: 0 });
        return;
      }
      busy = true;
      const t0 = performance.now();
      try {
        const result = await runShgPipeline(
          msg.bitmap,
          msg.tip,
          msg.side,
          msg.geo,
          !!msg.firstLock
        );
        if (msg.bitmap?.close) msg.bitmap.close();
        if (result.ok) {
          self.postMessage({
            type: "result",
            id: msg.id,
            ...result,
            pipeMs: performance.now() - t0,
          });
        } else {
          self.postMessage({
            type: "miss",
            id: msg.id,
            ...result,
            pipeMs: performance.now() - t0,
          });
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
