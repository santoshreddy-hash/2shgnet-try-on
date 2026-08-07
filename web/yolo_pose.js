import { yieldToPaint } from "./yield.js";

/**
 * Ultralytics YOLO pose ONNX in the browser.
 *
 * Supports:
 *   A) Raw export  (1, 56, N) — cx,cy,w,h,score + 17×(x,y,v)  [current models]
 *   B) NMS export  (1, 300, 57) or (1, 57, N) — xyxy,conf,cls + 17×(x,y,c)
 */
const NOSE = 0;
const LEFT_EYE = 1;
const RIGHT_EYE = 2;
const LEFT_EAR = 3;
const RIGHT_EAR = 4;

export class YoloPoseBrowser {
  /**
   * @param {import('onnxruntime-web').InferenceSession} session
   * @param {(data:Float32Array,dims:number[]) => any} makeTensor
   */
  constructor(session, makeTensor, imgsz = 640, conf = 0.12) {
    this.session = session;
    this.makeTensor = makeTensor;
    this.inputName = session.inputNames[0];
    this.imgsz = 640;
    this.conf = conf;
    this._canvas = document.createElement("canvas");
    this._canvas.width = 640;
    this._canvas.height = 640;
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    this._chw = new Float32Array(3 * 640 * 640);
    this.busy = false;
    this.last = null;
    this.lastMs = 0;
    /** When true, yield to paint between preprocess and WASM run. */
    this.yieldBeforeRun = false;
  }

  setImgsz(_imgsz) {
    // ONNX is fixed 640×640
    this.imgsz = 640;
    this._canvas.width = 640;
    this._canvas.height = 640;
    this._chw = new Float32Array(3 * 640 * 640);
  }

  async detect(source) {
    if (this.busy) return this.last;
    this.busy = true;
    const t0 = performance.now();
    try {
      const vw = source.videoWidth || source.width;
      const vh = source.videoHeight || source.height;
      if (!vw || !vh) return this.last;

      const r = Math.min(this.imgsz / vh, this.imgsz / vw);
      const nw = Math.round(vw * r);
      const nh = Math.round(vh * r);
      const padX = (this.imgsz - nw) / 2;
      const padY = (this.imgsz - nh) / 2;
      this._ctx.fillStyle = "#727272";
      this._ctx.fillRect(0, 0, this.imgsz, this.imgsz);
      this._ctx.drawImage(source, padX, padY, nw, nh);

      const { data } = this._ctx.getImageData(0, 0, this.imgsz, this.imgsz);
      const plane = this.imgsz * this.imgsz;
      const chw = this._chw;
      const inv = 1 / 255;
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        chw[p] = data[i] * inv;
        chw[plane + p] = data[i + 1] * inv;
        chw[2 * plane + p] = data[i + 2] * inv;
      }

      if (this.yieldBeforeRun) await yieldToPaint();

      const tensor = this.makeTensor(chw, [1, 3, this.imgsz, this.imgsz]);
      const out = await this.session.run({ [this.inputName]: tensor });
      const tensorOut = out[this.session.outputNames[0]];
      const arr = tensorOut.data;
      const dims = tensorOut.dims || [];

      const un = (x, y) => [(x - padX) / r, (y - padY) / r];

      let best = null;

      if (dims.length === 3 && dims[1] === 56) {
        const nDet = dims[2];
        const get = (c, i) => arr[c * nDet + i];
        let bestOff = -1;
        let bestScore = this.conf;
        for (let i = 0; i < nDet; i++) {
          const score = get(4, i);
          if (score < this.conf) continue;
          // Prefer detections with a clear ear tip (accuracy over raw box score)
          const le = get(5 + LEFT_EAR * 3 + 2, i);
          const re = get(5 + RIGHT_EAR * 3 + 2, i);
          const earC = Math.max(le, re);
          if (earC < 0.15) continue;
          const rank = score * (0.35 + 0.65 * earC);
          if (rank > bestScore) {
            bestScore = rank;
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
          best = { kpt, bbox: [x1, y1, x2, y2], conf: get(4, bestOff) };
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
        let bestScore = this.conf;
        for (let i = 0; i < nDet; i++) {
          const score = get(i, 4);
          if (score < this.conf) continue;
          const le = get(i, 6 + LEFT_EAR * 3 + 2);
          const re = get(i, 6 + RIGHT_EAR * 3 + 2);
          const earC = Math.max(le, re);
          if (earC < 0.15) continue;
          const rank = score * (0.35 + 0.65 * earC);
          if (rank > bestScore) {
            bestScore = rank;
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
          best = { kpt, bbox: [x1, y1, x2, y2], conf: get(bestOff, 4) };
        }
      }

      if (!best) {
        this.last = null;
        return null;
      }

      const left = best.kpt(LEFT_EAR);
      const right = best.kpt(RIGHT_EAR);
      const nose = best.kpt(NOSE);
      const leftEye = best.kpt(LEFT_EYE);
      const rightEye = best.kpt(RIGHT_EYE);
      const [x1, y1, x2, y2] = best.bbox;

      let side;
      if (left.c >= 0.15 || right.c >= 0.15) {
        side = left.c >= right.c ? "LEFT" : "RIGHT";
      } else {
        side =
          Math.abs(left.x - nose.x) >= Math.abs(right.x - nose.x)
            ? "LEFT"
            : "RIGHT";
      }
      const ear = side === "LEFT" ? left : right;
      const other = side === "LEFT" ? right : left;
      if (ear.c < 0.15) {
        this.last = null;
        return null;
      }
      if (other.c >= 0.35 && other.c >= ear.c * 0.7) {
        this.last = null;
        return null;
      }
      if (nose.c >= 0.2) {
        const dx = Math.abs(ear.x - nose.x);
        const d = Math.hypot(ear.x - nose.x, ear.y - nose.y);
        if (dx < 28 || d < 36) {
          this.last = null;
          return null;
        }
        if (dx < d * 0.55) {
          this.last = null;
          return null;
        }
      }

      let eyeDist = null;
      if (leftEye.c >= 0.2 && rightEye.c >= 0.2) {
        eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
      }

      this.last = {
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
      return this.last;
    } catch (e) {
      console.error("[YOLO]", e);
      this.last = null;
      return null;
    } finally {
      this.lastMs = performance.now() - t0;
      this.busy = false;
    }
  }
}
