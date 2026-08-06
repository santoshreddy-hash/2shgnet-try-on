/**
 * Ultralytics YOLO pose ONNX in the browser.
 *
 * Supports:
 *   A) Raw export  (1, 56, N) — cx,cy,w,h,score + 17×(x,y,v)  [current models]
 *   B) NMS export  (1, 300, 57) or (1, 57, N) — xyxy,conf,cls + 17×(x,y,c)
 *
 * Ear pick matches desktop EarCropper._pick_ear: soft prefer_side, switch when
 * the other ear wins (no hard lock that freezes landmarks across head turns).
 */
const NOSE = 0;
const LEFT_EYE = 1;
const RIGHT_EYE = 2;
const LEFT_EAR = 3;
const RIGHT_EAR = 4;
const EAR_MIN_CONF = 0.3;

export class YoloPoseBrowser {
  /**
   * @param {import('onnxruntime-web').InferenceSession} session
   * @param {(data:Float32Array,dims:number[]) => any} makeTensor
   */
  constructor(session, makeTensor, imgsz = 640, conf = 0.22) {
    this.session = session;
    this.makeTensor = makeTensor;
    this.inputName = session.inputNames[0];
    this.imgsz = imgsz;
    this.conf = conf;
    this._canvas = document.createElement("canvas");
    this._canvas.width = imgsz;
    this._canvas.height = imgsz;
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    this.busy = false;
    this.last = null;
    this.preferSide = null; // "LEFT" | "RIGHT" | null
  }

  /**
   * @param {CanvasImageSource} source
   * @param {{ preferSide?: string|null }} [opts]
   */
  async detect(source, opts = {}) {
    // Never return a stale detection — that freezes landmarks on the old ear
    if (this.busy) return null;
    this.busy = true;
    try {
      if (opts.preferSide !== undefined) {
        this.preferSide = opts.preferSide;
      }
      const vw = source.videoWidth || source.width;
      const vh = source.videoHeight || source.height;
      if (!vw || !vh) {
        this.last = null;
        return null;
      }

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
      const chw = new Float32Array(3 * plane);
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        chw[p] = data[i] / 255;
        chw[plane + p] = data[i + 1] / 255;
        chw[2 * plane + p] = data[i + 2] / 255;
      }
      const tensor = this.makeTensor(chw, [1, 3, this.imgsz, this.imgsz]);
      const out = await this.session.run({ [this.inputName]: tensor });
      const tensorOut = out[this.session.outputNames[0]];
      const arr = tensorOut.data;
      const dims = tensorOut.dims || [];

      const un = (x, y) => [(x - padX) / r, (y - padY) / r];

      let best = null;

      // A) Raw Ultralytics pose: (1, 56, N)
      if (dims.length === 3 && dims[1] === 56) {
        const nDet = dims[2];
        const get = (c, i) => arr[c * nDet + i];
        let bestOff = -1;
        let bestScore = this.conf;
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
        // B) NMS-style (1, N, 57) or (1, 57, N)
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

      const picked = this._pickEar(left, right, nose);
      if (!picked) {
        this.last = null;
        return null;
      }
      const { side, ear, other } = picked;

      let eyeDist = null;
      if (leftEye.c >= 0.2 && rightEye.c >= 0.2) {
        eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
      }

      this.preferSide = side;
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
      this.busy = false;
    }
  }

  /** Soft prefer_side; switch when the opposite ear clearly wins. */
  _pickEar(left, right, nose) {
    const prefer = (this.preferSide || "").toUpperCase();
    const isProfile = (ear, other, name) => {
      if (ear.c < EAR_MIN_CONF) return false;
      // Mid-turn / frontal: both ears visible → reject (no face landmarks)
      if (other.c >= 0.38 && other.c >= ear.c * 0.72) return false;
      if (nose?.c >= 0.2) {
        const dx = Math.abs(ear.x - nose.x);
        const d = Math.hypot(ear.x - nose.x, ear.y - nose.y);
        if (dx < 22 || d < 28) return false;
      }
      return true;
    };

    const cands = [];
    for (const [ear, other, name] of [
      [left, right, "LEFT"],
      [right, left, "RIGHT"],
    ]) {
      if (!isProfile(ear, other, name)) continue;
      let score = ear.c;
      if (prefer === name) score += 0.08;
      cands.push({ score, side: name, ear, other });
    }
    if (!cands.length) {
      // Last resort: higher-conf ear even if profile gate failed
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
}
