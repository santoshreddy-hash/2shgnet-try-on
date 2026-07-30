/**
 * YOLO26n-pose ONNX — tip + side + IOD cues for full-ear crop.
 * Output: (1, 300, 57) = xyxy + conf + cls + 17×(x,y,conf)
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
  constructor(session, makeTensor, imgsz = 640, conf = 0.35) {
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
  }

  async detect(source) {
    if (this.busy) return this.last;
    this.busy = true;
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
      // (1, 300, 57) or (1, 57, N) depending on export
      const dims = tensorOut.dims || [];
      let stride = 57;
      let nDet = 300;
      let get = (i, c) => arr[i * stride + c];
      if (dims.length === 3 && dims[1] === 57) {
        // (1, 57, N)
        nDet = dims[2];
        stride = nDet;
        get = (i, c) => arr[c * stride + i];
      } else if (dims.length === 3 && dims[2] === 57) {
        nDet = dims[1];
        stride = 57;
        get = (i, c) => arr[i * stride + c];
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
      if (bestOff < 0) {
        this.last = null;
        return null;
      }

      const un = (x, y) => [(x - padX) / r, (y - padY) / r];
      const kpt = (idx) => {
        const base = 6 + idx * 3;
        const [x, y] = un(get(bestOff, base), get(bestOff, base + 1));
        return { x, y, c: get(bestOff, base + 2) };
      };

      const left = kpt(LEFT_EAR);
      const right = kpt(RIGHT_EAR);
      const nose = kpt(NOSE);
      const leftEye = kpt(LEFT_EYE);
      const rightEye = kpt(RIGHT_EYE);
      const [x1, y1] = un(get(bestOff, 0), get(bestOff, 1));
      const [x2, y2] = un(get(bestOff, 2), get(bestOff, 3));

      let side;
      if (left.c >= 0.25 || right.c >= 0.25) {
        side = left.c >= right.c ? "LEFT" : "RIGHT";
      } else {
        side =
          Math.abs(left.x - nose.x) >= Math.abs(right.x - nose.x)
            ? "LEFT"
            : "RIGHT";
      }
      const ear = side === "LEFT" ? left : right;
      const other = side === "LEFT" ? right : left;
      if (ear.c < 0.30) {
        this.last = null;
        return null;
      }
      // Strong frontal only
      if (other.c >= 0.5 && other.c >= ear.c * 0.85) {
        this.last = null;
        return null;
      }
      if (nose.c >= 0.2) {
        const dx = Math.abs(ear.x - nose.x);
        const d = Math.hypot(ear.x - nose.x, ear.y - nose.y);
        if (dx < 22 || d < 28) {
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
        conf: bestScore,
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
}
