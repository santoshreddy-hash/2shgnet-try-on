/**
 * Main-thread clients for per-model Web Workers.
 * YOLO and SHGNet each get their own Worker + ORT WASM instance.
 */

class OrtWorkerBase {
  /**
   * @param {string} workerUrl
   * @param {string} label
   */
  constructor(workerUrl, label) {
    this.workerUrl = workerUrl;
    this.label = label;
    /** @type {Worker | null} */
    this.worker = null;
    this._seq = 1;
    /** @type {Map<number, {resolve:Function, reject:Function}>} */
    this._pending = new Map();
    this.ready = false;
    this.lastMs = 0;
  }

  _ensureWorker() {
    if (this.worker) return;
    this.worker = new Worker(this.workerUrl, { type: "module" });
    this.worker.onmessage = (ev) => {
      const msg = ev.data || {};
      const pending = this._pending.get(msg.id);
      if (!pending) return;
      this._pending.delete(msg.id);
      if (msg.ok) pending.resolve(msg);
      else pending.reject(new Error(msg.error || `${this.label} worker error`));
    };
    this.worker.onerror = (ev) => {
      const err = new Error(ev.message || `${this.label} worker error`);
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    };
  }

  _call(payload, transfer = []) {
    this._ensureWorker();
    const id = this._seq++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id }, transfer);
    });
  }

  /**
   * @param {{ modelUrl: string, wasmPaths?: string, imgsz?: number }} opts
   */
  async init(opts) {
    const msg = await this._call({
      type: "init",
      modelUrl: opts.modelUrl,
      wasmPaths: opts.wasmPaths || "/vendor/onnxruntime-web/dist/",
      imgsz: opts.imgsz,
    });
    this.ready = true;
    return msg.meta;
  }

  async setImgsz(imgsz) {
    if (!this.ready) return;
    const msg = await this._call({ type: "setImgsz", imgsz });
    return msg.imgsz;
  }

  terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.ready = false;
    this._pending.clear();
  }
}

export class YoloOrtWorker extends OrtWorkerBase {
  constructor(workerUrl = "/yolo-worker.js") {
    super(workerUrl, "YOLO");
  }

  /**
   * @param {HTMLCanvasElement|OffscreenCanvas|ImageBitmap} source
   */
  async detect(source) {
    let bitmap;
    if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) {
      bitmap = source;
    } else {
      bitmap = await createImageBitmap(source);
    }
    const msg = await this._call({ type: "yolo", bitmap }, [bitmap]);
    this.lastMs = msg.ms || 0;
    return msg.det || null;
  }
}

export class ShgOrtWorker extends OrtWorkerBase {
  constructor(workerUrl = "/shg-worker.js") {
    super(workerUrl, "SHG");
  }

  /**
   * @param {Float32Array} chw
   * @param {number[]} dims
   */
  async run(chw, dims = [1, 3, 256, 256]) {
    const copy = chw.slice();
    const msg = await this._call(
      { type: "shg", chw: copy.buffer, dims },
      [copy.buffer]
    );
    this.lastMs = msg.ms || 0;
    return {
      data: new Float32Array(msg.data),
      dims: msg.dims,
    };
  }
}

/**
 * Dual-worker facade (YOLO worker + SHG worker).
 * Keeps the previous OrtInferenceWorker call shape for infer.js.
 */
export class OrtInferenceWorker {
  constructor({
    yoloWorkerUrl = "/yolo-worker.js",
    shgWorkerUrl = "/shg-worker.js",
  } = {}) {
    this.yolo = new YoloOrtWorker(yoloWorkerUrl);
    this.shg = new ShgOrtWorker(shgWorkerUrl);
    this.ready = false;
    this.lastYoloMs = 0;
    this.lastShgMs = 0;
  }

  async init({
    yoloUrl = "/models/yolo/yolo26n-pose.onnx",
    shgUrl = "/models/shgnet/SHGNet-56.onnx",
    wasmPaths = "/vendor/onnxruntime-web/dist/",
    yoloImgsz = 640,
  } = {}) {
    const [yoloMeta, shgMeta] = await Promise.all([
      this.yolo.init({ modelUrl: yoloUrl, wasmPaths, imgsz: yoloImgsz }),
      this.shg.init({ modelUrl: shgUrl, wasmPaths }),
    ]);
    this.ready = this.yolo.ready && this.shg.ready;
    return { yolo: yoloMeta, shg: shgMeta };
  }

  async setYoloImgsz(imgsz) {
    return this.yolo.setImgsz?.(imgsz);
  }

  async detectYolo(source) {
    const det = await this.yolo.detect(source);
    this.lastYoloMs = this.yolo.lastMs;
    return det;
  }

  async runShg(chw, dims = [1, 3, 256, 256]) {
    const out = await this.shg.run(chw, dims);
    this.lastShgMs = this.shg.lastMs;
    return out;
  }

  terminate() {
    this.yolo.terminate();
    this.shg.terminate();
    this.ready = false;
  }
}
