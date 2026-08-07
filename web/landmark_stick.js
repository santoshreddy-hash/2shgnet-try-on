/**
 * Browser port of tracking/landmark_stick.py — keeps the landmark cloud glued
 * to ear texture between SHG updates (same role as desktop "ONNX stick").
 */
const WIN = 5;
const MAX_ITER = 6;
const EPS = 0.02;
const MAX_STEP = 48; // match tracking/landmark_stick.py
const SHG_BLEND = 0.42;
const ROI_PAD = 10;

function grabGray(ctx, x0, y0, w, h) {
  const cw = ctx.canvas.width;
  const ch = ctx.canvas.height;
  if (x0 < 0 || y0 < 0 || x0 + w > cw || y0 + h > ch || w < 16 || h < 16) return null;
  const img = ctx.getImageData(x0, y0, w, h);
  const g = new Float32Array(w * h);
  for (let i = 0, p = 0; i < img.data.length; i += 4, p++) {
    g[p] = 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
  }
  return { g, w, h, x0, y0 };
}

function sample(g, w, h, x, y) {
  if (x < 1 || y < 1 || x >= w - 2 || y >= h - 2) return null;
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * w + x0;
  return (
    g[i] * (1 - fx) * (1 - fy) +
    g[i + 1] * fx * (1 - fy) +
    g[i + w] * (1 - fx) * fy +
    g[i + w + 1] * fx * fy
  );
}

function lkPoint(prev, next, px, py) {
  let gx = 0;
  let gy = 0;
  for (let it = 0; it < MAX_ITER; it++) {
    let ix = 0;
    let iy = 0;
    let ixx = 0;
    let ixy = 0;
    let iyy = 0;
    let n = 0;
    for (let wy = -WIN; wy <= WIN; wy++) {
      for (let wx = -WIN; wx <= WIN; wx++) {
        const i1 = sample(prev.g, prev.w, prev.h, px + wx, py + wy);
        const i2 = sample(next.g, next.w, next.h, px + gx + wx, py + gy + wy);
        if (i1 == null || i2 == null) continue;
        const ixp =
          (sample(prev.g, prev.w, prev.h, px + wx + 1, py + wy) ?? i1) -
          (sample(prev.g, prev.w, prev.h, px + wx - 1, py + wy) ?? i1);
        const iyp =
          (sample(prev.g, prev.w, prev.h, px + wx, py + wy + 1) ?? i1) -
          (sample(prev.g, prev.w, prev.h, px + wx, py + wy - 1) ?? i1);
        const itv = i1 - i2;
        ix += ixp * itv;
        iy += iyp * itv;
        ixx += ixp * ixp;
        ixy += ixp * iyp;
        iyy += iyp * iyp;
        n++;
      }
    }
    if (n < 8) return null;
    const det = ixx * iyy - ixy * ixy;
    if (Math.abs(det) < 1e-3) return null;
    const dx = (iyy * ix - ixy * iy) / det;
    const dy = (ixx * iy - ixy * ix) / det;
    gx += dx;
    gy += dy;
    if (dx * dx + dy * dy < EPS * EPS) break;
  }
  return { dx: gx, dy: gy };
}

export class LandmarkStickTracker {
  constructor(nTrack = 14) {
    this.nTrack = Math.max(4, nTrack | 0);
    this.prev = null;
    this.absPts = null;
    this.trackIdx = null;
    this.ptsGen = -1;
    this.lastDx = 0;
    this.lastDy = 0;
  }

  reset() {
    this.prev = null;
    this.absPts = null;
    this.trackIdx = null;
    this.ptsGen = -1;
    this.lastDx = 0;
    this.lastDy = 0;
  }

  _seed(gray, pts, gen) {
    this.absPts = pts.map((p) => [p[0], p[1]]);
    const n = this.absPts.length;
    const k = Math.min(this.nTrack, Math.max(4, n - 1));
    this.trackIdx = [];
    for (let i = 0; i < k; i++) {
      this.trackIdx.push(Math.round((i * Math.min(54, n - 2)) / Math.max(1, k - 1)));
    }
    this.prev = gray;
    this.ptsGen = gen;
  }

  _roi(gray, box) {
    if (!this.absPts || !this.absPts.length) return null;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const i of this.trackIdx) {
      const p = this.absPts[i];
      if (p[0] < x0) x0 = p[0];
      if (p[1] < y0) y0 = p[1];
      if (p[0] > x1) x1 = p[0];
      if (p[1] > y1) y1 = p[1];
    }
    if (box) {
      x0 = Math.min(x0, box[0]);
      y0 = Math.min(y0, box[1]);
      x1 = Math.max(x1, box[2]);
      y1 = Math.max(y1, box[3]);
    }
    const pad = 40;
    const rx0 = Math.max(0, Math.floor(x0 - pad));
    const ry0 = Math.max(0, Math.floor(y0 - pad));
    const rw = Math.min(gray.canvasWidth, Math.ceil(x1 + pad)) - rx0;
    const rh = Math.min(gray.canvasHeight, Math.ceil(y1 + pad)) - ry0;
    return { rx0, ry0, rw, rh };
  }

  _medianDelta(prev, next, box) {
    if (!prev || !next || !this.absPts || !this.trackIdx) return null;
    const dxs = [];
    const dys = [];
    for (const i of this.trackIdx) {
      const p = this.absPts[i];
      const px = p[0] - prev.x0;
      const py = p[1] - prev.y0;
      const res = lkPoint(prev, next, px, py);
      if (!res) continue;
      const nx = p[0] + res.dx;
      const ny = p[1] + res.dy;
      if (box) {
        if (
          nx < box[0] - ROI_PAD ||
          ny < box[1] - ROI_PAD ||
          nx > box[2] + ROI_PAD ||
          ny > box[3] + ROI_PAD
        ) {
          continue;
        }
      }
      dxs.push(res.dx);
      dys.push(res.dy);
    }
    if (dxs.length < 3) return null;
    dxs.sort((a, b) => a - b);
    dys.sort((a, b) => a - b);
    const mid = (dxs.length / 2) | 0;
    let dx = dxs[mid];
    let dy = dys[mid];
    const step = Math.hypot(dx, dy);
    if (step > MAX_STEP) {
      dx *= MAX_STEP / step;
      dy *= MAX_STEP / step;
    }
    return { dx, dy };
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number[][]|null} workerPts new SHG/filtered abs points, or null
   * @param {number} ptsGen
   * @param {number[]|null} box [x1,y1,x2,y2]
   */
  update(ctx, workerPts, ptsGen, box = null) {
    const cw = ctx.canvas.width;
    const ch = ctx.canvas.height;
    if (!cw || !ch) return this.absPts;

    // ROI around current cloud / crop
    let x0 = 0;
    let y0 = 0;
    let side = Math.min(cw, ch);
    if (this.absPts && this.absPts.length) {
      let mx = 0;
      let my = 0;
      for (const p of this.absPts) {
        mx += p[0];
        my += p[1];
      }
      mx /= this.absPts.length;
      my /= this.absPts.length;
      side = Math.min(160, Math.min(cw, ch));
      x0 = Math.max(0, Math.min(cw - side, Math.round(mx - side * 0.5)));
      y0 = Math.max(0, Math.min(ch - side, Math.round(my - side * 0.5)));
    } else if (box) {
      side = Math.min(160, Math.max(64, Math.round(Math.max(box[2] - box[0], box[3] - box[1]) * 1.2)));
      x0 = Math.max(0, Math.min(cw - side, Math.round((box[0] + box[2]) * 0.5 - side * 0.5)));
      y0 = Math.max(0, Math.min(ch - side, Math.round((box[1] + box[3]) * 0.5 - side * 0.5)));
    }

    const next = grabGray(ctx, x0, y0, side, side);
    if (!next) {
      this.prev = null;
      return this.absPts;
    }
    next.canvasWidth = cw;
    next.canvasHeight = ch;

    const lk =
      this.prev && this.prev.w === next.w && this.prev.h === next.h
        ? this._medianDelta(this.prev, next, box)
        : null;
    this.lastDx = lk ? lk.dx : 0;
    this.lastDy = lk ? lk.dy : 0;

    if (workerPts && ptsGen !== this.ptsGen) {
      let pts = workerPts.map((p) => [p[0], p[1]]);
      if (lk && this.absPts && this.absPts.length === pts.length) {
        const predicted = this.absPts.map((p) => [p[0] + lk.dx, p[1] + lk.dy]);
        let shgCx = 0;
        let shgCy = 0;
        let predCx = 0;
        let predCy = 0;
        for (let i = 0; i < pts.length; i++) {
          shgCx += pts[i][0];
          shgCy += pts[i][1];
          predCx += predicted[i][0];
          predCy += predicted[i][1];
        }
        shgCx /= pts.length;
        shgCy /= pts.length;
        predCx /= pts.length;
        predCy /= pts.length;
        const ox = shgCx - predCx;
        const oy = shgCy - predCy;
        const w = SHG_BLEND;
        pts = pts.map((p, i) => [
          (1 - w) * p[0] + w * (predicted[i][0] + ox),
          (1 - w) * p[1] + w * (predicted[i][1] + oy),
        ]);
      }
      this._seed(next, pts, ptsGen);
      return this.absPts;
    }

    if (!this.absPts) {
      if (workerPts) this._seed(next, workerPts, ptsGen);
      return this.absPts;
    }

    if (!lk) {
      this.prev = next;
      return this.absPts;
    }

    this.absPts = this.absPts.map((p) => [p[0] + lk.dx, p[1] + lk.dy]);
    this.prev = next;
    return this.absPts;
  }
}
