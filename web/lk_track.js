/**
 * Desktop-parity tracking for the browser:
 *   - Lucas–Kanade tip track (matches live/desktop_onnx.track_tip_lk)
 *   - LandmarkStickTracker (matches tracking/landmark_stick.py)
 *
 * Gray buffers are Uint8Array length = w*h (row-major).
 */

const LK_WIN = 31;
const TIP_MAX_STEP = 48;
const STICK_MAX_STEP = 48;
const SHG_BLEND = 0.65; // match tracking/landmark_stick.py _SHG_BLEND

function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}

/** RGBA ImageData → grayscale Uint8Array */
export function rgbaToGray(imageData) {
  const { data, width, height } = imageData;
  const g = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    g[p] = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) | 0;
  }
  return { gray: g, width, height };
}

function sampleGray(gray, w, h, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) {
    const xi = clamp(x0, 0, w - 1);
    const yi = clamp(y0, 0, h - 1);
    return gray[yi * w + xi];
  }
  const dx = x - x0;
  const dy = y - y0;
  const i00 = gray[y0 * w + x0];
  const i10 = gray[y0 * w + x0 + 1];
  const i01 = gray[(y0 + 1) * w + x0];
  const i11 = gray[(y0 + 1) * w + x0 + 1];
  return (
    (1 - dx) * (1 - dy) * i00 +
    dx * (1 - dy) * i10 +
    (1 - dx) * dy * i01 +
    dx * dy * i11
  );
}

/**
 * Single-point Lucas–Kanade tip track (desktop track_tip_lk).
 * @returns {{ tip:{x,y}, moved:boolean }}
 */
export function trackTipLk(prevGray, nextGray, w, h, tip) {
  if (!prevGray || !nextGray || !tip || prevGray.length !== nextGray.length) {
    return { tip, moved: false };
  }
  const half = (LK_WIN - 1) >> 1;
  let dx = 0;
  let dy = 0;
  for (let iter = 0; iter < 20; iter++) {
    let A11 = 0,
      A12 = 0,
      A22 = 0,
      b1 = 0,
      b2 = 0;
    let n = 0;
    for (let wy = -half; wy <= half; wy += 2) {
      for (let wx = -half; wx <= half; wx += 2) {
        const px = tip.x + wx;
        const py = tip.y + wy;
        if (px < 1 || py < 1 || px >= w - 1 || py >= h - 1) continue;
        if (px + dx < 1 || py + dy < 1 || px + dx >= w - 1 || py + dy >= h - 1)
          continue;
        const Ix =
          (sampleGray(prevGray, w, h, px + 1, py) -
            sampleGray(prevGray, w, h, px - 1, py)) *
          0.5;
        const Iy =
          (sampleGray(prevGray, w, h, px, py + 1) -
            sampleGray(prevGray, w, h, px, py - 1)) *
          0.5;
        const It =
          sampleGray(nextGray, w, h, px + dx, py + dy) -
          sampleGray(prevGray, w, h, px, py);
        A11 += Ix * Ix;
        A12 += Ix * Iy;
        A22 += Iy * Iy;
        b1 += Ix * It;
        b2 += Iy * It;
        n++;
      }
    }
    if (n < 6) break;
    const det = A11 * A22 - A12 * A12;
    if (Math.abs(det) < 1e-3) break;
    const u = (A22 * -b1 - A12 * -b2) / det;
    const v = (A11 * -b2 - A12 * -b1) / det;
    dx += u;
    dy += v;
    if (u * u + v * v < 1e-4) break;
  }
  let nx = tip.x + dx;
  let ny = tip.y + dy;
  const step = Math.hypot(dx, dy);
  if (step < 0.08) return { tip, moved: false };
  if (step > TIP_MAX_STEP) {
    const s = TIP_MAX_STEP / step;
    nx = tip.x + dx * s;
    ny = tip.y + dy * s;
  }
  return {
    tip: { x: clamp(nx, 0, w - 1), y: clamp(ny, 0, h - 1) },
    moved: true,
  };
}

function lkMedianDelta(prevGray, nextGray, w, h, pts, cropBox) {
  if (!prevGray || !nextGray || !pts || pts.length < 3) return null;
  const half = (LK_WIN - 1) >> 1;
  const deltas = [];
  for (const p of pts) {
    let dx = 0;
    let dy = 0;
    let ok = false;
    for (let iter = 0; iter < 12; iter++) {
      let A11 = 0,
        A12 = 0,
        A22 = 0,
        b1 = 0,
        b2 = 0;
      let n = 0;
      for (let wy = -half; wy <= half; wy += 2) {
        for (let wx = -half; wx <= half; wx += 2) {
          const px = p.x + wx;
          const py = p.y + wy;
          if (px < 1 || py < 1 || px >= w - 1 || py >= h - 1) continue;
          if (px + dx < 1 || py + dy < 1 || px + dx >= w - 1 || py + dy >= h - 1)
            continue;
          const Ix =
            (sampleGray(prevGray, w, h, px + 1, py) -
              sampleGray(prevGray, w, h, px - 1, py)) *
            0.5;
          const Iy =
            (sampleGray(prevGray, w, h, px, py + 1) -
              sampleGray(prevGray, w, h, px, py - 1)) *
            0.5;
          const It =
            sampleGray(nextGray, w, h, px + dx, py + dy) -
            sampleGray(prevGray, w, h, px, py);
          A11 += Ix * Ix;
          A12 += Ix * Iy;
          A22 += Iy * Iy;
          b1 += Ix * It;
          b2 += Iy * It;
          n++;
        }
      }
      if (n < 6) break;
      const det = A11 * A22 - A12 * A12;
      if (Math.abs(det) < 1e-3) break;
      const u = (A22 * -b1 - A12 * -b2) / det;
      const v = (A11 * -b2 - A12 * -b1) / det;
      dx += u;
      dy += v;
      ok = true;
      if (u * u + v * v < 1e-4) break;
    }
    if (!ok) continue;
    const nx = p.x + dx;
    const ny = p.y + dy;
    if (cropBox) {
      const [x1, y1, x2, y2] = cropBox;
      const pad = 12;
      if (nx < x1 - pad || nx > x2 + pad || ny < y1 - pad || ny > y2 + pad)
        continue;
    }
    deltas.push([dx, dy]);
  }
  if (deltas.length < 3) return null;
  const dxs = deltas.map((d) => d[0]).sort((a, b) => a - b);
  const dys = deltas.map((d) => d[1]).sort((a, b) => a - b);
  let mdx = dxs[Math.floor(dxs.length / 2)];
  let mdy = dys[Math.floor(dys.length / 2)];
  const step = Math.hypot(mdx, mdy);
  if (step > STICK_MAX_STEP) {
    const s = STICK_MAX_STEP / step;
    mdx *= s;
    mdy *= s;
  }
  return { x: mdx, y: mdy };
}

/** Matches tracking/landmark_stick.LandmarkStickTracker */
export class LandmarkStickTracker {
  constructor(nTrack = 20) {
    this.nTrack = nTrack;
    this.prevGray = null;
    this.trackIdx = null;
    this.absPts = null;
    this.ptsGen = -1;
  }

  reset() {
    this.prevGray = null;
    this.trackIdx = null;
    this.absPts = null;
    this.ptsGen = -1;
  }

  _seed(gray, pts, gen) {
    this.absPts = pts.map((p) => [p[0], p[1]]);
    const n = this.absPts.length;
    const k = Math.min(this.nTrack, Math.max(4, n - 1));
    this.trackIdx = [];
    for (let i = 0; i < k; i++) {
      this.trackIdx.push(
        Math.round((i * Math.min(54, n - 2)) / Math.max(1, k - 1))
      );
    }
    this.prevGray = gray;
    this.ptsGen = gen;
  }

  update(gray, w, h, workerPts, ptsGen, cropBox = null) {
    const trackPts = () =>
      this.trackIdx.map((i) => ({
        x: this.absPts[i][0],
        y: this.absPts[i][1],
      }));

    const lk =
      this.prevGray && this.absPts
        ? lkMedianDelta(this.prevGray, gray, w, h, trackPts(), cropBox)
        : null;

    if (workerPts != null && ptsGen !== this.ptsGen) {
      let pts = workerPts.map((p) => [p[0], p[1]]);
      if (lk && this.absPts && pts.length === this.absPts.length) {
        const predicted = this.absPts.map((p) => [p[0] + lk.x, p[1] + lk.y]);
        let shgCx = 0,
          shgCy = 0,
          predCx = 0,
          predCy = 0;
        for (let i = 0; i < pts.length; i++) {
          shgCx += pts[i][0];
          shgCy += pts[i][1];
          predCx += predicted[i][0];
          predCy += predicted[i][1];
        }
        const inv = 1 / pts.length;
        const ox = shgCx * inv - predCx * inv;
        const oy = shgCy * inv - predCy * inv;
        for (let i = 0; i < predicted.length; i++) {
          predicted[i][0] += ox;
          predicted[i][1] += oy;
          pts[i][0] = (1 - SHG_BLEND) * pts[i][0] + SHG_BLEND * predicted[i][0];
          pts[i][1] = (1 - SHG_BLEND) * pts[i][1] + SHG_BLEND * predicted[i][1];
        }
      }
      this._seed(gray, pts, ptsGen);
      return this.absPts.map((p) => [p[0], p[1]]);
    }

    if (!this.prevGray || !this.absPts) {
      if (workerPts) this._seed(gray, workerPts, ptsGen);
      return this.absPts ? this.absPts.map((p) => [p[0], p[1]]) : null;
    }

    if (!lk) {
      this.prevGray = gray;
      return this.absPts.map((p) => [p[0], p[1]]);
    }

    this.absPts = this.absPts.map((p) => [p[0] + lk.x, p[1] + lk.y]);
    this.prevGray = gray;
    return this.absPts.map((p) => [p[0], p[1]]);
  }
}
