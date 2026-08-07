/**
 * Desktop-matching tip stick: Lucas–Kanade every display frame.
 * Same role as track_tip_lk in live/desktop_onnx.py — keeps tip glued so
 * tip-relative landmarks move with the ear between YOLO updates.
 */

const WIN = 7; // half-window → 15×15
const MAX_LEVEL = 2;
const MAX_ITER = 7;
const EPS = 0.015;
const TIP_MAX_STEP = 12;
// Match desktop track_tip_lk soft blend
const BLEND = 0.65;
const ROI_HALF = 56;

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x0
 * @param {number} y0
 * @param {number} w
 * @param {number} h
 */
function grabGrayRect(ctx, x0, y0, w, h) {
  const cw = ctx.canvas.width;
  const ch = ctx.canvas.height;
  if (x0 < 0 || y0 < 0 || x0 + w > cw || y0 + h > ch || w < 20 || h < 20) return null;
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

function down2(g, w, h) {
  const nw = w >> 1;
  const nh = h >> 1;
  const out = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const i = y * 2 * w + x * 2;
      out[y * nw + x] = (g[i] + g[i + 1] + g[i + w] + g[i + w + 1]) * 0.25;
    }
  }
  return { g: out, w: nw, h: nh };
}

function lkLevel(prev, next, px, py, guessX, guessY) {
  let gx = guessX;
  let gy = guessY;
  for (let it = 0; it < MAX_ITER; it++) {
    let ix = 0;
    let iy = 0;
    let ixx = 0;
    let ixy = 0;
    let iyy = 0;
    let n = 0;
    for (let wy = -WIN; wy <= WIN; wy++) {
      for (let wx = -WIN; wx <= WIN; wx++) {
        const x1 = px + wx;
        const y1 = py + wy;
        const i1 = sample(prev.g, prev.w, prev.h, x1, y1);
        const i2 = sample(next.g, next.w, next.h, x1 + gx + wx, y1 + gy + wy);
        if (i1 == null || i2 == null) continue;
        const ixp =
          (sample(prev.g, prev.w, prev.h, x1 + 1, y1) ?? i1) -
          (sample(prev.g, prev.w, prev.h, x1 - 1, y1) ?? i1);
        const iyp =
          (sample(prev.g, prev.w, prev.h, x1, y1 + 1) ?? i1) -
          (sample(prev.g, prev.w, prev.h, x1, y1 - 1) ?? i1);
        const diff = i1 - i2;
        ix += ixp * diff;
        iy += iyp * diff;
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

function trackAligned(prev, next, tipX, tipY) {
  const plx = tipX - prev.x0;
  const ply = tipY - prev.y0;
  if (plx < WIN + 2 || ply < WIN + 2 || plx > prev.w - WIN - 2 || ply > prev.h - WIN - 2) {
    return { x: tipX, y: tipY, ok: false };
  }

  const pyramidsP = [{ g: prev.g, w: prev.w, h: prev.h }];
  const pyramidsN = [{ g: next.g, w: next.w, h: next.h }];
  for (let l = 0; l < MAX_LEVEL; l++) {
    const p = pyramidsP[l];
    if (p.w < 28 || p.h < 28) break;
    pyramidsP.push(down2(p.g, p.w, p.h));
    pyramidsN.push(down2(pyramidsN[l].g, pyramidsN[l].w, pyramidsN[l].h));
  }

  const levels = Math.min(pyramidsP.length, pyramidsN.length);
  let dx = 0;
  let dy = 0;
  for (let lvl = levels - 1; lvl >= 0; lvl--) {
    const scale = 1 << lvl;
    const res = lkLevel(
      pyramidsP[lvl],
      pyramidsN[lvl],
      plx / scale,
      ply / scale,
      dx,
      dy
    );
    if (!res) {
      if (lvl === levels - 1) return { x: tipX, y: tipY, ok: false };
      dx *= 2;
      dy *= 2;
      continue;
    }
    dx = res.dx;
    dy = res.dy;
    if (lvl > 0) {
      dx *= 2;
      dy *= 2;
    }
  }

  let step = Math.hypot(dx, dy);
  if (step < 0.12) return { x: tipX, y: tipY, ok: false };
  if (step > TIP_MAX_STEP) {
    const s = TIP_MAX_STEP / step;
    dx *= s;
    dy *= s;
  }

  let nx = tipX + dx;
  let ny = tipY + dy;
  nx = (1 - BLEND) * tipX + BLEND * nx;
  ny = (1 - BLEND) * tipY + BLEND * ny;
  return { x: nx, y: ny, ok: true };
}

export class TipLkTracker {
  constructor() {
    /** @type {{ g: Float32Array, w: number, h: number, x0: number, y0: number } | null} */
    this.prev = null;
  }

  reset() {
    this.prev = null;
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {{x:number,y:number}} tip
   */
  update(ctx, tip) {
    const cw = ctx.canvas.width;
    const ch = ctx.canvas.height;
    if (!cw || !ch) return { x: tip.x, y: tip.y, ok: false };

    // Fixed ROI origin from previous tip so prev/next grayscale align
    let x0;
    let y0;
    let side;
    if (this.prev) {
      x0 = this.prev.x0;
      y0 = this.prev.y0;
      side = this.prev.w;
      // Recentering when tip drifts near ROI edge
      if (
        tip.x < x0 + WIN * 2 ||
        tip.y < y0 + WIN * 2 ||
        tip.x > x0 + side - WIN * 2 ||
        tip.y > y0 + side - WIN * 2
      ) {
        side = Math.min(ROI_HALF * 2, Math.min(cw, ch));
        x0 = Math.max(0, Math.min(cw - side, Math.round(tip.x - side * 0.5)));
        y0 = Math.max(0, Math.min(ch - side, Math.round(tip.y - side * 0.5)));
        this.prev = null;
      }
    } else {
      side = Math.min(ROI_HALF * 2, Math.min(cw, ch));
      x0 = Math.max(0, Math.min(cw - side, Math.round(tip.x - side * 0.5)));
      y0 = Math.max(0, Math.min(ch - side, Math.round(tip.y - side * 0.5)));
    }

    const next = grabGrayRect(ctx, x0, y0, side, side);
    if (!next) {
      this.prev = null;
      return { x: tip.x, y: tip.y, ok: false };
    }

    let tracked = { x: tip.x, y: tip.y, ok: false };
    if (this.prev && this.prev.w === next.w && this.prev.h === next.h) {
      tracked = trackAligned(this.prev, next, tip.x, tip.y);
    }
    this.prev = next;
    return tracked;
  }
}
