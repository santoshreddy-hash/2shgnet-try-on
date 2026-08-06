/**
 * Earring sprites from uploaded images + improved pendulum physics.
 *
 * Stud (anchor) locks to piercing #56. Top stud→joint fixed;
 * dangling part rotates about joint. Pointer drag swings the dangle.
 */

/**
 * Build sprite from an HTMLImageElement / ImageBitmap / canvas.
 * Anchor = top-center of opaque content (stud). Joint slightly below.
 * Near-white backgrounds are keyed out for typical product photos.
 */
export async function spriteFromImage(source, opts = {}) {
  const img = await ensureImage(source);
  const tw = img.naturalWidth || img.width;
  const th = img.naturalHeight || img.height;
  if (!tw || !th) throw new Error("Empty image");

  const maxSide = opts.maxSide || 360;
  const scale = Math.min(1, maxSide / Math.max(tw, th));
  const w = Math.max(1, Math.round(tw * scale));
  const h = Math.max(1, Math.round(th * scale));

  const raw = document.createElement("canvas");
  raw.width = w;
  raw.height = h;
  const rctx = raw.getContext("2d", { willReadFrequently: true });
  rctx.drawImage(img, 0, 0, w, h);

  // Key near-white / light gray backgrounds common in product shots
  const id = rctx.getImageData(0, 0, w, h);
  const d = id.data;
  const keyWhite = opts.keyWhite !== false;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 8) continue;
    if (keyWhite) {
      const r = d[i],
        g = d[i + 1],
        b = d[i + 2];
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      if (mx > 232 && mn > 210 && mx - mn < 28) {
        d[i + 3] = 0;
      }
    }
  }
  rctx.putImageData(id, 0, 0);

  // Bounding box of opaque pixels
  let x0 = w,
    y0 = h,
    x1 = 0,
    y1 = 0;
  let found = false;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = d[(y * w + x) * 4 + 3];
      if (a > 20) {
        found = true;
        if (x < x0) x0 = x;
        if (y < y0) y0 = y;
        if (x > x1) x1 = x;
        if (y > y1) y1 = y;
      }
    }
  }
  if (!found) {
    x0 = 0;
    y0 = 0;
    x1 = w - 1;
    y1 = h - 1;
  }

  const pad = 6;
  const cw = x1 - x0 + 1 + pad * 2;
  const ch = y1 - y0 + 1 + pad * 2;
  const canvas = document.createElement("canvas");
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(raw, x0, y0, x1 - x0 + 1, y1 - y0 + 1, pad, pad, x1 - x0 + 1, y1 - y0 + 1);

  // Stud = top-center of content; scan first opaque rows for center of mass X
  let topY = pad;
  let sumX = 0,
    n = 0;
  const cd = ctx.getImageData(0, 0, cw, ch).data;
  outer: for (let y = 0; y < Math.min(ch, pad + 24); y++) {
    for (let x = 0; x < cw; x++) {
      if (cd[(y * cw + x) * 4 + 3] > 40) {
        topY = y;
        break outer;
      }
    }
  }
  const band = Math.min(ch - 1, topY + 10);
  for (let y = topY; y <= band; y++) {
    for (let x = 0; x < cw; x++) {
      if (cd[(y * cw + x) * 4 + 3] > 40) {
        sumX += x;
        n++;
      }
    }
  }
  const anchorX = n ? sumX / n : cw / 2;
  const anchorY = topY + 2;
  const contentH = ch - pad - topY;
  const jointY = anchorY + Math.max(8, contentH * 0.12);

  return {
    canvas,
    anchorX,
    anchorY,
    jointY,
    width: cw,
    height: ch,
    name: opts.name || "Uploaded earring",
    fromUpload: true,
  };
}

function ensureImage(source) {
  if (source instanceof HTMLImageElement && source.complete && source.naturalWidth) {
    return Promise.resolve(source);
  }
  if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) {
    return Promise.resolve(source);
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    if (typeof source === "string") img.src = source;
    else if (source instanceof Blob) img.src = URL.createObjectURL(source);
    else reject(new Error("Unsupported image source"));
  });
}

/** Minimal default drop so studio works before upload. */
export function buildDefaultSprite() {
  const w = 48;
  const h = 96;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const ax = w / 2;
  const ay = 8;
  const jointY = ay + 14;

  const g = ctx.createRadialGradient(ax - 1, ay - 1, 0.5, ax, ay, 5);
  g.addColorStop(0, "#f2e6c9");
  g.addColorStop(0.5, "#c9a45c");
  g.addColorStop(1, "#6a5428");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(ax, ay, 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#c9a45c";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(ax, ay + 4);
  ctx.lineTo(ax, jointY);
  ctx.stroke();

  ctx.save();
  ctx.translate(ax, jointY);
  const rg = ctx.createLinearGradient(-8, 8, 8, 70);
  rg.addColorStop(0, "#f2e6c9");
  rg.addColorStop(0.5, "#c9a45c");
  rg.addColorStop(1, "#7a5c28");
  ctx.fillStyle = rg;
  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.bezierCurveTo(12, 22, 10, 55, 0, 70);
  ctx.bezierCurveTo(-10, 55, -12, 22, 0, 6);
  ctx.fill();
  ctx.restore();

  return {
    canvas,
    anchorX: ax,
    anchorY: ay,
    jointY,
    width: w,
    height: h,
    name: "Default drop",
    fromUpload: false,
  };
}

/**
 * Improved pendulum + pointer drag.
 * Drag sets a target angle; release keeps momentum and settles under gravity.
 */
export class EarringPhysics {
  constructor() {
    this.theta = 0;
    this.omega = 0;
    this.prevPx = null;
    this.prevPy = null;
    this.prevVx = 0;
    this.prevVy = 0;
    this.mode = "classic";
    this.freedom = 0.7;
    this.dragging = false;
    this.dragTarget = 0;
    this._dragSamples = [];
  }

  setMode(mode) {
    this.mode = mode === "subtle" ? "subtle" : "classic";
  }

  setFreedom(v) {
    this.freedom = Math.max(0, Math.min(1, Number(v) || 0));
  }

  reset() {
    this.theta = 0;
    this.omega = 0;
    this.prevPx = null;
    this.prevPy = null;
    this.prevVx = 0;
    this.prevVy = 0;
    this.dragging = false;
    this._dragSamples = [];
  }

  beginDrag(targetTheta) {
    this.dragging = true;
    this.dragTarget = targetTheta;
    this.omega = 0;
    this._dragSamples = [{ t: performance.now(), th: targetTheta }];
  }

  updateDrag(targetTheta) {
    if (!this.dragging) return;
    this.dragTarget = targetTheta;
    const t = performance.now();
    this._dragSamples.push({ t, th: targetTheta });
    while (this._dragSamples.length > 8) this._dragSamples.shift();
  }

  endDrag() {
    if (!this.dragging) return;
    this.dragging = false;
    // Impart angular velocity from recent drag motion
    const s = this._dragSamples;
    if (s.length >= 2) {
      const a = s[0];
      const b = s[s.length - 1];
      const dt = Math.max(1e-3, (b.t - a.t) / 1000);
      this.omega = (b.th - a.th) / dt;
      const cap = 8 + 10 * this.freedom;
      this.omega = Math.max(-cap, Math.min(cap, this.omega));
    }
    this._dragSamples = [];
  }

  /**
   * @param {number} px pierce x
   * @param {number} py pierce y
   * @param {number} dt seconds
   */
  step(px, py, dt) {
    const d = Math.max(1e-3, Math.min(0.05, dt || 1 / 30));
    let ax = 0;
    let ay = 0;
    if (this.prevPx != null) {
      const vx = (px - this.prevPx) / d;
      const vy = (py - this.prevPy) / d;
      ax = (vx - this.prevVx) / d;
      ay = (vy - this.prevVy) / d;
      // Low-pass accel a bit to reduce jitter
      ax = 0.55 * ax + 0.45 * (this._lax || 0);
      ay = 0.55 * ay + 0.45 * (this._lay || 0);
      this._lax = ax;
      this._lay = ay;
      this.prevVx = vx;
      this.prevVy = vy;
    } else {
      this.prevVx = 0;
      this.prevVy = 0;
      this._lax = 0;
      this._lay = 0;
    }
    this.prevPx = px;
    this.prevPy = py;

    const classic = this.mode === "classic";
    const maxAng = (classic ? 0.95 : 0.42) * (0.2 + 0.8 * this.freedom);

    if (this.dragging) {
      // Soft follow drag direction (critically-damped-ish spring)
      const stiff = classic ? 48 : 70;
      const damp = classic ? 14 : 20;
      const err = this.dragTarget - this.theta;
      // wrap-ish not needed for small angles
      const alpha = stiff * err - damp * this.omega;
      this.omega += alpha * d;
      this.theta += this.omega * d;
      this.theta = Math.max(-maxAng * 1.15, Math.min(maxAng * 1.15, this.theta));
      return this.theta;
    }

    // Gravity restoring + pierce acceleration drive (horizontal + slight vertical)
    const g = classic ? 26 : 40;
    const damp = (classic ? 0.85 : 2.4) * (1.15 - 0.4 * this.freedom);
    const driveX = (classic ? 0.0042 : 0.0018) * this.freedom;
    const driveY = (classic ? 0.0011 : 0.0004) * this.freedom;
    // Extra kick from pierce velocity (natural “lag behind” feel)
    const velKick = (classic ? 0.012 : 0.005) * this.freedom;

    const alpha =
      -g * Math.sin(this.theta) -
      damp * this.omega -
      driveX * ax -
      driveY * ay * Math.sin(this.theta) -
      velKick * (this.prevVx || 0);

    this.omega += alpha * d;
    this.theta += this.omega * d;

    // Soft wall at max angle
    if (this.theta > maxAng) {
      this.theta = maxAng;
      this.omega *= -0.25;
    } else if (this.theta < -maxAng) {
      this.theta = -maxAng;
      this.omega *= -0.25;
    }
    // Tiny rest snap
    if (Math.abs(this.theta) < 0.008 && Math.abs(this.omega) < 0.15) {
      this.theta *= 0.9;
      this.omega *= 0.85;
    }
    return this.theta;
  }
}

/** Joint position in canvas space (fixed relative to pierce). */
export function jointScreen(sprite, px, py, scale) {
  const s = scale || 1;
  return {
    x: px,
    y: py + (sprite.jointY - sprite.anchorY) * s,
  };
}

/** Angle from vertical-down toward pointer (atan2(dx, dy)). */
export function angleFromJoint(jx, jy, mx, my) {
  return Math.atan2(mx - jx, my - jy);
}

/**
 * Hit-test the swinging body (from joint along current theta).
 */
export function hitDangle(sprite, px, py, theta, scale, mx, my) {
  if (!sprite) return false;
  const s = scale || 1;
  const j = jointScreen(sprite, px, py, s);
  const len = (sprite.height - sprite.jointY) * s;
  const halfW = Math.max(18, (sprite.width * 0.45) * s);
  // Point in dangle-local coords (rotate by -theta about joint)
  const dx = mx - j.x;
  const dy = my - j.y;
  const c = Math.cos(-theta);
  const sn = Math.sin(-theta);
  const lx = dx * c - dy * sn;
  const ly = dx * sn + dy * c;
  return ly >= -8 && ly <= len + 12 && Math.abs(lx) <= halfW + 10;
}

/**
 * Draw earring: stud→joint fixed; bottom rotates about joint.
 */
export function drawEarring(ctx, sprite, px, py, theta, scale = 1) {
  if (!sprite || !Number.isFinite(px) || !Number.isFinite(py)) return;
  const s = scale;
  const ax = sprite.anchorX;
  const ay = sprite.anchorY;
  const jy = sprite.jointY;

  ctx.save();
  ctx.translate(px, py);
  ctx.scale(s, s);
  ctx.translate(-ax, -ay);

  // Fixed top
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, sprite.width, jy + 0.5);
  ctx.clip();
  ctx.drawImage(sprite.canvas, 0, 0);
  ctx.restore();

  // Swinging bottom
  ctx.save();
  ctx.translate(ax, jy);
  ctx.rotate(theta);
  ctx.translate(-ax, -jy);
  ctx.beginPath();
  ctx.rect(0, jy - 0.5, sprite.width, sprite.height - jy + 1);
  ctx.clip();
  ctx.drawImage(sprite.canvas, 0, 0);
  ctx.restore();

  ctx.restore();
}

/** Map client/pointer coords → canvas pixel coords. */
export function clientToCanvas(canvas, clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / Math.max(1, r.width);
  const sy = canvas.height / Math.max(1, r.height);
  return {
    x: (clientX - r.left) * sx,
    y: (clientY - r.top) * sy,
  };
}
