/** One Euro landmarks — steady at rest, tracks ear when head moves. */
export class OneEuro1D {
  constructor(minCutoff = 0.8, beta = 0.121, dCutoff = 1.19) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = null;
  }

  static alpha(cutoff, dt) {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / Math.max(dt, 1e-6));
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = null;
  }

  filter(x, dt) {
    if (this.xPrev == null) {
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / Math.max(dt, 1e-6);
    const aD = OneEuro1D.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuro1D.alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }
}

export class OneEuroLandmarks {
  constructor(
    n = 55,
    minCutoff = 0.8,
    beta = 0.121,
    dCutoff = 1.19,
    restSpeedPx = 5,
    restHoldFrames = 3,
    restReleaseMult = 2.0
  ) {
    this.n = n;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.fx = Array.from({ length: n }, () => new OneEuro1D(minCutoff, beta, dCutoff));
    this.fy = Array.from({ length: n }, () => new OneEuro1D(minCutoff, beta, dCutoff));
    this.side = null;
    this.lastOut = null;
    this.restFrames = 0;
    this.restSpeedPx = restSpeedPx;
    this.restHoldFrames = restHoldFrames;
    this.restReleaseMult = restReleaseMult;
    this.frozen = false;
  }

  setParams({ minCutoff, beta, dCutoff } = {}) {
    if (minCutoff != null) this.minCutoff = Number(minCutoff);
    if (beta != null) this.beta = Number(beta);
    if (dCutoff != null) this.dCutoff = Number(dCutoff);
    for (const f of this.fx) {
      f.minCutoff = this.minCutoff;
      f.beta = this.beta;
      f.dCutoff = this.dCutoff;
    }
    for (const f of this.fy) {
      f.minCutoff = this.minCutoff;
      f.beta = this.beta;
      f.dCutoff = this.dCutoff;
    }
  }

  reset() {
    for (const f of this.fx) f.reset();
    for (const f of this.fy) f.reset();
    this.side = null;
    this.lastOut = null;
    this.restFrames = 0;
    this.frozen = false;
  }

  update(pts, dt, side = null, { maxStepPx = 14, snap = false } = {}) {
    if (side && this.side && side !== this.side) this.reset();
    if (side) this.side = side;

    const n = Math.min(pts.length, this.n);
    let cur = pts.map((p) => [p[0], p[1]]);

    if (snap || this.lastOut == null) {
      for (let i = 0; i < n; i++) {
        this.fx[i].xPrev = cur[i][0];
        this.fy[i].xPrev = cur[i][1];
        this.fx[i].dxPrev = 0;
        this.fy[i].dxPrev = 0;
      }
      this.lastOut = cur.slice(0, n).map((p) => [p[0], p[1]]);
      this.restFrames = 0;
      this.frozen = false;
      return this.lastOut.map((p) => [p[0], p[1]]);
    }

    if (maxStepPx > 0) {
      for (let i = 0; i < n; i++) {
        const dx = cur[i][0] - this.lastOut[i][0];
        const dy = cur[i][1] - this.lastOut[i][1];
        const dist = Math.hypot(dx, dy);
        if (dist > maxStepPx) {
          const s = maxStepPx / dist;
          cur[i][0] = this.lastOut[i][0] + dx * s;
          cur[i][1] = this.lastOut[i][1] + dy * s;
        }
      }
    }

    if (this.restSpeedPx > 0) {
      const speeds = [];
      for (let i = 0; i < n; i++) {
        speeds.push(
          Math.hypot(cur[i][0] - this.lastOut[i][0], cur[i][1] - this.lastOut[i][1]) /
            Math.max(dt, 1e-6)
        );
      }
      speeds.sort((a, b) => a - b);
      const med = speeds[Math.floor(speeds.length / 2)];
      const release = this.restSpeedPx * this.restReleaseMult;

      if (this.frozen) {
        if (med < release) {
          for (let i = 0; i < n; i++) {
            this.fx[i].xPrev = this.lastOut[i][0];
            this.fy[i].xPrev = this.lastOut[i][1];
            this.fx[i].dxPrev = 0;
            this.fy[i].dxPrev = 0;
          }
          return this.lastOut.map((p) => [p[0], p[1]]);
        }
        this.frozen = false;
        this.restFrames = 0;
      } else if (med < this.restSpeedPx) {
        this.restFrames++;
        if (this.restFrames >= this.restHoldFrames) {
          this.frozen = true;
          for (let i = 0; i < n; i++) {
            this.fx[i].xPrev = this.lastOut[i][0];
            this.fy[i].xPrev = this.lastOut[i][1];
            this.fx[i].dxPrev = 0;
            this.fy[i].dxPrev = 0;
          }
          return this.lastOut.map((p) => [p[0], p[1]]);
        }
      } else {
        this.restFrames = 0;
      }
    }

    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = [
        this.fx[i].filter(cur[i][0], dt),
        this.fy[i].filter(cur[i][1], dt),
      ];
    }
    this.lastOut = out;
    return out;
  }

  /** Smooth shape in tip-relative space; output stays locked to tip. */
  updateRelative(pts, tip, dt, side = null, { maxStepPx = 8, snap = false } = {}) {
    const tx = tip.x ?? tip[0];
    const ty = tip.y ?? tip[1];
    const rel = pts.map(([x, y]) => [x - tx, y - ty]);
    const relSmooth = this.update(rel, dt, side, { maxStepPx, snap });
    return relSmooth.map(([x, y]) => [x + tx, y + ty]);
  }
}
