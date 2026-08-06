/** One Euro landmarks — steady at rest, tracks ear when head moves. */
export class OneEuro1D {
  constructor(minCutoff = 1.8, beta = 0.85, dCutoff = 1.19) {
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
    n = 56,
    minCutoff = 1.8,
    beta = 0.85,
    dCutoff = 1.19,
    restSpeedPx = 1.5,
    restHoldFrames = 2,
    restReleaseMult = 1.5
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
    this.restHoldFrames = Math.max(1, Math.floor(restHoldFrames));
    this.restReleaseMult = Math.max(1.1, Number(restReleaseMult));
    this.frozen = false;
  }

  /** Apply jewellery / desktop one_euro_settings.json fields. */
  applySettings(s = {}) {
    this.setParams({
      minCutoff: s.min_cutoff ?? s.minCutoff,
      beta: s.beta,
      dCutoff: s.d_cutoff ?? s.dCutoff,
    });
    if (s.rest_speed_px != null || s.restSpeedPx != null) {
      this.restSpeedPx = Number(s.rest_speed_px ?? s.restSpeedPx);
    }
    if (s.rest_hold_frames != null || s.restHoldFrames != null) {
      this.restHoldFrames = Math.max(
        1,
        Math.floor(Number(s.rest_hold_frames ?? s.restHoldFrames))
      );
    }
    if (s.rest_release_mult != null || s.restReleaseMult != null) {
      this.restReleaseMult = Math.max(
        1.1,
        Number(s.rest_release_mult ?? s.restReleaseMult)
      );
    }
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

      // Rest freeze only damps tip-relative shape jitter — tip is never frozen.
      if (this.frozen) {
        if (med < release) {
          return this.lastOut.map((p) => [p[0], p[1]]);
        }
        this.frozen = false;
        this.restFrames = 0;
      } else if (med < this.restSpeedPx) {
        this.restFrames++;
        if (this.restFrames >= this.restHoldFrames) {
          this.frozen = true;
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

  /** Smooth tip-relative offsets only; reconstruct with live tip (never lag tip). */
  updateRelative(pts, tip, dt, side = null, { maxStepPx = 42, snap = false } = {}) {
    const tx = tip.x ?? tip[0];
    const ty = tip.y ?? tip[1];
    const rel = pts.map(([x, y]) => [x - tx, y - ty]);
    const relSmooth = this.filterOffsets(rel, dt, side, { maxStepPx, snap });
    return relSmooth.map(([x, y]) => [x + tx, y + ty]);
  }

  /** One Euro on tip-relative offsets only (shape). */
  filterOffsets(rel, dt, side = null, { maxStepPx = 42, snap = false } = {}) {
    return this.update(rel, dt, side, { maxStepPx, snap });
  }

  /** landmarks = latestTip + offsets (zero tip latency). */
  compose(tip, rel) {
    const tx = tip.x ?? tip[0];
    const ty = tip.y ?? tip[1];
    return rel.map(([x, y]) => [x + tx, y + ty]);
  }

  /**
   * Align filter state to a tip-relative shape without introducing lag.
   * Used by rigid tip-hold so the next SHG update doesn't jump.
   */
  syncRelative(rel) {
    const n = Math.min(rel.length, this.n);
    this.lastOut = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = rel[i][0];
      const y = rel[i][1];
      this.lastOut[i] = [x, y];
      this.fx[i].xPrev = x;
      this.fy[i].xPrev = y;
      this.fx[i].dxPrev = 0;
      this.fy[i].dxPrev = 0;
    }
    this.restFrames = 0;
    this.frozen = false;
  }
}
