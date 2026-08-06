/**
 * Ear crop / landmark gates — must match train/config.py (desktop live).
 * Adaptive performance must NOT change these constants.
 */

export const CROP_PAD = 2.15; // train.config.CROP_PAD
export const REFINE_PAD = 1.35;
export const EAR_KEYPOINT_MIN_CONF = 0.15; // train.config.EAR_KEYPOINT_MIN_CONF
export const YOLO_BOX_CONF = 0.25;
export const MIN_SHG_SCORE = 0.08; // performance_profiles quality.min_shg_score
export const PIERCING_INDEX = 55;

/** Pinna height — match working reference web/infer.js */
export function pinnaHeight(yolo, vw, vh) {
  const fmin = Math.min(vw, vh);
  const tip = yolo.tip;
  const cands = [];
  let tipNose = null;
  if (yolo.nose?.c >= 0.2) {
    tipNose = Math.hypot(tip.x - yolo.nose.x, tip.y - yolo.nose.y);
    if (tipNose > fmin * 0.03) cands.push(tipNose * 0.55);
  }
  if (yolo.eyeDist && yolo.eyeDist > fmin * 0.02) cands.push(yolo.eyeDist * 0.9);
  const [, y1, , y2] = yolo.bbox || [0, 0, 0, 0];
  const bh = y2 - y1;
  if (bh > 1) cands.push(bh * 0.12);
  if (!cands.length) return Math.max(40, fmin * 0.12);
  cands.sort((a, b) => a - b);
  let h =
    cands.length === 1 ? cands[0] : cands[Math.floor(cands.length / 2)];
  if (tipNose != null && tipNose > 1) h = Math.min(h, tipNose * 0.7);
  return Math.max(40, Math.min(fmin * 0.2, h));
}

/** Reject frontal / dual-ear ambiguity — hide landmarks during L↔R turns. */
export function isSideProfile(yolo) {
  if (!yolo?.tip) return false;
  const conf = yolo.earConf ?? yolo.ear?.c ?? 0;
  if (conf < EAR_KEYPOINT_MIN_CONF) return false;

  // Both ears strong → head is frontal / turning — not a clean single-ear lock
  if (yolo.earOtherConf != null && yolo.earConf != null) {
    if (yolo.earOtherConf >= 0.35 && yolo.earOtherConf >= yolo.earConf * 0.7)
      return false;
  }

  if (!yolo.nose || yolo.nose.c < 0.2) return true;
  const dx = Math.abs(yolo.tip.x - yolo.nose.x);
  const d = Math.hypot(yolo.tip.x - yolo.nose.x, yolo.tip.y - yolo.nose.y);
  // Tip near nose = face / transition, not a side ear
  if (dx < 28 || d < 36) return false;
  // Tip must be clearly lateral vs nose for a usable side profile
  if (dx < d * 0.55) return false;
  return true;
}

export function medial(yolo, tip, side, vw) {
  if (yolo.nose?.c >= 0.2) {
    const vx = yolo.nose.x - tip.x;
    const vy = yolo.nose.y - tip.y;
    const n = Math.hypot(vx, vy);
    if (n > 1e-3) return [vx / n, vy / n];
  }
  const vx = 0.5 * vw - tip.x;
  if (Math.abs(vx) > 1e-3) return [Math.sign(vx), 0];
  return [side === "LEFT" ? -1 : 1, 0];
}

/** Jewellery landmark gate — tip near cloud, correct scale */
export function landmarksOk(pts, tipPt, sidePx) {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  const bw = x1 - x0;
  const bh = y1 - y0;
  const span = Math.max(bw, bh);
  const ratio = span / Math.max(1, sidePx);
  if (ratio < 0.4 || ratio > 0.88) return false;
  if (Math.min(bw, bh) < span * 0.28) return false;
  let mx = 0,
    my = 0;
  for (const [x, y] of pts) {
    mx += x;
    my += y;
  }
  mx /= pts.length;
  my /= pts.length;
  if (Math.hypot(mx - tipPt.x, my - tipPt.y) > sidePx * 0.45) return false;
  const padX = 0.08 * bw;
  const padY = 0.08 * bh;
  if (tipPt.x < x0 - padX || tipPt.x > x1 + padX) return false;
  if (tipPt.y < y0 - padY || tipPt.y > y1 + padY) return false;
  return true;
}

export function pierceQuality(pts, tipPt, sidePx, score) {
  const pierce = pts[Math.min(PIERCING_INDEX, pts.length - 1)];
  if (!pierce) return score * 0.5;
  const tipD = Math.hypot(pierce[0] - tipPt.x, pierce[1] - tipPt.y);
  const below = Math.max(0, pierce[1] - tipPt.y) / Math.max(1, sidePx);
  const ratio = tipD / Math.max(1, sidePx);
  const ratioScore = 1.0 - Math.min(1.0, Math.abs(ratio - 0.28) / 0.28);
  const belowScore = Math.max(0, Math.min(1, below / 0.2));
  const okBonus = landmarksOk(pts, tipPt, sidePx) ? 1 : 0;
  return 0.45 * score + 0.25 * ratioScore + 0.2 * belowScore + 0.1 * okBonus;
}

/** Tip-centered crop — jewellery: light medial + slight down (0.06) */
export function tipCropCenter(tipPt, pinna, yolo, side, vw) {
  const [mx] = medial(yolo, tipPt, side, vw);
  return {
    ncx: tipPt.x + mx * (0.1 * pinna),
    ncy: tipPt.y + 0.06 * pinna,
    mx,
  };
}

export function rescueCropCenter(tipPt, pinna, mx) {
  return {
    cx: tipPt.x + mx * (0.1 * pinna),
    cy: tipPt.y + 0.06 * pinna,
  };
}

export function blendRawRel(oldRel, newRel) {
  if (!oldRel || oldRel.length !== newRel.length) return newRel;
  return newRel.map((p, i) => [
    0.55 * oldRel[i][0] + 0.45 * p[0],
    0.55 * oldRel[i][1] + 0.45 * p[1],
  ]);
}
