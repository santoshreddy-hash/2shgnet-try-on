/**
 * Ear crop / landmark gates — must match train/crop.py (desktop live).
 * Adaptive performance must NOT change these constants.
 */

export const CROP_PAD = 2.15; // train.config.CROP_PAD
export const REFINE_PAD = 1.35;
export const EAR_KEYPOINT_MIN_CONF = 0.15; // train.config.EAR_KEYPOINT_MIN_CONF
/** Match desktop EarCropper / ultralytics tip search */
export const YOLO_BOX_CONF = 0.12;
export const MIN_SHG_SCORE = 0.08; // performance_profiles quality.min_shg_score
export const PIERCING_INDEX = 55;

/** Port of train/crop.py estimate_pinna_h — ear-sized, not face-sized. */
export function pinnaHeight(yolo, vw, vh) {
  const fmin = Math.min(vw, vh);
  const tip = yolo.tip;
  const cands = [];
  let tipNose = null;

  if (yolo.nose?.c >= 0.2) {
    tipNose = Math.hypot(tip.x - yolo.nose.x, tip.y - yolo.nose.y);
    if (tipNose > fmin * 0.04) cands.push(tipNose * 0.58);
  }
  if (yolo.eyeDist && yolo.eyeDist > fmin * 0.02) {
    cands.push(yolo.eyeDist * 1.0);
  }
  const [x1, y1, x2, y2] = yolo.bbox || [0, 0, 0, 0];
  const bh = y2 - y1;
  const bw = x2 - x1;
  if (bh > 1) cands.push(bh * 0.12);
  if (bw > 1) cands.push(bw * 0.16);

  if (!cands.length) return Math.max(44, fmin * 0.12);
  cands.sort((a, b) => a - b);
  let pinna =
    cands.length === 1 ? cands[0] : cands[Math.floor(cands.length / 2)];
  if (tipNose != null && tipNose > fmin * 0.06) {
    pinna = Math.min(pinna, tipNose * 0.7);
  }
  return Math.max(44, Math.min(0.22 * fmin, pinna));
}

/** Port of train/crop.py is_side_profile */
export function isSideProfile(yolo) {
  if (!yolo?.tip) return false;
  const conf = yolo.earConf ?? yolo.ear?.c ?? 0;
  if (conf < EAR_KEYPOINT_MIN_CONF) return false;

  if (yolo.earOtherConf != null && yolo.earConf != null) {
    if (yolo.earOtherConf >= 0.35 && yolo.earOtherConf >= yolo.earConf * 0.7) {
      return false;
    }
  }

  if (!yolo.nose || yolo.nose.c < 0.2) return true;
  const dx = Math.abs(yolo.tip.x - yolo.nose.x);
  const d = Math.hypot(yolo.tip.x - yolo.nose.x, yolo.tip.y - yolo.nose.y);
  // Slightly softer than desktop — browser cams are often 3/4, not pure side
  if (dx < 22 || d < 30) return false;
  if (dx < d * 0.48) return false;
  return true;
}

/** Port of train/crop.py medial_unit */
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

/** Port of train/crop.py landmarks_ok (WASM soft-argmax: slightly wider bands). */
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
  if (ratio < 0.22 || ratio > 1.05) return false;
  if (Math.min(bw, bh) < span * 0.14) return false;
  let mx = 0,
    my = 0;
  for (const [x, y] of pts) {
    mx += x;
    my += y;
  }
  mx /= pts.length;
  my /= pts.length;
  if (Math.hypot(mx - tipPt.x, my - tipPt.y) > sidePx * 0.65) return false;
  const padX = 0.2 * bw;
  const padY = 0.2 * bh;
  if (tipPt.x < x0 - padX || tipPt.x > x1 + padX) return false;
  if (tipPt.y < y0 - padY || tipPt.y > y1 + padY) return false;

  const pierce = pts[Math.min(PIERCING_INDEX, pts.length - 1)];
  if (pierce) {
    const tipD = Math.hypot(pierce[0] - tipPt.x, pierce[1] - tipPt.y);
    if (tipD < 0.04 * sidePx || tipD > 0.85 * sidePx) return false;
    if (pierce[1] < tipPt.y - 0.14 * sidePx) return false;
  }
  return true;
}

/** Port of train/crop.py pierce_quality */
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

/**
 * Primary tip-centered crop — train/crop.py build_crop_meta:
 * medial 0.10*pinna + down 0.17*pinna (lobe stays in frame).
 */
export function tipCropCenter(tipPt, pinna, yolo, side, vw) {
  const [mx] = medial(yolo, tipPt, side, vw);
  return {
    ncx: tipPt.x + mx * (0.1 * pinna),
    ncy: tipPt.y + 0.17 * pinna,
    mx,
  };
}

/** Rescue when tip escapes square — desktop uses 0.06 down. */
export function rescueCropCenter(tipPt, pinna, mx) {
  return {
    cx: tipPt.x + mx * (0.1 * pinna),
    cy: tipPt.y + 0.06 * pinna,
  };
}

export function blendRawRel(oldRel, newRel) {
  if (!oldRel || oldRel.length !== newRel.length) return newRel;
  return newRel.map((p, i) => [
    0.2 * oldRel[i][0] + 0.8 * p[0],
    0.2 * oldRel[i][1] + 0.8 * p[1],
  ]);
}
