/**
 * Match Python `preprocess_ear_bgr` / training exactly:
 *   canvas RGB → BGR → YCrCb → equalize Y → BGR → /255 → CHW (B,G,R)
 */
export function equalizeHist(yPlane) {
  const n = yPlane.length;
  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) hist[yPlane[i]]++;
  const cdf = new Uint32Array(256);
  let sum = 0;
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    sum += hist[i];
    cdf[i] = sum;
    if (!cdfMin && sum) cdfMin = sum;
  }
  const denom = Math.max(1, n - cdfMin);
  const map = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    map[i] = Math.round(((cdf[i] - cdfMin) / denom) * 255);
  }
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = map[yPlane[i]];
  return out;
}

/**
 * @param {ImageData} imageData 256×256 RGBA from canvas (RGB order)
 * @returns {Float32Array} CHW length 3*256*256 in BGR channel order, [0,1]
 */
export function canvasRgbaToBgrChw(imageData) {
  const { data } = imageData;
  const n = 256 * 256;
  const R = new Float32Array(n);
  const G = new Float32Array(n);
  const B = new Float32Array(n);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    R[p] = data[i];
    G[p] = data[i + 1];
    B[p] = data[i + 2];
  }

  // OpenCV COLOR_BGR2YCrCb (same as RGB coeffs; B/R only for Cr/Cb)
  const Y = new Uint8Array(n);
  const Cr = new Uint8Array(n);
  const Cb = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    const y = 0.299 * R[p] + 0.587 * G[p] + 0.114 * B[p];
    Y[p] = y < 0 ? 0 : y > 255 ? 255 : Math.round(y);
    let cr = (R[p] - y) * 0.713 + 128;
    let cb = (B[p] - y) * 0.564 + 128;
    Cr[p] = cr < 0 ? 0 : cr > 255 ? 255 : Math.round(cr);
    Cb[p] = cb < 0 ? 0 : cb > 255 ? 255 : Math.round(cb);
  }

  const Yeq = equalizeHist(Y);

  // OpenCV COLOR_YCrCb2BGR
  const chw = new Float32Array(3 * n);
  const inv = 1 / 255;
  for (let p = 0; p < n; p++) {
    const y = Yeq[p];
    const cr = Cr[p] - 128;
    const cb = Cb[p] - 128;
    let r = y + 1.403 * cr;
    let g = y - 0.344 * cb - 0.714 * cr;
    let b = y + 1.773 * cb;
    r = r < 0 ? 0 : r > 255 ? 255 : r;
    g = g < 0 ? 0 : g > 255 ? 255 : g;
    b = b < 0 ? 0 : b > 255 ? 255 : b;
    // CHW BGR (matches OpenCV / training)
    chw[p] = b * inv;
    chw[n + p] = g * inv;
    chw[2 * n + p] = r * inv;
  }
  return chw;
}

/**
 * Soft-argmax around peak for sub-pixel landmarks (64×64 → 256 space).
 * @param {import('onnxruntime-web').Tensor} hm
 * @returns {number[][]} 55×[x,y] in 256 crop space
 */
/**
 * Soft-argmax around peak — matches Python heatmaps_to_points_soft (radius=2, exp).
 * Also returns mean peak score when returnScore=true.
 */
export function heatmapsToPointsSoft(hm, inputSize = 256, radius = 2) {
  const dims = hm.dims;
  const data = hm.data;
  const n = dims.length === 4 ? dims[1] : dims[0];
  const h = dims.length === 4 ? dims[2] : dims[1];
  const w = dims.length === 4 ? dims[3] : dims[2];
  const scaleX = inputSize / w;
  const scaleY = inputSize / h;
  const plane = h * w;
  const pts = new Array(n);
  let peakSum = 0;

  for (let i = 0; i < n; i++) {
    const off = i * plane;
    let best = -Infinity;
    let bestIdx = 0;
    for (let k = 0; k < plane; k++) {
      const v = data[off + k];
      if (v > best) {
        best = v;
        bestIdx = k;
      }
    }
    peakSum += best;
    const yy = (bestIdx / w) | 0;
    const xx = bestIdx % w;
    const y0 = Math.max(0, yy - radius);
    const y1 = Math.min(h - 1, yy + radius);
    const x0 = Math.max(0, xx - radius);
    const x1 = Math.min(w - 1, xx + radius);
    let sum = 0;
    let sx = 0;
    let sy = 0;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const wt = Math.exp(data[off + y * w + x] - best);
        sum += wt;
        sx += wt * x;
        sy += wt * y;
      }
    }
    if (sum < 1e-12) {
      pts[i] = [xx * scaleX, yy * scaleY];
    } else {
      pts[i] = [(sx / sum) * scaleX, (sy / sum) * scaleY];
    }
  }
  pts.score = peakSum / Math.max(1, n);
  return pts;
}
