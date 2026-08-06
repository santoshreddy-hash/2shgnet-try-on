/** Shared ORT setup for model Web Workers. */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";

export function configureOrt(wasmPath) {
  ort.env.wasm.wasmPaths = wasmPath;
  const canSAB =
    typeof SharedArrayBuffer !== "undefined" &&
    (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
  // Cap threads per worker so two workers don't oversubscribe
  const threads = canSAB
    ? Math.min(2, self.navigator?.hardwareConcurrency || 2)
    : 1;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false;
  return threads;
}

export { ort };
