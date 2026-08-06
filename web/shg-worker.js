/**
 * SHGNet-56 Web Worker — owns its own ORT WASM + SHGNet session.
 * Isolated from YOLO so both models never share one worker thread.
 */
import * as ort from "/vendor/onnxruntime-web/dist/ort.wasm.min.mjs";

let shgSession = null;

async function init(msg) {
  const { modelUrl, wasmPaths } = msg;
  ort.env.wasm.wasmPaths = wasmPaths || "/vendor/onnxruntime-web/dist/";
  // Multi-thread when cross-origin isolated (COOP/COEP on Vercel)
  const cores =
    typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 2 : 2;
  ort.env.wasm.numThreads =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated
      ? Math.min(4, Math.max(1, cores >> 1))
      : 1;
  ort.env.wasm.proxy = false;

  shgSession = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return {
    input: shgSession.inputNames[0],
    output: shgSession.outputNames[0],
  };
}

async function runShg(chwBuf, dims) {
  const data = new Float32Array(chwBuf);
  const tensor = new ort.Tensor("float32", data, dims);
  const t0 = performance.now();
  const out = await shgSession.run({ [shgSession.inputNames[0]]: tensor });
  const ms = performance.now() - t0;
  const tensorOut = out[shgSession.outputNames[0]];
  const copy = tensorOut.data.slice
    ? tensorOut.data.slice()
    : Float32Array.from(tensorOut.data);
  return {
    data: copy.buffer,
    dims: tensorOut.dims || [],
    ms,
  };
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  const { id, type } = msg;
  try {
    if (type === "init") {
      const meta = await init(msg);
      self.postMessage({ id, ok: true, type: "init", meta });
      return;
    }
    if (type === "shg") {
      if (!shgSession) throw new Error("SHG worker not initialized");
      const result = await runShg(msg.chw, msg.dims);
      self.postMessage(
        {
          id,
          ok: true,
          type: "shg",
          dims: result.dims,
          ms: result.ms,
          data: result.data,
        },
        [result.data]
      );
      return;
    }
    throw new Error(`Unknown SHG worker message: ${type}`);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      type,
      error: String(err?.message || err),
    });
  }
};
