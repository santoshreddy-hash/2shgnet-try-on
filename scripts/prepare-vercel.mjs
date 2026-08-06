#!/usr/bin/env node
/**
 * Build a static Vercel bundle under .vercel-out/
 * (ONNX weights are gitignored — must be copied from local disk.)
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, ".vercel-out");

function rmrf(p) {
  fs.rmSync(p, { recursive: true, force: true });
}

function mkdirp(p) {
  fs.mkdirSync(p, { recursive: true });
}

function copyFile(src, dest) {
  mkdirp(path.dirname(dest));
  fs.copyFileSync(src, dest);
}

function copyDir(src, dest, filter) {
  if (!fs.existsSync(src)) return;
  mkdirp(dest);
  for (const name of fs.readdirSync(src)) {
    const s = path.join(src, name);
    const d = path.join(dest, name);
    const st = fs.statSync(s);
    if (st.isDirectory()) {
      if (name === "node_modules") continue;
      copyDir(s, d, filter);
    } else if (!filter || filter(name, s)) {
      copyFile(s, d);
    }
  }
}

rmrf(OUT);
mkdirp(OUT);

// Web app (JS/HTML — skip server/mobile helpers and local Vercel link state)
copyDir(path.join(ROOT, "web"), OUT, (name) => {
  if (name === "package.json" || name === "package-lock.json") return false;
  if (name === "server.mjs" || name === "mobile.mjs") return false;
  if (name === ".vercel") return false;
  return true;
});
copyDir(path.join(ROOT, "web", "earring"), path.join(OUT, "earring"), (name) => name !== ".vercel");

// ONNX runtime vendor — only files the demo actually loads
const ortSrc = path.join(ROOT, "web", "node_modules", "onnxruntime-web");
const ortDest = path.join(OUT, "vendor", "onnxruntime-web");
if (!fs.existsSync(ortSrc)) {
  console.error("Missing web/node_modules/onnxruntime-web — run: cd web && npm install");
  process.exit(1);
}
const distSrc = path.join(ortSrc, "dist");
const distDest = path.join(ortDest, "dist");
mkdirp(distDest);
const keep = new Set([
  "ort.wasm.min.mjs",
  "ort.wasm.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
]);
for (const name of fs.readdirSync(distSrc)) {
  if (keep.has(name) || (name.startsWith("ort-wasm") && name.endsWith(".wasm"))) {
    // Prefer only the default SIMD threaded + wasm min entry
    if (
      name === "ort.wasm.min.mjs" ||
      name === "ort-wasm-simd-threaded.mjs" ||
      name === "ort-wasm-simd-threaded.wasm"
    ) {
      copyFile(path.join(distSrc, name), path.join(distDest, name));
    }
  }
}
copyFile(
  path.join(ortSrc, "package.json"),
  path.join(ortDest, "package.json")
);

// Models (required for inference)
const yolo = path.join(ROOT, "models", "yolo", "yolo26n-pose.onnx");
const shg = path.join(ROOT, "models", "shgnet", "SHGNet-56.onnx");
for (const f of [yolo, shg]) {
  if (!fs.existsSync(f)) {
    console.error(`Missing model: ${f}`);
    process.exit(1);
  }
}
copyFile(yolo, path.join(OUT, "models", "yolo", "yolo26n-pose.onnx"));
copyFile(shg, path.join(OUT, "models", "shgnet", "SHGNet-56.onnx"));

// Shared JSON configs
copyFile(
  path.join(ROOT, "performance_profiles.json"),
  path.join(OUT, "performance_profiles.json")
);
copyFile(
  path.join(ROOT, "one_euro_settings.json"),
  path.join(OUT, "one_euro_settings.json")
);

// Vercel headers — COOP/COEP for SharedArrayBuffer / threaded WASM
const vercelJson = {
  headers: [
    {
      source: "/(.*)",
      headers: [
        { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
        { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
      ],
    },
  ],
  rewrites: [{ source: "/earring", destination: "/earring/" }],
};
fs.writeFileSync(path.join(OUT, "vercel.json"), JSON.stringify(vercelJson, null, 2));

// Size report
const sizeOf = (p) => {
  try {
    return (fs.statSync(p).size / (1024 * 1024)).toFixed(1);
  } catch {
    return "?";
  }
};
console.log(`Prepared ${OUT}`);
console.log(`  YOLO  ${sizeOf(path.join(OUT, "models/yolo/yolo26n-pose.onnx"))} MB`);
console.log(`  SHG   ${sizeOf(path.join(OUT, "models/shgnet/SHGNet-56.onnx"))} MB`);
console.log(`  Deploy with: npx vercel deploy .vercel-out --yes --prod`);
