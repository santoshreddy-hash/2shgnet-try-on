#!/usr/bin/env node
/**
 * Static server for browser ONNX / WASM demo.
 * Serves ../models/ + this folder. Open http://127.0.0.1:8765
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PORT = Number(process.env.PORT || 8765);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".data": "application/octet-stream",
};

function resolveUrl(urlPath) {
  const clean = decodeURIComponent(urlPath.split("?")[0]);
  if (clean === "/" || clean === "") {
    return path.join(__dirname, "index.html");
  }
  if (clean.startsWith("/models/")) {
    return path.join(ROOT, clean.slice(1));
  }
  if (clean.startsWith("/vendor/")) {
    return path.join(__dirname, "node_modules", clean.slice("/vendor/".length));
  }
  return path.join(__dirname, clean.replace(/^\//, ""));
}

const server = http.createServer((req, res) => {
  const filePath = resolveUrl(req.url || "/");
  if (!filePath.startsWith(ROOT) && !filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end(`Not found: ${req.url}`);
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": MIME[ext] || "application/octet-stream",
      "Cache-Control":
        ext === ".onnx" || ext === ".wasm" ? "public, max-age=3600" : "no-cache",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
    });
    res.end(data);
  });
});

const HOST = process.env.HOST || "0.0.0.0";

server.listen(PORT, HOST, () => {
  const shg = path.join(ROOT, "models/shgnet/SHGNet-56.onnx");
  const yolo = path.join(ROOT, "models/yolo/yolo26n-pose.onnx");
  const mb = (p) => {
    try {
      return (fs.statSync(p).size / (1024 * 1024)).toFixed(1);
    } catch {
      return "?";
    }
  };
  console.log(`SHGNet-56 ONNX web (WASM, no Gradio)`);
  console.log(`  local:   http://127.0.0.1:${PORT}`);
  console.log(`  network: http://<this-machine-ip>:${PORT}`);
  console.log(`  SHGNet ONNX: ${mb(shg)} MB`);
  console.log(`  YOLO ONNX:   ${mb(yolo)} MB`);
  console.log(`  + onnxruntime-web WASM (~5–15 MB, cached)`);
});
