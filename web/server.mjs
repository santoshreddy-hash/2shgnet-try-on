#!/usr/bin/env node
/**
 * Static server for browser ONNX / WASM demos.
 *   http://127.0.0.1:8765/           — landmark demo
 *   http://127.0.0.1:8765/earring/   — earring virtual try-on
 *
 * Serves ../models/ + web/ + vendor (onnxruntime-web).
 * COOP/COEP headers enable SharedArrayBuffer for threaded WASM.
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
  ".svg": "image/svg+xml",
  ".data": "application/octet-stream",
};

function resolveUrl(urlPath) {
  const clean = decodeURIComponent((urlPath || "/").split("?")[0]);

  if (clean === "/earring" || clean === "/earring/") {
    return path.join(__dirname, "earring", "index.html");
  }
  if (clean === "/" || clean === "") {
    return path.join(__dirname, "index.html");
  }
  if (clean === "/one_euro_settings.json") {
    return path.join(ROOT, "one_euro_settings.json");
  }
  if (clean === "/performance_profiles.json") {
    return path.join(ROOT, "performance_profiles.json");
  }
  if (clean.startsWith("/models/")) {
    return path.join(ROOT, clean.slice(1));
  }
  if (clean.startsWith("/vendor/")) {
    return path.join(__dirname, "node_modules", clean.slice("/vendor/".length));
  }
  // Directory → index.html
  const underWeb = path.join(__dirname, clean.replace(/^\//, ""));
  try {
    if (fs.existsSync(underWeb) && fs.statSync(underWeb).isDirectory()) {
      return path.join(underWeb, "index.html");
    }
  } catch {
    /* fall through */
  }
  return underWeb;
}

const server = http.createServer((req, res) => {
  // Pretty redirect
  if ((req.url || "").split("?")[0] === "/earring") {
    res.writeHead(302, { Location: "/earring/" });
    res.end();
    return;
  }

  const filePath = resolveUrl(req.url || "/");
  const allowed =
    filePath.startsWith(ROOT + path.sep) ||
    filePath.startsWith(__dirname + path.sep) ||
    filePath === ROOT ||
    filePath === __dirname;
  if (!allowed) {
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
  console.log(`SHGNet-56 web (WASM)`);
  console.log(`  landmark demo: http://127.0.0.1:${PORT}/`);
  console.log(`  earring try-on: http://127.0.0.1:${PORT}/earring/`);
  console.log(`  SHGNet ONNX: ${mb(shg)} MB`);
  console.log(`  YOLO ONNX:   ${mb(yolo)} MB`);
});
