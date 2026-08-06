#!/usr/bin/env node
/**
 * One-command mobile preview:
 *   npm run mobile
 *
 * Starts the local server + an HTTPS tunnel so phones can use the camera.
 * Prefers: cloudflared → ngrok → npx localtunnel
 */
import { spawn, spawnSync } from "node:child_process";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8765);
const kids = [];

function which(cmd) {
  const r = spawnSync(process.platform === "win32" ? "where" : "which", [cmd], {
    encoding: "utf8",
  });
  return r.status === 0 ? (r.stdout || "").trim().split("\n")[0] : null;
}

function waitForPort(port, ms = 15000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(`http://127.0.0.1:${port}/`, (res) => {
        res.resume();
        resolve();
      });
      req.on("error", () => {
        if (Date.now() - start > ms) reject(new Error(`Server not up on :${port}`));
        else setTimeout(tryOnce, 200);
      });
    };
    tryOnce();
  });
}

function track(child, label) {
  kids.push(child);
  child.on("exit", (code, signal) => {
    if (signal) console.log(`[mobile] ${label} stopped (${signal})`);
    else if (code && code !== 0) console.log(`[mobile] ${label} exited ${code}`);
  });
  return child;
}

function shutdown() {
  for (const c of kids) {
    try {
      c.kill("SIGTERM");
    } catch {
      /* ignore */
    }
  }
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

function startServer() {
  const child = spawn(process.execPath, [path.join(__dirname, "server.mjs")], {
    cwd: __dirname,
    env: { ...process.env, PORT: String(PORT), HOST: "0.0.0.0" },
    stdio: ["ignore", "inherit", "inherit"],
  });
  return track(child, "server");
}

function printUrl(url) {
  console.log("");
  console.log("────────────────────────────────────────");
  console.log("  Phone URL (HTTPS — camera works):");
  console.log(`  ${url}`);
  console.log("────────────────────────────────────────");
  console.log("  Keep this terminal open. Ctrl+C to stop.");
  console.log("  On phone: Load models → allow camera → Start live cam");
  console.log("");
}

function watchForUrl(child, patterns, label) {
  return new Promise((resolve, reject) => {
    let buf = "";
    let done = false;
    const finish = (fn, arg) => {
      if (done) return;
      done = true;
      fn(arg);
    };
    const onChunk = (chunk) => {
      const text = chunk.toString();
      buf += text;
      process.stdout.write(text);
      for (const re of patterns) {
        const m = buf.match(re);
        if (m?.[1]) {
          finish(resolve, m[1].replace(/\/$/, ""));
          return;
        }
      }
    };
    child.stdout?.on("data", onChunk);
    child.stderr?.on("data", onChunk);
    child.on("error", (e) => finish(reject, e));
    child.on("exit", (code) => {
      if (!done && code) finish(reject, new Error(`${label} exited early (${code})`));
    });
    setTimeout(() => {
      finish(reject, new Error(`${label}: no public URL within 45s`));
    }, 45000);
  });
}

async function startTunnel() {
  const cf = which("cloudflared");
  if (cf) {
    console.log("[mobile] tunnel: cloudflared");
    const child = spawn(cf, ["tunnel", "--url", `http://127.0.0.1:${PORT}`], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    track(child, "cloudflared");
    return watchForUrl(
      child,
      [/(https:\/\/[a-z0-9-]+\.trycloudflare\.com)/i],
      "cloudflared"
    );
  }

  const ng = which("ngrok");
  if (ng) {
    console.log("[mobile] tunnel: ngrok");
    const child = spawn(ng, ["http", String(PORT), "--log=stdout", "--log-format=json"], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    track(child, "ngrok");
    return watchForUrl(
      child,
      [
        /"url":"(https:\/\/[^"]+)"/,
        /url=(https:\/\/[a-z0-9.-]+\.ngrok[^\s"]*)/i,
        /(https:\/\/[a-z0-9-]+\.ngrok-free\.(?:app|dev))/i,
        /(https:\/\/[a-z0-9-]+\.ngrok\.io)/i,
      ],
      "ngrok"
    );
  }

  console.log("[mobile] tunnel: localtunnel (npx — no install needed)");
  const child = spawn("npx", ["--yes", "localtunnel", "--port", String(PORT)], {
    stdio: ["ignore", "pipe", "pipe"],
    shell: process.platform === "win32",
  });
  track(child, "localtunnel");
  return watchForUrl(
    child,
    [/(https:\/\/[a-z0-9-]+\.loca\.lt)/i, /your url is:\s*(https:\/\/\S+)/i],
    "localtunnel"
  );
}

async function main() {
  console.log(`[mobile] checking :${PORT} …`);
  let serverAlreadyUp = false;
  try {
    await waitForPort(PORT, 800);
    serverAlreadyUp = true;
    console.log(`[mobile] reusing server already on :${PORT}`);
  } catch {
    console.log(`[mobile] starting server on :${PORT} …`);
    startServer();
    await waitForPort(PORT);
  }
  console.log(`[mobile] local: http://127.0.0.1:${PORT}/`);

  try {
    const url = await startTunnel();
    printUrl(url);
    if (serverAlreadyUp) {
      console.log("[mobile] note: Ctrl+C stops the tunnel only (server was already running).");
    }
  } catch (e) {
    console.error(`[mobile] tunnel failed: ${e.message}`);
    console.error("Install one of: brew install cloudflare/cloudflare/cloudflared");
    console.error("            or: brew install ngrok/ngrok/ngrok");
    console.error(`Or run: npx --yes localtunnel --port ${PORT}`);
    shutdown();
  }
}

main();
