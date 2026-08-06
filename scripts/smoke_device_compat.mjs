/**
 * Smoke tests for browser device compatibility classification.
 * Run: node scripts/smoke_device_compat.mjs
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const perfPath = join(root, "web", "performance.js");

// Import classifier without a browser DOM
const { classifyBrowserCapability } = await import(
  pathToFileURL(perfPath).href
);

const raw = JSON.parse(
  readFileSync(join(root, "performance_profiles.json"), "utf8")
);
const cfg = raw.browser_auto_detect || {};

/** @type {{name: string, signals: object, expect: string}[]} */
const cases = [
  {
    name: "desktop_high_8c_8gb",
    signals: { cores: 8, deviceMemoryGb: 8, mobile: false, webgpu: false },
    expect: "high",
  },
  {
    name: "desktop_high_6c_4gb",
    signals: { cores: 6, deviceMemoryGb: 4, mobile: false, webgpu: false },
    expect: "high",
  },
  {
    name: "desktop_medium_4c_4gb",
    signals: { cores: 4, deviceMemoryGb: 4, mobile: false, webgpu: false },
    expect: "medium",
  },
  {
    name: "desktop_low_2c_2gb",
    signals: { cores: 2, deviceMemoryGb: 2, mobile: false, webgpu: false },
    expect: "low",
  },
  {
    name: "mobile_flagship_capped_medium",
    signals: { cores: 8, deviceMemoryGb: 8, mobile: true, webgpu: true },
    expect: "medium",
  },
  {
    name: "mobile_mid_medium",
    signals: { cores: 4, deviceMemoryGb: 4, mobile: true, webgpu: false },
    expect: "medium",
  },
  {
    name: "mobile_weak_low",
    signals: { cores: 2, deviceMemoryGb: 2, mobile: true, webgpu: false },
    expect: "low",
  },
  {
    name: "webgpu_does_not_force_high_wasm",
    signals: { cores: 4, deviceMemoryGb: 4, mobile: false, webgpu: true },
    expect: "medium",
  },
  {
    name: "unknown_mem_strong_cpu_high",
    signals: { cores: 8, deviceMemoryGb: 0, mobile: false, webgpu: false },
    expect: "high",
  },
];

let failed = 0;
for (const c of cases) {
  const r = classifyBrowserCapability(c.signals, cfg);
  const ok = r.recommended === c.expect;
  console.log(
    `${ok ? "PASS" : "FAIL"} ${c.name}: got=${r.recommended} expect=${c.expect} ` +
      `score=${r.score.toFixed(1)} · ${r.detail}`
  );
  if (!ok) failed += 1;
}

// Forced-mode sanity: classifier is independent; resolve applies force in browser.
const forceCheck = ["high", "medium", "low"];
console.log("force modes available:", forceCheck.join(", "));

// One Euro per tier present
const oe = JSON.parse(readFileSync(join(root, "one_euro_settings.json"), "utf8"));
for (const t of forceCheck) {
  const p = oe.profiles?.[t];
  const ok =
    p &&
    typeof p.min_cutoff === "number" &&
    typeof p.beta === "number" &&
    typeof p.max_step_px === "number";
  console.log(`${ok ? "PASS" : "FAIL"} one_euro.${t}`);
  if (!ok) failed += 1;
}

if (failed) {
  console.error(`\n${failed} failure(s)`);
  process.exit(1);
}
console.log("\nALL PASS");
