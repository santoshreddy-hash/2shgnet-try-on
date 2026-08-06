/**
 * EARRING Try On — upload image → live studio.
 * Stud = piercing #56. No landmark overlay. Drag the dangle to swing.
 */
import { EarTryOnPipeline, PIERCING_INDEX } from "./pipeline.js";
import {
  buildDefaultSprite,
  spriteFromImage,
  EarringPhysics,
  drawEarring,
  jointScreen,
  angleFromJoint,
  hitDangle,
  clientToCanvas,
} from "./earring.js";

const screens = {
  splash: document.getElementById("splash"),
  product: document.getElementById("product"),
  studio: document.getElementById("studio"),
};

function showScreen(name) {
  for (const [k, el] of Object.entries(screens)) {
    el.classList.toggle("active", k === name);
  }
  location.hash = name;
}

function screenFromHash() {
  const h = (location.hash || "#splash").replace("#", "");
  if (h === "product" || h === "studio" || h === "splash") showScreen(h);
  else showScreen("splash");
}

let sprite = buildDefaultSprite();
const physics = new EarringPhysics();
let lastPierce = null;
let lastScale = 1;
let lastTheta = 0;
let lastFrameTs = 0;
let previewRaf = 0;

const previewCanvas = document.getElementById("previewSprite");
const previewCtx = previewCanvas.getContext("2d");
const out = document.getElementById("out");
const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const loadBtn = document.getElementById("loadModels");
const startBtn = document.getElementById("startCam");
const stopBtn = document.getElementById("stopCam");
const freedomEl = document.getElementById("freedom");
const freedomVal = document.getElementById("freedomVal");
const uploadHint = document.getElementById("uploadHint");
const dropZone = document.getElementById("dropZone");
const earFile = document.getElementById("earFile");
const studioEarFile = document.getElementById("studioEarFile");

const pipe = new EarTryOnPipeline({
  video,
  canvas: out,
  onStatus: (s) => {
    statusEl.textContent = s;
  },
});

function setSprite(sp) {
  sprite = sp;
  document.getElementById("prodName").textContent = sp.name || "Your earring";
  document.getElementById("studioProdName").textContent = sp.name || "Your earring";
  if (uploadHint) {
    uploadHint.style.display = sp.fromUpload ? "none" : "";
  }
  physics.reset();
}

async function loadEarringFile(file) {
  if (!file) return;
  statusEl.textContent = "Processing earring image…";
  try {
    const sp = await spriteFromImage(file, {
      name: file.name.replace(/\.[^.]+$/, "") || "Uploaded earring",
    });
    setSprite(sp);
    statusEl.textContent = `Loaded “${sp.name}”. Stud = top of image.`;
  } catch (e) {
    statusEl.textContent = `Image failed: ${e?.message || e}`;
  }
}

function paintPreview() {
  if (!screens.product.classList.contains("active")) {
    previewRaf = requestAnimationFrame(paintPreview);
    return;
  }
  previewCanvas.width = 220;
  previewCanvas.height = 280;
  previewCtx.clearRect(0, 0, 220, 280);
  if (!sprite) {
    previewRaf = requestAnimationFrame(paintPreview);
    return;
  }
  const fit = Math.min(180 / sprite.width, 220 / sprite.height);
  const th = Math.sin(performance.now() / 850) * 0.22;
  drawEarring(previewCtx, sprite, 110, 28, th, fit);
  previewRaf = requestAnimationFrame(paintPreview);
}

function updateButtons() {
  startBtn.disabled = !(pipe.ready && !pipe.live);
  stopBtn.disabled = !pipe.live;
  loadBtn.disabled = pipe.live;
}

pipe.onOverlay = (overlay, meta) => {
  const ctx = pipe.ctx;
  const now = performance.now();
  const dt = lastFrameTs ? Math.min(0.05, (now - lastFrameTs) / 1000) : 1 / 30;
  lastFrameTs = now;

  // Stud locks to piercing #56 (tip-rigid). Dangle physics only.
  if (sprite && (overlay?.pierce || overlay?.landmarks?.[PIERCING_INDEX])) {
    const pierce = overlay.pierce
      ? [overlay.pierce.x, overlay.pierce.y]
      : overlay.landmarks[PIERCING_INDEX];
    if (pierce) {
      const [px, py] = pierce;
      const theta = physics.step(px, py, dt);
      lastPierce = { x: px, y: py };
      lastTheta = theta;
      let scale = 1;
      if (overlay.box) {
        const side = Math.max(
          40,
          (overlay.box[2] - overlay.box[0] + overlay.box[3] - overlay.box[1]) / 2
        );
        const targetH = side * 0.55;
        scale = Math.max(0.35, Math.min(1.8, targetH / Math.max(1, sprite.height)));
      }
      lastScale = scale;
      drawEarring(ctx, sprite, px, py, theta, scale);
    }
  }

  const fps = meta?.fps ? meta.fps.toFixed(0) : "-";
  const pipeMs = meta?.pipeMs ? meta.pipeMs.toFixed(0) : "-";
  const drag = physics.dragging ? " · dragging" : "";
  const lock = lastPierce ? "locked #56" : "find side profile";
  const prof = meta?.profile ? ` · ${meta.profile}` : "";
  const cadence =
    meta?.yoloEvery && meta?.shgEvery
      ? ` · Y/${meta.yoloEvery} S/${meta.shgEvery}`
      : "";
  statusEl.textContent =
    `${pipe.live ? "LIVE" : "Idle"} ${fps}/${meta?.targetFps || "-"} fps${prof}${cadence} · pipe ${pipeMs} ms${drag}\n` +
    `${sprite?.name || "Earring"} · ${lock}\n` +
    `Swing ${physics.mode} · freedom ${physics.freedom.toFixed(2)}`;
};

/* —— Pointer drag on dangling part —— */
function pointerPos(e) {
  return clientToCanvas(out, e.clientX, e.clientY);
}

out.addEventListener("pointerdown", (e) => {
  if (!pipe.live || !lastPierce || !sprite) return;
  const p = pointerPos(e);
  if (
    hitDangle(sprite, lastPierce.x, lastPierce.y, lastTheta, lastScale, p.x, p.y)
  ) {
    const j = jointScreen(sprite, lastPierce.x, lastPierce.y, lastScale);
    const th = angleFromJoint(j.x, j.y, p.x, p.y);
    physics.beginDrag(th);
    out.setPointerCapture(e.pointerId);
    e.preventDefault();
  }
});

out.addEventListener("pointermove", (e) => {
  if (!physics.dragging || !lastPierce || !sprite) return;
  const p = pointerPos(e);
  const j = jointScreen(sprite, lastPierce.x, lastPierce.y, lastScale);
  physics.updateDrag(angleFromJoint(j.x, j.y, p.x, p.y));
  e.preventDefault();
});

function endPointer(e) {
  if (physics.dragging) {
    physics.endDrag();
    try {
      out.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* already released */
    }
  }
}
out.addEventListener("pointerup", endPointer);
out.addEventListener("pointercancel", endPointer);
out.style.touchAction = "none";
out.style.cursor = "grab";

/* —— Nav / upload —— */
document.getElementById("enterStudio").addEventListener("click", () => showScreen("product"));
document.getElementById("backSplash").addEventListener("click", (e) => {
  e.preventDefault();
  showScreen("splash");
});
document.getElementById("toSplashFromProduct").addEventListener("click", () => showScreen("splash"));
document.getElementById("tryOnBtn").addEventListener("click", () => {
  showScreen("studio");
  physics.reset();
});
document.getElementById("backProduct").addEventListener("click", () => {
  if (pipe.live) pipe.stopCamera();
  updateButtons();
  showScreen("product");
});

document.getElementById("pickFileBtn").addEventListener("click", () => earFile.click());
dropZone.addEventListener("click", (e) => {
  if (e.target === earFile) return;
  earFile.click();
});
earFile.addEventListener("change", () => {
  const f = earFile.files?.[0];
  if (f) loadEarringFile(f);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag");
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) loadEarringFile(f);
});

document.getElementById("studioUploadBtn").addEventListener("click", () => studioEarFile.click());
studioEarFile.addEventListener("change", () => {
  const f = studioEarFile.files?.[0];
  if (f) loadEarringFile(f);
});

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  try {
    await pipe.loadModels();
    updateButtons();
  } catch (e) {
    statusEl.textContent = `Load failed: ${e?.message || e}`;
    loadBtn.disabled = false;
  }
});

startBtn.addEventListener("click", async () => {
  physics.reset();
  lastPierce = null;
  try {
    await pipe.startCamera();
  } catch (_) {
    /* status set */
  }
  updateButtons();
});

stopBtn.addEventListener("click", () => {
  pipe.stopCamera();
  updateButtons();
});

document.getElementById("modeClassic").addEventListener("click", () => {
  physics.setMode("classic");
  document.getElementById("modeClassic").classList.add("active");
  document.getElementById("modeSubtle").classList.remove("active");
});
document.getElementById("modeSubtle").addEventListener("click", () => {
  physics.setMode("subtle");
  document.getElementById("modeSubtle").classList.add("active");
  document.getElementById("modeClassic").classList.remove("active");
});

freedomEl.addEventListener("input", () => {
  const v = Number(freedomEl.value);
  physics.setFreedom(v);
  freedomVal.textContent = v.toFixed(2);
});

window.addEventListener("hashchange", screenFromHash);
window.addEventListener("beforeunload", () => pipe.stopCamera());

setSprite(sprite);
physics.setFreedom(Number(freedomEl.value));
screenFromHash();
updateButtons();
paintPreview();
statusEl.textContent =
  "Upload an earring image, Load models, then Start camera.\nDrag the dangling part to swing it.";
