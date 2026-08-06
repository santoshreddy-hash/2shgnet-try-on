/**
 * Yield helpers for INP — never block the next paint after a user gesture.
 * Does not change landmark / ONNX results.
 */

/** Wait for the next animation frame (paint opportunity). */
export function yieldToPaint() {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 0);
    }
  });
}

/**
 * Yield so the browser can handle input + paint (INP-critical).
 * Prefers scheduler.yield when available.
 */
export async function yieldForInput() {
  const sched = globalThis.scheduler;
  if (sched && typeof sched.yield === "function") {
    await sched.yield();
    return;
  }
  // Double-rAF: ensures a frame is painted before heavy work resumes
  await yieldToPaint();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await yieldToPaint();
}

/** Run fn when the main thread is idle (CPU probe, non-critical demote). */
export function runWhenIdle(fn, timeoutMs = 1200) {
  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(
      () => {
        try {
          fn();
        } catch (e) {
          console.warn("[idle]", e);
        }
      },
      { timeout: timeoutMs }
    );
    return;
  }
  setTimeout(() => {
    try {
      fn();
    } catch (e) {
      console.warn("[idle]", e);
    }
  }, Math.min(200, timeoutMs));
}
