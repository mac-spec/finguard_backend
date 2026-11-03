/*
 * FinGuard browser effects
 *
 * Import this module once from the React application entry point:
 *   import "./finguard-effects.js";
 *
 * Adds an ambient gold particle, click sparkles, and helpers for risk alerts.
 * The module is deliberately framework-free so it can be used by any React page.
 */

const REDUCED_MOTION = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;

const style = document.createElement("style");
style.dataset.finguardEffects = "true";
style.textContent = `
  .fg-effects-root {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    isolation: isolate;
  }
  .fg-gold-orb {
    position: absolute;
    width: clamp(240px, 34vw, 560px);
    aspect-ratio: 1;
    left: 50%;
    top: 48%;
    border-radius: 50%;
    opacity: .72;
    transform: translate(-50%, -50%);
    background:
      radial-gradient(circle at 36% 34%,
        rgba(255, 248, 202, .98) 0 1.2%,
        rgba(255, 211, 92, .88) 2.4%,
        rgba(238, 169, 33, .48) 14%,
        rgba(186, 120, 13, .20) 33%,
        rgba(154, 100, 16, .07) 51%,
        transparent 70%);
    filter: blur(1.2px) drop-shadow(0 0 26px rgba(255, 192, 55, .33));
    will-change: transform;
    animation: fg-orb-drift 13s ease-in-out infinite alternate,
               fg-orb-pulse 5s ease-in-out infinite;
  }
  .fg-gold-orb::before,
  .fg-gold-orb::after {
    content: "";
    position: absolute;
    inset: 12%;
    border-radius: 50%;
    border: 1px solid rgba(255, 211, 112, .16);
    box-shadow: 0 0 35px rgba(244, 181, 54, .12), inset 0 0 35px rgba(244, 181, 54, .08);
  }
  .fg-gold-orb::after {
    inset: 23%;
    border-color: rgba(255, 231, 170, .12);
  }
  .fg-click-spark {
    position: fixed;
    width: 6px;
    height: 6px;
    z-index: 2147483000;
    border-radius: 50%;
    pointer-events: none;
    background: var(--spark-color, #67f7ff);
    box-shadow: 0 0 8px 2px var(--spark-color, #67f7ff),
                0 0 22px 5px color-mix(in srgb, var(--spark-color, #67f7ff) 45%, transparent);
    animation: fg-spark-fly 620ms cubic-bezier(.14,.75,.3,1) forwards;
  }
  .fg-click-spark::before,
  .fg-click-spark::after {
    content: "";
    position: absolute;
    inset: -5px 2px;
    border-radius: 50%;
    border: 1px solid var(--spark-color, #67f7ff);
    opacity: .8;
  }
  .fg-click-spark::after { transform: rotate(60deg); }
  @keyframes fg-orb-drift {
    0% { transform: translate(-58%, -44%) scale(.82); }
    35% { transform: translate(-39%, -57%) scale(1.07); }
    70% { transform: translate(-62%, -62%) scale(.94); }
    100% { transform: translate(-43%, -39%) scale(1.12); }
  }
  @keyframes fg-orb-pulse {
    0%,100% { opacity: .55; filter: blur(1.2px) drop-shadow(0 0 22px rgba(255,192,55,.24)); }
    50% { opacity: .88; filter: blur(.6px) drop-shadow(0 0 42px rgba(255,192,55,.46)); }
  }
  @keyframes fg-spark-fly {
    from { opacity: 1; transform: translate(0,0) scale(1); }
    to { opacity: 0; transform: translate(var(--spark-x), var(--spark-y)) scale(.15); }
  }
  .fg-risk-toast-stack {
    position: fixed;
    z-index: 2147482000;
    top: 20px;
    right: 20px;
    display: grid;
    gap: 10px;
    width: min(390px, calc(100vw - 32px));
    pointer-events: none;
  }
  .fg-risk-toast {
    pointer-events: auto;
    display: grid;
    grid-template-columns: 10px 1fr auto;
    gap: 12px;
    align-items: start;
    padding: 14px 15px;
    color: #f7f4ea;
    background: rgba(12, 14, 22, .94);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 14px;
    box-shadow: 0 14px 38px rgba(0,0,0,.38);
    backdrop-filter: blur(18px);
    animation: fg-toast-in 250ms ease-out both;
    font: 13px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
  }
  .fg-risk-toast[data-level="MEDIUM"] { border-color: rgba(255,190,70,.55); }
  .fg-risk-toast[data-level="HIGH"] { border-color: rgba(255,75,100,.72); box-shadow: 0 14px 42px rgba(255,45,75,.16); }
  .fg-risk-toast__dot { width: 9px; height: 9px; margin-top: 4px; border-radius: 50%; background: #ffc04a; box-shadow: 0 0 12px currentColor; }
  .fg-risk-toast[data-level="HIGH"] .fg-risk-toast__dot { background: #ff526d; }
  .fg-risk-toast__title { margin: 0 0 4px; font-weight: 750; letter-spacing: .04em; }
  .fg-risk-toast__body { margin: 0; color: rgba(247,244,234,.78); overflow-wrap: anywhere; }
  .fg-risk-toast__close { border: 0; color: #aaa; background: transparent; cursor: pointer; font-size: 18px; line-height: 1; }
  @keyframes fg-toast-in { from { opacity: 0; transform: translateY(-8px) scale(.98); } to { opacity: 1; transform: translateY(0) scale(1); } }
  @media (prefers-reduced-motion: reduce) {
    .fg-gold-orb, .fg-click-spark, .fg-risk-toast { animation: none !important; }
  }
`;
document.head.appendChild(style);

function installAmbientOrb() {
  if (document.querySelector(".fg-effects-root")) return;
  const root = document.createElement("div");
  root.className = "fg-effects-root";
  root.setAttribute("aria-hidden", "true");
  const orb = document.createElement("div");
  orb.className = "fg-gold-orb";
  root.appendChild(orb);

  // Place behind the app without changing the app's stacking contexts.
  document.body.prepend(root);
  const appRoot = document.querySelector("#root, #app, [data-app-root]");
  if (appRoot) {
    appRoot.style.position ||= "relative";
    appRoot.style.zIndex ||= "1";
  }
}

function installClickSparkles() {
  if (REDUCED_MOTION || window.__fgSparklesInstalled) return;
  window.__fgSparklesInstalled = true;
  const colors = ["#67f7ff", "#b78cff", "#ffe078", "#ff77ca"];
  document.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const count = 7;
    for (let i = 0; i < count; i += 1) {
      const spark = document.createElement("span");
      const angle = (Math.PI * 2 * i) / count + Math.random() * .35;
      const distance = 18 + Math.random() * 22;
      spark.className = "fg-click-spark";
      spark.style.left = `${event.clientX}px`;
      spark.style.top = `${event.clientY}px`;
      spark.style.setProperty("--spark-x", `${Math.cos(angle) * distance}px`);
      spark.style.setProperty("--spark-y", `${Math.sin(angle) * distance}px`);
      spark.style.setProperty("--spark-color", colors[Math.floor(Math.random() * colors.length)]);
      document.body.appendChild(spark);
      spark.addEventListener("animationend", () => spark.remove(), { once: true });
    }
  }, { passive: true });
}

function getToastStack() {
  let stack = document.querySelector(".fg-risk-toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.className = "fg-risk-toast-stack";
    stack.setAttribute("aria-live", "assertive");
    stack.setAttribute("aria-relevant", "additions");
    document.body.appendChild(stack);
  }
  return stack;
}

let audioContext;
function playRiskTone(level) {
  // Browsers require a user gesture before Web Audio can play. Resume only
  // after the user has interacted with the page; no audio files or tracking.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;
  try {
    audioContext ||= new AudioCtx();
    if (audioContext.state === "suspended") audioContext.resume();
    const beepCount = level === "HIGH" ? 3 : 1;
    const gap = level === "HIGH" ? 190 : 0;
    for (let i = 0; i < beepCount; i += 1) {
      const startAt = audioContext.currentTime + i * gap / 1000;
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = level === "HIGH" ? 920 : 660;
      gain.gain.setValueAtTime(0.0001, startAt);
      gain.gain.exponentialRampToValueAtTime(0.12, startAt + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, startAt + 0.13);
      oscillator.connect(gain);
      gain.connect(audioContext.destination);
      oscillator.start(startAt);
      oscillator.stop(startAt + 0.145);
    }
  } catch (error) {
    console.warn("FinGuard notification sound could not play:", error);
  }
}

const notifiedTransactionIds = new Set();
export function notifyFraudRisk(alert, options = {}) {
  const level = String(alert?.fraud_risk_level || alert?.risk_level || options.level || "MEDIUM").toUpperCase();
  if (level !== "MEDIUM" && level !== "HIGH") return;
  const transactionId = String(alert?.id || alert?.txn_id || alert?.transaction_id || "");
  if (transactionId && notifiedTransactionIds.has(transactionId)) return;
  if (transactionId) {
    notifiedTransactionIds.add(transactionId);
    if (notifiedTransactionIds.size > 1000) {
      const first = notifiedTransactionIds.values().next().value;
      notifiedTransactionIds.delete(first);
    }
  }

  const stack = getToastStack();
  const toast = document.createElement("section");
  toast.className = "fg-risk-toast";
  toast.dataset.level = level;
  const dot = document.createElement("span");
  dot.className = "fg-risk-toast__dot";
  const content = document.createElement("div");
  const title = document.createElement("p");
  title.className = "fg-risk-toast__title";
  title.textContent = level === "HIGH" ? "HIGH RISK TRANSACTION" : "TRANSACTION NEEDS REVIEW";
  const body = document.createElement("p");
  body.className = "fg-risk-toast__body";
  const amount = Number(alert?.amount);
  const amountText = Number.isFinite(amount) ? ` · Amount: ${new Intl.NumberFormat(undefined, { style: "currency", currency: options.currency || "INR", maximumFractionDigits: 2 }).format(amount)}` : "";
  const reasonList = Array.isArray(alert?.reasons) ? alert.reasons.slice(0, 2).join(" ") : "";
  body.textContent = `${alert?.description || alert?.receiver || alert?.id || "A transaction"}${amountText}${reasonList ? ` — ${reasonList}` : ""}`;
  content.append(title, body);
  const close = document.createElement("button");
  close.className = "fg-risk-toast__close";
  close.type = "button";
  close.setAttribute("aria-label", "Dismiss notification");
  close.textContent = "×";
  close.addEventListener("click", () => toast.remove(), { once: true });
  toast.append(dot, content, close);
  stack.appendChild(toast);
  while (stack.children.length > 4) stack.firstElementChild.remove();
  window.setTimeout(() => toast.remove(), level === "HIGH" ? 12000 : 8000);
  playRiskTone(level);
}

/**
 * Feed the existing /api/v1/transactions/scan response into the notification UI.
 * Call once for each new scan response. LOW risk is intentionally silent.
 */
export function handleTransactionScanResponse(response) {
  if (!response || !Array.isArray(response.alerts)) return;
  response.alerts.forEach((alert) => {
    const level = String(alert?.fraud_risk_level || "MEDIUM").toUpperCase();
    if (level === "HIGH" || level === "MEDIUM") notifyFraudRisk(alert, { level });
  });
}

function boot() {
  installAmbientOrb();
  installClickSparkles();
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}

// Expose a small integration surface for plain-script consumers; ES module
// imports remain the preferred path for React/Vite.
window.FinGuardEffects = Object.freeze({
  notifyFraudRisk,
  handleTransactionScanResponse,
});
