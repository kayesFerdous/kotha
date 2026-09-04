/* ==========================================================================
   Kotha — the pill's behaviour.

   THE ENTIRE RUST↔UI CONTRACT IS TWO EVENTS
   -----------------------------------------
   Rust emits, the UI listens. Nothing goes the other way, and the UI never
   calls into Rust. That is deliberate: it keeps this whole directory
   replaceable without touching a line of the app, and it is why mock.js can
   stand in for the entire backend in a browser tab.

     emit("kotha://state", "idle" | "listening" | "thinking" | "done")
     emit("kotha://level", <number 0..1>)          // ~30 per second

   `level` is a plain RMS of the last audio frame, unshaped. All the curve
   fitting that makes it look good lives in shape() below, so tuning the
   waveform never means recompiling Rust.

   WHAT THIS FILE IS ALLOWED TO DO
   -------------------------------
   Set `data-state` on <html>, and write `--v` on each bar. That is all. Every
   transition, colour and easing is in pill.css, so a visual change is a CSS
   change. If you find yourself animating from here, put it back in the
   stylesheet.

   THE ONE PIECE OF STATE THE UI OWNS
   ----------------------------------
   How long the tick stays up after `done` before the pill leaves. That is
   presentation timing, not application state, so Rust does not send an `idle`
   after a `done` — see DONE_DWELL.
   ========================================================================== */

const STATES = ["idle", "listening", "thinking", "done"];

/** How long the tick lingers before the pill fades out, in ms. */
const DONE_DWELL = 1100;

/* Waveform shaping. These four numbers are the whole feel of the thing.

   GAIN/CURVE  Speech RMS sits low — a normal voice is maybe 0.05–0.2, and a
               linear mapping leaves the pill looking dead. The curve lifts
               quiet speech much more than loud, which is roughly how hearing
               works anyway.
   ATTACK      1.0 = a syllable hits its full height on the frame it arrives.
               Anything less reads as laggy.
   RELEASE     How much of the previous height survives into the next frame.
               This is what turns thirty discrete samples a second into a
               wave that falls away instead of flickering. */
const GAIN = 3.6;
const CURVE = 0.85;
const RELEASE = 0.80;

const root = document.documentElement;
const wave = document.querySelector(".wave");
const label = document.querySelector(".sr");

/* Bars are generated rather than written into the HTML so that --bar-count in
   pill.css stays the single place the number lives. */
const count = Number(getComputedStyle(root).getPropertyValue("--bar-count")) || 21;
const bars = Array.from({ length: count }, (_, i) => {
  const el = document.createElement("span");
  el.className = "bar";
  el.style.setProperty("--i", i);   // used by the `thinking` keyframe delay
  el.style.setProperty("--v", 0);
  wave.append(el);
  return el;
});

/* The waveform is a scrolling history, not a spectrum: index 0 is the oldest
   sample and the newest enters at the right. Every bar is a real measurement
   that really happened, which is the only reason it is honest to draw twenty
   one of them from a single number per frame. */
const history = new Array(count).fill(0);
let held = 0;
let doneTimer = null;

function shape(level) {
  return Math.min(1, Math.pow(Math.max(0, level) * GAIN, CURVE));
}

/** One microphone frame. Called ~30 times a second while listening. */
function push(level) {
  const v = shape(level);
  // Fast attack, slow release: jump straight up, ease back down.
  held = v > held ? v : held * RELEASE + v * (1 - RELEASE);

  history.shift();
  history.push(held);
  for (let i = 0; i < count; i++) bars[i].style.setProperty("--v", history[i].toFixed(3));
}

function drain() {
  history.fill(0);
  held = 0;
  for (const b of bars) b.style.setProperty("--v", 0);
}

function setState(next) {
  if (!STATES.includes(next)) {
    console.warn(`kotha: unknown state ${next}`);
    return;
  }
  clearTimeout(doneTimer);
  root.dataset.state = next;
  label.textContent =
    { idle: "", listening: "Listening", thinking: "Transcribing", done: "Done" }[next];

  if (next === "listening") drain();
  // The pill leaves on its own after a `done`; Rust does not have to say so.
  if (next === "done") doneTimer = setTimeout(() => setState("idle"), DONE_DWELL);
}

/* --------------------------------------------------------------------------
   Wiring
   --------------------------------------------------------------------------
   Inside Tauri, listen for the two events. Outside it, mock.js finds these on
   window and drives them instead — same code path, no branches in the UI. */

window.kotha = { setState, push, drain, STATES };

if (window.__TAURI__) {
  const { listen } = window.__TAURI__.event;
  listen("kotha://state", (e) => setState(e.payload));
  listen("kotha://level", (e) => push(e.payload));
}

setState("idle");
