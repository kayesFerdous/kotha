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

/* Waveform shaping. These numbers are the whole feel of the thing.

   The level arrives as linear RMS, and loudness is heard in decibels, so that
   is the scale it is drawn on. Measured through the M2's built-in microphone,
   2026-09-14, in the same 33 ms windows Rust sends: a quiet room sits at -46
   to -42 dBFS and never rose above -38; ordinary speech is roughly -35 to
   -15. The curve this replaces (RMS × 3.6, to the power 0.85) drew -30 dBFS
   at 16% of a bar — about two pixels — so the wave looked dead while the user
   talked, and the first obvious motion was the `thinking` animation after
   they stopped.

   QUIET         dBFS drawn as a resting dot. Just above that room's loudest
                 silence, so an empty room does not twitch.
   LOUD          dBFS drawn at full height. Raised speech reaches it; ordinary
                 speech lands around half to three quarters.
   FLOOR_*       The quiet end follows the room. A fan or a café lifts the
                 floor by FLOOR_RISE dB a frame (1 dB a second), and anything
                 quieter pulls it straight back down, which the gaps between
                 syllables do all the time. FLOOR_MAX stops a long loud
                 sentence dragging the floor up into the speech itself.
   RELEASE       How much of the previous height survives into the next
                 frame. This is what turns thirty discrete samples a second
                 into a wave that falls away instead of flickering. Attack is
                 instant: a syllable hits its full height on the frame it
                 arrives, and anything slower reads as lag. */
const QUIET = -40;
const LOUD = -12;
const FLOOR_RISE = 1 / 30;
const FLOOR_MARGIN = 4;
const FLOOR_MAX = -30;
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
/* The room's noise, in dBFS. Lives for the life of the page, so a second
   dictation in the same room starts already adapted. */
let floor = QUIET - FLOOR_MARGIN;

function shape(level) {
  const db = 20 * Math.log10(Math.max(level, 1e-6));
  const next = db < floor ? db : floor + FLOOR_RISE;
  floor = Math.min(FLOOR_MAX, Math.max(QUIET - FLOOR_MARGIN, next));
  const quiet = floor + FLOOR_MARGIN;
  return Math.min(1, Math.max(0, (db - quiet) / (LOUD - quiet)));
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
  const prev = root.dataset.state;
  root.dataset.state = next;
  label.textContent =
    { idle: "", listening: "Listening", thinking: "Transcribing", done: "Done" }[next];

  /* A new dictation starts from a flat line. Coming back from `thinking` is
     not a new dictation — the model was decoding one sentence while the user
     spoke the next, and levels kept arriving the whole time (Rust emits them
     from the microphone, not from the decode loop). Wiping the history here
     would throw away the last two thirds of a second of real speech and make
     the wave jump from a flat line to full height. */
  if (next === "listening" && prev !== "thinking") drain();
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
