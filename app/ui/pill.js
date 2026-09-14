/* ==========================================================================
   Kotha — the pill's behaviour.

   THE ENTIRE RUST↔UI CONTRACT IS TWO EVENTS
   -----------------------------------------
   Rust emits, the UI listens. Nothing goes the other way, and the UI never
   calls into Rust. That is deliberate: it keeps this whole directory
   replaceable without touching a line of the app, and it is why mock.js can
   stand in for the entire backend in a browser tab.

     emit("kotha://state", "idle" | "listening" | "thinking" | "done" | "error")
     emit("kotha://level", <number 0..1>)          // 30 per second, always
     emit("kotha://theme", "system" | "dark" | "light")

   `theme` arrives on every change and again just before the pill is shown.
   Twice, because the pill is hidden between dictations and this page cannot
   ask — the contract is one-way, so being told again at show time is what
   replaces a read. It costs one event per dictation and removes the whole
   class of "the pill is the wrong colour until you restart". Resolving what
   `system` means is theme.js's job, not this file's.

   `level` is a plain RMS of the last 1/30 s of audio, unshaped. All the
   curve fitting that makes it look good lives in shape() and push() below,
   so tuning the glyph never means recompiling Rust. Levels keep coming in
   every state the microphone is open for — `thinking` included, because the
   user may still be talking while the model decodes what they said a moment
   ago.

   There is no "armed" and no "paused" state, and there should not be: a live
   microphone in a silent room leaves five bars resting at --rest, which
   already says both. Rust would have to guess a threshold to send them, and
   the UI already has the number that threshold would be guessed from.

   WHAT THIS FILE IS ALLOWED TO DO
   -------------------------------
   Set `data-state` on <html>, and write `--b` on each of the five bars. That
   is all. Every transition, colour and easing is in pill.css, so a visual
   change is a CSS change. If you find yourself animating from here, put it
   back in the stylesheet.

   THE ONE PIECE OF STATE THE UI OWNS
   ----------------------------------
   How long `done` and `error` stay up before the pill leaves. That is
   presentation timing, not application state, so Rust does not send an `idle`
   after either — see DWELL.
   ========================================================================== */

const STATES = ["idle", "listening", "thinking", "done", "error"];

/** How long a terminal state lingers before the capsule collapses, in ms.
    `error` holds longer because a broken glyph is a thing to notice, and the
    user may not have been looking at the pill when it broke. */
const DWELL = { done: 900, error: 2200 };

/* Level shaping. These numbers are the whole feel of the thing.

   The level arrives as linear RMS, and loudness is heard in decibels, so that
   is the scale it is drawn on. Measured through the M2's built-in microphone,
   2026-09-14, in the same 33 ms windows Rust sends: a quiet room sits at -46
   to -42 dBFS and never rose above -38; ordinary speech is roughly -35 to
   -15. A curve without this calibration put -30 dBFS at 16% of full — so the
   glyph looked dead while the user talked, and the first obvious motion was
   the decode chase after they stopped.

   QUIET         dBFS drawn as an unlit glyph. Just above that room's loudest
                 silence, so an empty room does not twitch.
   LOUD          dBFS drawn at full brightness. Raised speech reaches it;
                 ordinary speech lands around half to three quarters.
   FLOOR_*       The quiet end follows the room. A fan or a café lifts the
                 floor by FLOOR_RISE dB a frame (1 dB a second), and anything
                 quieter pulls it straight back down, which the gaps between
                 syllables do all the time. FLOOR_MAX stops a long loud
                 sentence dragging the floor up into the speech itself.
   RELEASE       How much of the previous brightness survives into the next
                 frame. This is what turns thirty discrete samples a second
                 into a light that falls away instead of strobing. Attack is
                 instant: a syllable hits full brightness on the frame it
                 arrives, and anything slower reads as lag. */
const QUIET = -40;
const LOUD = -12;
const FLOOR_RISE = 1 / 30;
const FLOOR_MARGIN = 4;
const FLOOR_MAX = -30;
const RELEASE = 0.80;

/* How the one level number becomes five brightnesses.

   The glyph fills from the middle outward, so each bar is offset by how far
   it sits from the centre: the middle one starts lighting immediately, the
   inner pair once the voice is past REACH, the outer pair past two REACH.
   RAMP is how much louder again it takes that bar to reach full.

   Why outward from the middle, rather than left to right: left to right is a
   meter, and a meter invites you to read a value off it. This is not a
   measurement anyone needs — it exists so the user can tell at a glance that
   the microphone is hearing them. Symmetry has no scale to read, so the eye
   takes it in and lets go. It also keeps the decode chase, which does run
   left to right, unmistakably a different thing. */
const DISTANCE = [2, 1, 0, 1, 2];
const REACH = 0.26;
const RAMP = 0.34;

const root = document.documentElement;
const segs = [...document.querySelectorAll(".seg")];
const label = document.querySelector(".sr");

/* How long a bar takes to dim, read once from the stylesheet so the number
   still lives in exactly one place. Read once and not per frame on purpose:
   getComputedStyle forces a style flush, and doing that thirty times a second
   to look up a constant would cost more than the easing it configures. */
const FALL = getComputedStyle(root).getPropertyValue("--fall").trim() || "90ms";

let held = 0;
let wasRising = false;
let dwellTimer = null;
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

/** One microphone frame. Called ~30 times a second while the mic is open. */
function push(level) {
  const v = shape(level);
  const rising = v > held;
  // Fast attack, slow release: jump straight to full, ease back down.
  held = rising ? v : held * RELEASE + v * (1 - RELEASE);

  /* The other half of that, and the half CSS cannot express on its own: a
     transition eases in both directions or neither, so the duration has to be
     switched from here. Brightening lands on the frame it happens; dimming
     keeps --fall.

     Written on <html> rather than on each bar, because custom properties
     inherit — one write covers all five. Only on the frames the direction
     actually flips, because writing an inherited custom property invalidates
     style for everything below it, and speech flips direction perhaps a few
     times a second while this function runs thirty. */
  if (rising !== wasRising) {
    root.style.setProperty("--ease", rising ? "0ms" : FALL);
    wasRising = rising;
  }

  for (let i = 0; i < segs.length; i++) {
    const b = (held - DISTANCE[i] * REACH) / RAMP;
    segs[i].style.setProperty("--b", Math.min(1, Math.max(0, b)).toFixed(3));
  }
}

function drain() {
  held = 0;
  for (const s of segs) s.style.setProperty("--b", 0);
}

function setState(next) {
  if (!STATES.includes(next)) {
    console.warn(`kotha: unknown state ${next}`);
    return;
  }
  clearTimeout(dwellTimer);
  const prev = root.dataset.state;
  root.dataset.state = next;
  label.textContent = {
    idle: "",
    listening: "Listening",
    thinking: "Transcribing",
    done: "Done",
    error: "Dictation failed",
  }[next];

  /* A new dictation starts dark. Coming back from `thinking` is not a new
     dictation — the model was decoding one sentence while the user spoke the
     next, and levels kept arriving the whole time (Rust emits them from the
     microphone, not from the decode loop). Resetting the release envelope
     here would blink the glyph off in the middle of a word. */
  if (next === "listening" && prev !== "thinking") drain();
  // The pill leaves on its own after a terminal state; Rust need not say so.
  if (DWELL[next]) dwellTimer = setTimeout(() => setState("idle"), DWELL[next]);
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
  listen("kotha://theme", (e) => kothaTheme(e.payload));
}

setState("idle");
