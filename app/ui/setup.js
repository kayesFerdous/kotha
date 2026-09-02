/* ==========================================================================
   Kotha — the first-run window's behaviour.

   THE CONTRACT, WHICH IS ONE EVENT AND TWO CALLS
   ----------------------------------------------
   The pill's contract is one-way: Rust emits, the UI listens, and the UI never
   calls into Rust (see the header of pill.js — that stays true). This window is
   the single exception in the app, and it exists for exactly one reason:
   778 MB should not leave without somebody pressing a button.

     invoke("start_download")                       // the button, and only it
     invoke("hotkey_label") -> "F9"                  // or null if nothing bound
     listen("kotha://download", { done, total })    // progress, ~1 per MB
     listen("kotha://download", { error })          // it went wrong, in words

   The second call is asked, not pushed, because the answer is only wanted once
   and asking has no race — an event emitted while this file was still parsing
   would simply be missed.

   One event with two shapes rather than two events, because the window only
   ever asks one question of it: is this still going?

   WHAT THIS FILE IS ALLOWED TO DO
   -------------------------------
   Set `data-phase` on <body>, write `--p` on the bar, and put text in
   #progress. Everything else — which panel is visible, how the bar fills — is
   in pill.css under `body.setup`.

   MOCK
   ----
   The bottom of this file notices there is no Tauri and fakes the download, so
   app/ui/setup.html opens in a browser and runs. Same trade as mock.js: it
   ships inside the app and stands down at runtime, which is cheaper than a
   build step that exists only to delete it.
   ========================================================================== */

const body = document.body;
const meter = document.querySelector(".meter");
const progress = document.getElementById("progress");
const button = document.getElementById("go");
const how = document.getElementById("how");

const MB = 1024 * 1024;

/** The one piece of state: which panel is up. */
function phase(name) {
  body.dataset.phase = name;
}

/** Progress, in the two forms a person actually reads: a bar and megabytes. */
function show(done, total) {
  phase(done >= total ? "ready" : "downloading");
  meter.style.setProperty("--p", total ? done / total : 0);
  progress.textContent =
    `${Math.round(done / MB)} of ${Math.round(total / MB)} MB`;
}

/**
 * What the "Ready" panel tells the user to do. The whole sentence, because the
 * two versions of it are not the same sentence with a word swapped.
 *
 * Naming a key that is not bound would be worse than naming none: pressing it
 * is the very first thing a new user does with this window.
 */
function ready(key) {
  const kbd = (k) => `<kbd>${k}</kbd>`;
  how.innerHTML = key
    ? `Press ${key.split("+").map(kbd).join(" + ")} anywhere, say something, and
       press it again. The text lands where your cursor is.`
    : `Another application already has Kotha's hotkey, so there is nothing to
       press yet — pick a different one under <b>Hotkey</b> in the tray menu.
       Until then, the tray icon's <b>Dictate</b> starts one.`;
}

function failed(message) {
  phase("downloading");
  progress.textContent = message;
  progress.classList.add("bad");
}

function onProgress({ done, total, error }) {
  if (error) failed(error);
  else show(done, total);
}

button.addEventListener("click", (e) => {
  window.__DIAG = JSON.stringify({
    trusted: e.isTrusted, detail: e.detail, type: e.type,
    x: e.clientX, y: e.clientY, pointerId: e.pointerId,
    active: document.activeElement && document.activeElement.id,
    at: Math.round(performance.now()),
  });
  button.disabled = true;
  phase("downloading");
  progress.classList.remove("bad");
  start();
});

/* ------------------------------------------------------------------ Tauri */

let start;

if (window.__TAURI__) {
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  listen("kotha://download", (e) => onProgress(e.payload));
  invoke("hotkey_label").then(ready);
  start = () => invoke("start_download", { why: window.__DIAG });
} else {
  /* ----------------------------------------------------------------- mock */
  console.info("kotha: no backend, faking the download");

  ready("F9");
  const TOTAL = 778 * MB;
  start = () => {
    let done = 0;
    const tick = setInterval(() => {
      done = Math.min(TOTAL, done + 30 * MB);
      onProgress({ done, total: TOTAL });
      if (done >= TOTAL) clearInterval(tick);
    }, 120);
  };
}
