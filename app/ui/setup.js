/* ==========================================================================
   Kotha — the first-run window's behaviour.

   THE CONTRACT, WHICH IS ONE EVENT AND ONE CALL
   ---------------------------------------------
   The pill's contract is one-way: Rust emits, the UI listens, and the UI never
   calls into Rust (see the header of pill.js — that stays true). This window is
   the single exception in the app, and it exists for exactly one reason:
   778 MB should not leave without somebody pressing a button.

     invoke("start_download")                       // the button, and only it
     listen("kotha://download", { done, total })    // progress, ~1 per MB
     listen("kotha://download", { error })          // it went wrong, in words

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

function failed(message) {
  phase("downloading");
  progress.textContent = message;
  progress.classList.add("bad");
}

function onProgress({ done, total, error }) {
  if (error) failed(error);
  else show(done, total);
}

button.addEventListener("click", () => {
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
  start = () => invoke("start_download");
} else {
  /* ----------------------------------------------------------------- mock */
  console.info("kotha: no backend, faking the download");

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
