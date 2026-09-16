/* ==========================================================================
   Kotha — the first-run window's behaviour.

   THE CONTRACT, WHICH IS ONE EVENT AND FOUR CALLS
   -----------------------------------------------
   The pill's contract is one-way: Rust emits, the UI listens, and the UI never
   calls into Rust (see the header of pill.js — that stays true). This window is
   the single exception in the app, and it exists for exactly one reason:
   778 MB should not leave without somebody pressing a button.

     invoke("start_download")                       // the button, and only it
     invoke("hotkey_label") -> "F9"                  // or null if nothing bound
     invoke("settings_get") -> { paste, pasteModes, ... }   // the second question
     invoke("settings_set", { key, value })          // the answer to it
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
const outputBox = document.getElementById("output");
const pasteBox = document.getElementById("paste");

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

/* The two answers the Ready panel is written from. Filled in before it shows,
   and `paste` again whenever the user flips the choice below it. */
let hotkey = null;
let paste = "copy";

/**
 * What the "Ready" panel tells the user to do. The whole sentence, because the
 * versions of it are not the same sentence with a word swapped.
 *
 * Naming a key that is not bound would be worse than naming none: pressing it
 * is the very first thing a new user does with this window.
 *
 * The last clause tracks the text output, and has to. "The text lands where
 * your cursor is" was written here unconditionally while the shipped default
 * was the clipboard, so the first thing this window did was make a promise the
 * app would not keep.
 */
function ready() {
  const kbd = (k) => `<kbd>${k}</kbd>`;
  const lands =
    paste === "copy"
      ? `The text goes to your clipboard, ready to paste.`
      : `The text lands where your cursor is.`;
  how.innerHTML = hotkey
    ? `Press ${hotkey.split("+").map(kbd).join(" + ")} anywhere, say something, and
       press it again. ${lands}`
    : `Another application already has Kotha's hotkey, so there is nothing to
       press yet — pick a different one under <b>Hotkey</b> in the tray menu.
       Until then, the tray icon's <b>Dictate</b> starts one.`;
}

/* The route this window recommends. `paste` and not `portal`: it is in every
   platform's list, and it is the one that does not raise a permission dialog
   before the user has dictated anything. */
const RECOMMENDED = "paste";

/**
 * The text-output group — this window's second question, and its only control.
 *
 * Preselected on "Paste at the cursor", and the preselection is *saved* rather
 * than only drawn. A recommendation the app does not act on is not a
 * recommendation, and leaving it unsaved is exactly what sent every new user's
 * first dictation to the clipboard without a word about it.
 *
 * Saving it hands nothing away on its own. Synthetic input is still gated:
 * macOS has to be told to allow Accessibility, and Linux's portal raises its
 * own dialog. That gate is the second of the two opt-ins; this is the first,
 * and the point is that it is now asked rather than assumed.
 *
 * A user who deliberately chose the clipboard keeps it — `pasteChosen` is what
 * separates an answer from the default, which is why Rust sends it.
 */
async function output(s) {
  paste = s.paste;

  // KOTHA_PASTE decided already. The settings window disables the group and
  // captions it; here there is nothing useful to show, so it stays hidden.
  if (s.pasteForced) return ready();

  if (!s.pasteChosen && paste !== RECOMMENDED) {
    try {
      await savePaste(RECOMMENDED);
      paste = RECOMMENDED;
    } catch (e) {
      // Not fatal: the window still works and the group below shows what the
      // app is actually doing. Loud, because a silent failure here is a first
      // dictation that goes somewhere the user was not told about.
      console.error("kotha: could not save the text output", e);
    }
  }

  pasteBox.replaceChildren(
    ...s.pasteModes.map(([value, label]) => {
      const row = document.createElement("label");
      row.className = "opt";

      const input = document.createElement("input");
      input.type = "radio";
      input.name = "paste";
      input.value = value;
      // The property, not the attribute: this is the live selection.
      input.checked = value === paste;

      const name = document.createElement("span");
      name.className = "opt-name";
      name.textContent = label;

      row.append(input, name);
      return row;
    })
  );

  outputBox.hidden = false;
  ready();
}

/* One listener on the container rather than one per radio, as the settings
   window does it. The tick is only adopted once the save has landed, so the
   group never shows a route the app is not on. */
pasteBox.addEventListener("change", async (e) => {
  const chosen = e.target.value;
  const previous = paste;
  try {
    await savePaste(chosen);
    paste = chosen;
  } catch (err) {
    console.error("kotha: could not save the text output", err);
    const back = pasteBox.querySelector(`input[value="${previous}"]`);
    if (back) back.checked = true;
  }
  ready();
});

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
let readSettings;
let savePaste;

if (window.__TAURI__) {
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  listen("kotha://download", (e) => onProgress(e.payload));
  start = () => invoke("start_download", { why: window.__DIAG });
  readSettings = () => invoke("settings_get");
  savePaste = (value) => invoke("settings_set", { key: "paste", value });

  invoke("hotkey_label").then((key) => {
    hotkey = key;
    ready();
  });
} else {
  /* ----------------------------------------------------------------- mock */
  console.info("kotha: no backend, faking the download");

  hotkey = "F9";
  ready();

  // Shaped like a Linux backend, as the rest of this mock is: three routes,
  // and nobody has answered yet. Flip `pasteChosen` to see the window leave a
  // deliberate clipboard choice alone.
  const fake = {
    paste: "copy",
    pasteModes: [
      ["copy", "Clipboard only"],
      ["paste", "Paste at the cursor"],
      ["portal", "Paste at the cursor (portal)"],
    ],
    pasteChosen: false,
    pasteForced: false,
  };
  readSettings = async () => ({ ...fake });
  savePaste = async (value) => {
    fake.paste = value;
    fake.pasteChosen = true;
  };

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

/* The Ready panel's second answer, fetched once and bringing the group with
   it. The hotkey arrives on its own call above; both write into `ready()`. */
readSettings()
  .then(output)
  .catch((e) => console.error("kotha: settings unavailable", e));
